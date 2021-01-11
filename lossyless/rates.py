import torch
import math
import compressai
from compressai.entropy_models import GaussianConditional, EntropyModel
from compressai.models.utils import update_registered_buffers
import einops
from .helpers import atleast_ndim, BASE_LOG, kl_divergence, weights_init, mean
from .distributions import get_marginalDist
from .architectures import MLP


__all__ = ["get_rate_estimator"]

### HELPERS ###
def get_rate_estimator(name, z_dim=None, p_ZlX=None, n_z_samples=None, **kwargs):
    """Return the correct entropy coder."""
    if "H_" in name:
        if "fact" in name:
            return HRateFactorizedPrior(z_dim, n_z_samples=n_z_samples, **kwargs)
        elif "hyper" in name:
            return HRateHyperprior(z_dim, n_z_samples=n_z_samples, **kwargs)

    elif "MI_" in name:
        q_Z = get_marginalDist(
            kwargs.pop("prior_fam"), p_ZlX, **(kwargs.pop("prior_kwargs"))
        )
        return MIRate(q_Z, **kwargs)

    raise ValueError(f"Unkown rate estimator={name}.")


# only difference is that works with flatten z (it needs 4D tensor not 2d)
class EntropyBottleneck(compressai.entropy_models.EntropyBottleneck):
    def forward(self, z):
        # entropy bottleneck takes 4 dim as inputs (as if images, where dim is channel)
        n_z = z.size(0)
        z = einops.rearrange(z, "n_z b (c e1 e2) -> (n_z b) c e1 e2", e1=1, e2=1)
        z_hat, q_z = super().forward(z)
        z_hat = einops.rearrange(
            z_hat, "(n_z b) c e1 e2 -> n_z b (c e1 e2)", n_z=n_z, e1=1, e2=1
        )
        q_z = einops.rearrange(
            q_z, "(n_z b) c e1 e2 -> n_z b (c e1 e2)", n_z=n_z, e1=1, e2=1
        )
        return z_hat, q_z

    def compress(self, z):
        z = atleast_ndim(z, 4)
        return super().compress(z)

    def decompress(self, z_strings):
        z_hat = super().decompress(z_strings, [1, 1])
        z_hat = einops.rearrange(z_hat, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1)
        return z_hat


### BASE ###
class RateEstimator(compressai.models.CompressionModel):
    """Base class for a coder, i.e. a model that perform entropy/MI coding."""

    is_can_compress = False  # wether can compress

    def __init__(self):
        # directly call nn.Module because you don't want to call the constructor of `CompressionModel`
        torch.nn.Module.__init__(self)

    def reset_parameters(self):
        weights_init(self)

    # returning the loss is necessary to work with `EntropyBottleneck`
    def forward(self, z, p_Zlx):
        """Performs the approx compression and returns loss.

        Parameters
        ----------
        z : torch.Tensor shape=[n_z_dim, batch_shape, z_dim]
            Representation to compress.

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoder which should be used to perform compression.

        Returns
        -------
        z : torch.Tensor shape=[n_z_dim, batch_shape, z_dim]
            Representation after compression.

        rates : torch.Tensor shape=[n_z_dim, batch_shape]
            Theoretical number of bits (rate) needed for compression.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        raise NotImplementedError()

    def compress(self, z):
        """Performs the actual compression.

        Parameters
        ----------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation to compress. Note that there's no n_z_dim!

        Returns
        -------
        all_strings : list (len n_hyper) of list (len batch_shape) of bytes
            Compressed representations in bytes. The first list (of len n_latents) contains the
            representations for each hyper latents
        """
        raise NotImplementedError()

    def decompress(self, all_strings):
        """Performs the actual decompression.

        Parameters
        ----------
        all_strings : list (len n_hyper) of list (len batch_shape) of bytes
            Compressed representations in bytes. The first list (of len n_latents) contains the
            representations for each hyper latents

        Returns
        -------
        z_hat : torch.Tensor shape=[batch_shape, z_dim]
            Approx. decompressed representation. Note that there's no n_z_dim!
        """
        raise NotImplementedError()

    def real_rate(self, z):
        """Compute actual number of bits (rate), necessary for encoding z.

        Parameters
        ----------
        z : torch.Tensor shape=[n_z_dim, batch_shape, z_dim]
            Representation to compress. Note that there's n_z_dim!
        """
        z = einops.rearrange(z, "n_z b d -> (n_z b) d", n_z=z.size(0))
        all_strings = self.compress(z)
        # sum over all latents (for hierachical). mean over batch and n_z.
        n_bytes = sum(mean([len(s) for s in strings]) for strings in all_strings)
        return n_bytes * 8  # bits


### MUTUAL INFORMATION CODERS. Min I[X,Z] ###


class MIRate(RateEstimator):
    """
    Model that codes using the (approximate) mutual information I[Z,X].

    Notes
    -----
    - This is always correct, but in the deterministic case I[Z,X] = H[Z] so it's easier to use an
    `HCoder`.

    Parameters
    ----------
    q_Z : nn.Module
        Prior to use for compression
    """

    def __init__(self, q_Z):
        super().__init__()
        self.q_Z = q_Z
        self.reset_parameters()

    def update(self, force):
        pass

    def forward(self, z, p_Zlx):
        # batch shape: [] ; event shape: [z_dim]
        q_Z = self.q_Z()

        # E_x[KL[p(Z|x) || q(Z)]]. shape: [n_z_samples, batch_size]
        kl = kl_divergence(p_Zlx, q_Z, z_samples=z, is_reduce=False)

        z_hat = z

        logs = dict(
            I_q_ZX=kl.mean() / math.log(BASE_LOG),
            H_ZlX=p_Zlx.entropy().mean(0) / math.log(BASE_LOG),
        )
        # upper bound on H[Z] (cross entropy)
        logs["H_q_Z"] = logs["I_q_ZX"] + logs["H_ZlX"]
        other = dict()

        return z_hat, kl, logs, other


### ENTROPY CODERS. Min I[X,Z] ###
# all of the following assume that `p_Zlx` should be deterministic here (i.e. Delta distribution).
# minor differences from and credits to them https://github.com/InterDigitalInc/CompressAI/blob/edd62b822186d81903c4a53c3f9b0806e9d7f388/compressai/models/priors.py
# a little messy for reshaping as compressai assumes 4D image as inputs (but we compress 2D vectors)


class HRateFactorizedPrior(RateEstimator):
    """
    Model that codes using the (approximate) entropy H[Z]. Factorized prior in [1].

    Parameters
    ----------
    z_dim : int
        Size of the representation.

    n_z_samples : int, optional
        Number of z samples. Currently if > 1 cannot perform actual compress.

    kwargs:
        Additional arguments to `EntropyBottleneck`.

    References
    ----------
    [1] Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv
    preprint arXiv:1802.01436 (2018).
    """

    def __init__(self, z_dim, n_z_samples=1, **kwargs):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(z_dim, **kwargs)
        self.is_can_compress = n_z_samples == 1

    def forward(self, z, _):

        z_hat, q_z = self.entropy_bottleneck(z)

        # - log q(z). shape :  [n_z_dim, batch_shape]
        neg_log_q_z = -torch.log(q_z).sum(-1)

        logs = dict(H_q_Z=neg_log_q_z.mean() / math.log(BASE_LOG), H_ZlX=0)

        if not self.training:
            logs["n_bits"] = self.real_rate(z)

        other = dict()

        return z_hat, neg_log_q_z, logs, other

    def compress(self, z):
        # list for generality when hyperprior
        return [self.entropy_bottleneck.compress(z)]

    def decompress(self, all_strings):
        assert isinstance(all_strings, list) and len(all_strings) == 1
        return self.entropy_bottleneck.decompress(all_strings[0])

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        z_dim = state_dict["entropy_bottleneck._matrices.0"].size(0)
        net = cls(z_dim)
        net.load_state_dict(state_dict)
        return net


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class HRateHyperprior(RateEstimator):
    """
    Model that codes using the (approximate) entropy H[Z]. Scale hyperprior in [1].

    Parameters
    ----------
    z_dim : int
        Size of the representation.

    factor_dim : int, optional
        By how much to decrease the dimensionality of the side information.

    is_pred_mean : bool, optional
        Whether to learn the mean of the gaussian for the side information (as in mean scale hyper
        prior of [2]).

    n_z_samples : int, optional
        Number of z samples. Currently if > 1 cannot perform actual compress.

    kwargs:
        Additional arguments to `EntropyBottleneck`.

    References
    ----------
    [1] Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv
    preprint arXiv:1802.01436 (2018).
    [2] Minnen, David, Johannes Ballé, and George D. Toderici. "Joint autoregressive and hierarchical
    priors for learned image compression." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, z_dim, factor_dim=5, is_pred_mean=True, n_z_samples=1, **kwargs):
        super().__init__()
        side_z_dim = z_dim // factor_dim

        self.is_can_compress = n_z_samples == 1
        self.is_pred_mean = is_pred_mean
        self.entropy_bottleneck = EntropyBottleneck(side_z_dim, **kwargs)
        self.gaussian_conditional = GaussianConditional(None)

        kwargs_mlp = dict(n_hid_layers=2, hid_dim=max(z_dim, 256))
        self.h_a = MLP(z_dim, side_z_dim, **kwargs_mlp)

        if self.is_pred_mean:
            z_dim *= 2  # predicting mean and var

        self.h_s = MLP(side_z_dim, z_dim, **kwargs_mlp)

    def chunk_params(self, gaussian_params):
        if self.is_pred_mean:
            scales_hat, means_hat = gaussian_params.chunk(2, -1)
        else:
            scales_hat, means_hat = gaussian_params, None
        return scales_hat, means_hat

    def forward(self, z, _):

        # shape: [n_z_dim, batch_shape, side_z_dim]
        side_z = self.h_a(z)
        side_z_hat, q_s = self.entropy_bottleneck(side_z)

        # scales_hat and means_hat (if not None). shape: [n_z_dim, batch_shape, z_dim]
        gaussian_params = self.h_s(side_z_hat)
        scales_hat, means_hat = self.chunk_params(gaussian_params)

        # shape: [n_z_dim, batch_shape, z_dim]
        z_hat, q_zls = self.gaussian_conditional(z, scales_hat, means=means_hat)

        # - log q(s). shape :  [n_z_dim, batch_shape]
        neg_log_q_s = -torch.log(q_s).sum(-1)

        # - log q(z|s). shape :  [n_z_dim, batch_shape]
        neg_log_q_zls = -torch.log(q_zls).sum(-1)

        # - log q(z,s)
        neg_log_q_zs = neg_log_q_s + neg_log_q_zls

        logs = dict(
            H_q_ZlS=neg_log_q_zls.mean() / math.log(BASE_LOG),
            # this in reality is H_q_ZS but calling H_q_Z for better comparison (think that Z'=Z,S)
            H_q_Z=neg_log_q_zs.mean() / math.log(BASE_LOG),
            H_q_S=neg_log_q_s.mean() / math.log(BASE_LOG),
            H_ZlX=0,
        )

        if not self.training:
            logs["n_bits"] = self.real_rate(z)

        other = dict()

        return z_hat, neg_log_q_zs, logs, other

    def get_indexes_means_hat(self, side_z_strings):

        # side_z and side_z_hat. shape: [batch_shape, side_z_dim]
        side_z_hat = self.entropy_bottleneck.decompress(side_z_strings)

        # shape: [batch_shape, z_dim]
        gaussian_params = self.h_s(side_z_hat)

        # z, scales_hat, means_hat, indexes. shape: [batch_shape, z_dim, 1, 1]
        scales_hat, means_hat = self.chunk_params(gaussian_params)
        scales_hat = atleast_ndim(scales_hat, 4)
        means_hat = atleast_ndim(scales_hat, 4)
        means_hat = atleast_ndim(scales_hat, 4)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        return indexes, means_hat

    def compress(self, z):

        # side_z and side_z_hat. shape: [batch_shape, side_z_dim]
        side_z = self.h_a(z)
        # len n_z_dim list of bytes
        side_z_strings = self.entropy_bottleneck.compress(side_z)

        # shape: [batch_shape, z_dim, 1, 1]
        indexes, means_hat = self.get_indexes_means_hat(side_z_strings)
        z = atleast_ndim(z, 4)

        z_strings = self.gaussian_conditional.compress(z, indexes, means=means_hat)
        return [z_strings, side_z_strings]

    def decompress(self, all_strings):
        assert isinstance(all_strings, list) and len(all_strings) == 2
        z_strings, side_z_strings = all_strings
        # shape: [batch_shape, z_dim, 1, 1]
        indexes, means_hat = self.get_indexes_means_hat(side_z_strings)

        z_hat = self.gaussian_conditional.decompress(
            z_strings, indexes, means=means_hat
        )
        z_hat = einops.rearrange(z_hat, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1)
        return z_hat

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        z_dim = state_dict["h_a.module.0.weight"].size(1)
        factor_dim = z_dim // state_dict["h_a.module.6.weight"].size(0)
        is_pred_mean = state_dict["h_s.module.6.weight"].size(0) // 10 == 2

        net = cls(z_dim, factor_dim=factor_dim, is_pred_mean=is_pred_mean)
        net.load_state_dict(state_dict)
        return net

    def update(self, force=False, scale_table=None):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)
