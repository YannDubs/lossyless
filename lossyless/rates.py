import io
import logging
import math
from contextlib import closing

import numpy as np

import compressai
import einops
import torch
from compressai.entropy_models import EntropyBottleneck as CompressaiEntropyBottleneck
from compressai.entropy_models.entropy_models import (
    EntropyModel,
    GaussianConditional,
    _EntropyCoder,
    default_entropy_coder,
)
from compressai.models.utils import update_registered_buffers

from .architectures import MLP
from .distributions import get_marginalDist
from .helpers import (
    BASE_LOG,
    OrderedSet,
    Timer,
    atleast_ndim,
    kl_divergence,
    mean,
    to_numpy,
    weights_init,
)

logger = logging.getLogger(__name__)
__all__ = ["get_rate_estimator"]

### HELPERS ###
def get_rate_estimator(mode, z_dim=None, p_ZlX=None, **kwargs):
    """Return the correct entropy coder."""
    if mode == "lossless":
        return Lossless(z_dim)

    elif mode == "H_factorized":
        return HRateFactorizedPrior(z_dim, **kwargs)

    elif mode == "H_hyper":
        return HRateHyperprior(z_dim, **kwargs)

    elif mode == "H_spatial":
        return HRateHyperpriorSpatial(z_dim, p_ZlX, **kwargs)

    elif mode == "MI":
        q_Z = get_marginalDist(
            kwargs.pop("prior_fam"), p_ZlX, **(kwargs.pop("prior_kwargs"))
        )
        return MIRate(z_dim, q_Z, **kwargs)

    raise ValueError(f"Unkown rate estimator={mode}.")


# only difference is that we trreat all as a single vector, while in compressai treat as a small image
# with channels.Each channel in compressai (and thus each point in our vector) is treated differently
# but in compressai they "share" computations across spatial
class EntropyBottleneck(CompressaiEntropyBottleneck):
    def forward(self, z):
        # entropy bottleneck takes 4 dim as inputs (as if images, where dim is channel)
        z = einops.rearrange(z, "b (c e1 e2) -> b c e1 e2", e1=1, e2=1)
        z_hat, q_z = super().forward(z)
        z_hat = einops.rearrange(z_hat, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1)
        q_z = einops.rearrange(q_z, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1)
        return z_hat, q_z

    def compress(self, z):
        z = atleast_ndim(z, 4)
        return super().compress(z)

    def decompress(self, z_strings):
        z_hat = super().decompress(z_strings, [1, 1])
        z_hat = einops.rearrange(z_hat, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1)
        return z_hat


### BASE ###
class RateEstimator(torch.nn.Module):
    """Base class for a coder, i.e. a model that perform entropy/MI coding.

    Parameters
    ----------
    z_dim : int, optional
        Dimensionality of the representation.

    warmup_k_epoch : int, optional
        Number of epochs not to backprop through the featurizer => warming up of the rate estimator.
        The output will be z instead of z_hat, but the estimator will be warmed up by trying to
        reconstruct z.

    is_endToEnd : bool, optional
        Whether to train all the model in an end to end fashion. If not then the rate will not be
        backproped through the featurizer (you will just try to reconstruct Z without too much distortion).
    """

    is_can_compress = False  # wether can compress

    def __init__(self, z_dim, warmup_k_epoch=0, is_endToEnd=True):
        super().__init__()
        self.z_dim = z_dim
        self.warmup_k_epoch = warmup_k_epoch
        self.is_endToEnd = is_endToEnd

    def reset_parameters(self):
        weights_init(self)

    @torch.cuda.amp.autocast(False)  # precision here is important
    def forward(self, z, p_Zlx, parent=None):
        """Performs the approx compression and returns loss.

        Parameters
        ----------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation to compress.

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoder which should be used to perform compression.

        parent : LearnableCompressor, optional
            Parent module. This is useful for some rates if they need access to other parts of the
            model.

        Returns
        -------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation after compression.

        rates : torch.Tensor shape=[batch_shape]
            Theoretical number of bits (rate) needed for compression.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        z_hat, rates, r_logs, r_other = self.forward_help(z, p_Zlx, parent)

        if (not self.is_endToEnd) or (parent.current_epoch < self.warmup_k_epoch):
            # during disjoint training z_hat will still be returned but the rate is computed
            # without backpropagating throught the encoder => only train rate_estimator parameters

            # make sure not changing featurizer. *0 to make sure pytorch does not complain about no grad
            z_detached = z.detach() + z * 0
            p_Zlx_detached = p_Zlx.detach(is_grad_flow=True)

            _, rates, *_ = self.forward_help(z_detached, p_Zlx_detached, parent)

        return z_hat, rates, r_logs, r_other

    def forward_help(self, z, p_Zlx, parent=None):
        """Performs the approx compression and returns loss.

        Parameters
        ----------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation to compress.

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoder which should be used to perform compression.

        parent : LearnableCompressor, optional
            Parent module. This is useful for some rates if they need access to other parts of the
            model.

        Returns
        -------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation after compression.

        rates : torch.Tensor shape=[batch_shape]
            Theoretical number of bits (rate) needed for compression.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        raise NotImplementedError()

    def compress(self, z, parent=None):
        """Performs the actual compression.

        Parameters
        ----------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation to compress. 

        parent : LearnableCompressor, optional
            Parent module. This is useful for some rates if they need access to other parts of the
            model.

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
            Approx. decompressed representation. 
        """
        raise NotImplementedError()

    def real_rate(self, z, is_return_logs=False, parent=None):
        """Compute actual number of bits (rate), necessary for encoding z.

        Parameters
        ----------
        z : torch.Tensor shape=[batch_shape, z_dim]
            Representation to compress.

        is_return_logs : bool, optional
            Whether to return a dictionnary to log in addition to n_bits.

        parent : LearnableCompressor, optional
            Parent module. This is useful for some rates if they need access to other parts of the
            model.

        Returns
        -------
        n_bits : int
            Average bits per image

        if is_return_logs:
            logs : dict
                Additional values that can be useful to log.
        """
        batch, z_dim = z.shape

        with Timer() as compress_timer:
            all_strings = self.compress(z, parent=parent)

        if is_return_logs:
            with Timer() as decompress_timer:
                _ = self.decompress(all_strings)

        # sum over all latents (for hierachical). mean over batch.
        n_bytes = sum(mean([len(s) for s in strings]) for strings in all_strings)
        n_bits = n_bytes * 8

        if is_return_logs:
            logs = dict(
                compress_time=compress_timer.duration / batch,
                receiver_time=decompress_timer.duration / batch,
                n_bits=n_bits,
            )
            return n_bits, logs

        return n_bits

    def aux_loss(self, force):
        """Auxilary loss for the rate estimator / coders. This will be called separately from main loss."""
        raise NotImplementedError()

    def aux_parameters(self):
        """
        Returns an iterator over the model parameters that should be trained by auxilary optimizer.
        These are all the parameters of the prior.
        """
        raise NotImplementedError()

    def make_pickable_(self):
        """Ensure that the Estimator is pickable, which is necessary for distributed training."""
        for m in self.modules():  # recursive iteration over all
            if isinstance(m, EntropyModel):
                m.entropy_coder = None

    def undo_pickable_(self):
        """Undo `make_pickable_`, e.g. ensures that the coder is available."""
        for m in self.modules():
            if isinstance(m, EntropyModel):
                # TODO : allow resetting non default coder
                m.entropy_coder = _EntropyCoder(default_entropy_coder())

    def update(self, force=False):
        """Updates the entropy model values. Needs to be called once after training to be able to
        later perform the evaluation with an actual entropy coder.

        Parameters
        ----------
        force : bool, optional
            Overwrite previous values.
        """
        updated = True

        for m in self.children():  # recursive iteration over all
            if isinstance(m, CompressaiEntropyBottleneck):
                rv = m.update(force=force)
                updated &= rv
            elif isinstance(m, GaussianConditional):
                scale_table = get_scale_table()
                rv = m.update_scale_table(scale_table, force=force)
                updated &= rv
        return updated

    def prepare_compressor_(self):
        """Ensure that the model can be used for compression"""

        # make sure that the coder is available
        self.undo_pickable_()

        # mae sure that the parameters for compressing are available
        self.update(force=True)


### No Compresssion CODERS ###
class Lossless(RateEstimator):
    """Model that performs lossless comrpession of representations."""

    def forward_help(self, z, _, parent=None):
        batch_size, z_dim = z.shape
        z_hat = z

        with closing(io.BytesIO()) as f:
            np.savez_compressed(f, to_numpy(z_hat))
            bit_rate = f.getbuffer().nbytes * 8 / (batch_size)

        nats_rate = bit_rate * math.log(2)
        # ensure that doesn't complain that no grad and put correct shape
        # note that we only compute average and not acatually per example memory usage
        # shape: [batch_size]
        rates = (
            nats_rate + z.mean(dim=-1) * 0
        )  # if bit_rate is large and 16 floating point might give inf

        # in bits
        logs = dict()
        other = dict()

        return z_hat, rates, logs, other

    def aux_parameters(self):
        return iter(())  # no parameters


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

    kwargs :
        Additional arguemnts to `RateEstimator`.
    """

    def __init__(self, z_dim, q_Z, **kwargs):
        super().__init__(z_dim, **kwargs)
        self.q_Z = q_Z
        self.reset_parameters()

    def forward_help(self, z, p_Zlx, parent=None):
        # batch shape: [] ; event shape: [z_dim]
        q_Z = self.q_Z()

        # E_x[KL[p(Z|x) || q(Z)]]. shape: [batch_size]
        kl = kl_divergence(p_Zlx, q_Z, z_samples=z)

        z_hat = z

        logs = dict(
            I_q_ZX=kl.mean() / math.log(BASE_LOG),
            H_ZlX=p_Zlx.entropy().mean(0) / math.log(BASE_LOG),
        )
        # upper bound on H[Z] (cross entropy)
        logs["H_q_Z"] = logs["I_q_ZX"] + logs["H_ZlX"]
        other = dict()

        return z_hat, kl, logs, other

    def aux_parameters(self):
        return iter(())  # no parameters


### ENTROPY CODERS. Min H[Z] ###
# all of the following assume that `p_Zlx` should be deterministic here (i.e. Delta distribution).
# minor differences from and credits to them https://github.com/InterDigitalInc/CompressAI/blob/edd62b822186d81903c4a53c3f9b0806e9d7f388/compressai/models/priors.py
# a little messy for reshaping as compressai assumes 4D image as inputs (but we compress 2D vectors)
class HRateEstimator(RateEstimator):
    """Base class for a coder, i.e. a model that perform entropy/MI coding.

    Parameters
    ----------
    z_dim : int
        Size of the representation.

    kwargs_ent_bottleneck : dict, optional
        Additional arguments to `EntropyBottleneck`.

    invertible_processing : {None, "psd", "diag"}, optional
        Wether to apply an invertible linear layer before the bottleneck and then undo it (apply inverse).
        This is especially important when the encoder is pretrained, indeed the entropy bottleneck 
        adds  noise in [-0.5,0.5] to z and treats all dimensions as independent. THis is theoretically 
        fine as the previous layer can learn to perform all processing that is needed such that those 
        2 (arbitrary) conventions are met. But in practice this can (1) make learning harder for 
        the distortion; (2) cannot be changed when the encoder is pretrained. "diag" applies a
        diagonal matrix. "psd" applies a positive definite matrix (better but a little slower).
        None doesn't apply anything.

    kwargs :
        Additional arguments to `RateEstimator`
    """

    is_can_compress = True

    def __init__(
        self, z_dim, kwargs_ent_bottleneck={}, invertible_processing="diag", **kwargs,
    ):
        self.invertible_processing = invertible_processing
        super().__init__(z_dim, **kwargs)

        self.kwargs_ent_bottleneck = kwargs_ent_bottleneck

        if self.invertible_processing == "diag":
            self.scaling = torch.nn.Parameter(torch.ones(z_dim))
            self.biasing = torch.nn.Parameter(torch.zeros(z_dim))
        elif self.invertible_processing == "psd":
            self.scaling = torch.nn.Parameter(torch.randn(z_dim, z_dim))
            self.biasing = torch.nn.Parameter(torch.zeros(z_dim))
            self.register_buffer("eye", torch.eye(z_dim), persistent=False)
        elif self.invertible_processing is None:
            pass
        else:
            raise ValueError(
                f"Unkown invertible_processing={self.invertible_processing}."
            )

    aux_loss = compressai.models.CompressionModel.aux_loss

    def reset_parameters(self):
        weights_init(self)

        if self.invertible_processing == "diag":
            self.scaling = torch.nn.Parameter(torch.ones(self.z_dim))
            self.biasing = torch.nn.Parameter(torch.zeros(self.z_dim))
        elif self.invertible_processing == "psd":
            self.scaling = torch.nn.Parameter(torch.randn(self.z_dim, self.z_dim))
            self.biasing = torch.nn.Parameter(torch.zeros(self.z_dim))

    def process_z_in(self, z):
        z = z.float()
        kwargs = dict()

        # TODO remove flag and only keep the simple "diag" if work well
        if self.invertible_processing == "diag":
            z = (z + self.biasing) * self.scaling.exp()

        elif self.invertible_processing == "psd":
            mat = torch.mm(self.scaling, self.scaling.T) + 1e-1 * self.eye
            z = z + self.biasing
            z = torch.matmul(z, mat)

            kwargs["mat"] = mat

        return z, kwargs

    def process_z_out(self, z_hat, mat=None):
        if self.invertible_processing == "diag":
            z_hat = (z_hat / self.scaling.exp()) - self.biasing

        elif self.invertible_processing == "psd":
            if mat is None:
                mat = torch.mm(self.scaling, self.scaling.T) + 1e-1 * self.eye
            chol = torch.cholesky(mat)
            z_hat = torch.cholesky_solve(z_hat.T, chol).T
            z_hat = z_hat - self.biasing

        return z_hat

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        try:
            # Dynamically update the entropy bottleneck buffers related to the CDFs
            policy = "resize"  # resize when loading  (even if already called "update")
            update_registered_buffers(
                self.entropy_bottleneck,
                f"{prefix}entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
                policy=policy,
            )
        except KeyError:
            # if the model is set to featurizer then nothing is in state_dict
            pass

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def aux_parameters(self):
        # all parameters of the CDF
        return OrderedSet(
            p for n, p in self.named_parameters() if n.endswith(".quantiles")
        )

    @property
    def is_coder_updated(self):
        """Whether the coder is updated => can be used."""
        is_coder_updated = True

        for m in self.modules():  # recursive iteration over all
            if isinstance(m, EntropyModel):
                is_coder_updated &= m._offset.numel() > 0

        return is_coder_updated

    @property
    def is_coder_present(self):
        """Whether the coder is present => can be used."""
        is_coder_present = True

        for m in self.modules():  # recursive iteration over all
            if isinstance(m, EntropyModel):
                is_coder_present &= m.entropy_coder is not None

        return is_coder_present

    @property
    def is_compute_real_rate(self):
        """Whether compute the real rate."""
        return (not self.training) and self.is_coder_updated and self.is_coder_present


class HRateFactorizedPrior(HRateEstimator):
    """
    Model that codes using the (approximate) entropy H[Z]. Factorized prior in [1].

    Parameters
    ----------
    z_dim : int
        Size of the representation..

    kwargs :
        Additional arguments to `HRateEstimator`

    References
    ----------
    [1] Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv
    preprint arXiv:1802.01436 (2018).
    """

    def __init__(self, z_dim, **kwargs):
        super().__init__(z_dim, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(
            self.z_dim, **self.kwargs_ent_bottleneck
        )
        self.reset_parameters()

    def forward_help(self, z, _, parent=None):

        z_in, kwargs = self.process_z_in(z)

        z_hat, q_z = self.entropy_bottleneck(z_in)

        # - log q(z). shape :  [batch_shape]
        neg_log_q_z = -torch.log(q_z).sum(-1)

        logs = dict(H_q_Z=neg_log_q_z.mean() / math.log(BASE_LOG), H_ZlX=0)

        if self.is_compute_real_rate:
            n_bits, logs2 = self.real_rate(z, is_return_logs=True)
            logs.update(logs2)
            logs["n_bits"] = n_bits

        other = dict()

        z_hat = self.process_z_out(z_hat, **kwargs)

        return z_hat, neg_log_q_z, logs, other

    def compress(self, z, parent=None):
        z_in, _ = self.process_z_in(z)
        # list for generality when hyperprior
        return [self.entropy_bottleneck.compress(z_in)]

    def decompress(self, all_strings):
        assert isinstance(all_strings, list) and len(all_strings) == 1
        z_hat = self.entropy_bottleneck.decompress(all_strings[0])
        return self.process_z_out(z_hat)


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class HRateHyperprior(HRateEstimator):
    """
    Model that codes using the (approximate) entropy H[Z]. Scale hyperprior in [1].

    Parameters
    ----------
    z_dim : int
        Size of the representation.

    factor_dim : int, optional
        By how much to decrease the dimensionality of the side information.

    side_z_dim : int, optional
        SIze of the side representations, if give will overwrite `factor_dim`.

    is_pred_mean : bool, optional
        Whether to learn the mean of the gaussian for the side information (as in mean scale hyper
        prior of [2]).

    kwargs :
        Additional arguments to `HRateEstimator`

    References
    ----------
    [1] Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv
    preprint arXiv:1802.01436 (2018).
    [2] Minnen, David, Johannes Ballé, and George D. Toderici. "Joint autoregressive and hierarchical
    priors for learned image compression." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(
        self, z_dim, factor_dim=5, side_z_dim=None, is_pred_mean=True, **kwargs,
    ):
        super().__init__(z_dim, **kwargs)

        if side_z_dim is None:
            side_z_dim = self.z_dim // factor_dim

        self.side_z_dim = side_z_dim
        self.is_pred_mean = is_pred_mean
        self.entropy_bottleneck = EntropyBottleneck(
            side_z_dim, **self.kwargs_ent_bottleneck
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.side_encoder, self.z_encoder = self.get_encoders()
        self.reset_parameters()

    def get_encoders(self):
        kwargs_mlp = dict(n_hid_layers=2, hid_dim=max(self.z_dim, 256))

        side_encoder = MLP(self.z_dim, self.side_z_dim, **kwargs_mlp)

        z_dim = self.z_dim
        if self.is_pred_mean:
            z_dim *= 2  # predicting mean and var

        z_encoder = MLP(self.side_z_dim, z_dim, **kwargs_mlp)

        return side_encoder, z_encoder

    def chunk_params(self, gaussian_params):
        if self.is_pred_mean:
            scales_hat, means_hat = gaussian_params.chunk(2, -1)
        else:
            scales_hat, means_hat = gaussian_params, None
        return scales_hat, means_hat

    def forward_help(self, z, _, __):
        z_in, kwargs = self.process_z_in(z)

        # shape: [ batch_shape, side_z_dim]
        side_z = self.side_encoder(z_in)
        side_z_hat, q_s = self.entropy_bottleneck(side_z)

        # scales_hat and means_hat (if not None). shape: [batch_shape, z_dim]
        gaussian_params = self.z_encoder(side_z_hat)
        scales_hat, means_hat = self.chunk_params(gaussian_params)

        # shape: [ batch_shape, z_dim]
        z_hat, q_zls = self.gaussian_conditional(z_in, scales_hat, means=means_hat)

        # - log q(s). shape :  [batch_shape]
        neg_log_q_s = -torch.log(q_s).sum(-1)

        # - log q(z|s). shape :  [batch_shape]
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

        if self.is_compute_real_rate:
            n_bits, logs2 = self.real_rate(z, is_return_logs=True)
            logs.update(logs2)
            logs["n_bits"] = n_bits

        other = dict()

        z_hat = self.process_z_out(z_hat, **kwargs)

        return z_hat, neg_log_q_zs, logs, other

    def get_indexes_means_hat(self, side_z_strings):

        # side_z and side_z_hat. shape: [batch_shape, side_z_dim]
        side_z_hat = self.entropy_bottleneck.decompress(side_z_strings)

        # shape: [batch_shape, z_dim]
        gaussian_params = self.z_encoder(side_z_hat)

        # z, scales_hat, means_hat, indexes. shape: [batch_shape, z_dim, 1, 1]
        scales_hat, means_hat = self.chunk_params(gaussian_params)
        scales_hat = atleast_ndim(scales_hat, 4)
        means_hat = atleast_ndim(scales_hat, 4)
        means_hat = atleast_ndim(scales_hat, 4)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        return indexes, means_hat

    def compress(self, z, parent=None):
        z_in, _ = self.process_z_in(z)

        # side_z and side_z_hat. shape: [batch_shape, side_z_dim]
        side_z = self.side_encoder(z_in)
        # list of bytes
        side_z_strings = self.entropy_bottleneck.compress(side_z)

        # shape: [batch_shape, z_dim, 1, 1]
        indexes, means_hat = self.get_indexes_means_hat(side_z_strings)
        z_in = atleast_ndim(z_in, 4)

        z_strings = self.gaussian_conditional.compress(z_in, indexes, means=means_hat)
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
        return self.process_z_out(z_hat)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        try:
            # Dynamically update the entropy bottleneck buffers related to the CDFs
            policy = "resize"  # resize when loading  (even if already called "update")
            update_registered_buffers(
                self.gaussian_conditional,
                f"{prefix}gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
                policy=policy,
            )
        except KeyError:
            # if the model is set to featurizer then nothing is in state_dict
            pass

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


# Should really have allowed Z to be a tensor instead of a vector (in all parts of the code)
class HRateHyperpriorSpatial(HRateHyperprior):
    """HRateHyperprior but treats the spatial dimensions separately. THis only makes sense if the representation
    can be treated spatially, which is the case with `BALLE` architecture. CUrrently only works
    with square spaces.

    Parameters
    ----------
    z_dim : int
        Size of the representation as given. The representation is seen as a vector that should be 
        flattened to (n_channels,-1,-1) where -1 is sqrt(z_dim // n_channels).

    P_ZlX : CondDist, optional
        Conditional distribution of the encoder. This should have a parameters 
        `P_ZlX.mapper.channel_out_dim` that gives the number of channels of the latent z.

    kwargs :
        Additional arguments to HRateHyperprior.
    """

    def __init__(
        self, z_dim, P_ZlX, **kwargs,
    ):
        self.raw_z_dim = z_dim
        self.n_channels = P_ZlX.mapper.channel_out_dim
        self.side_dim = int(math.sqrt(z_dim / self.n_channels))

        super().__init__(self.n_channels, **kwargs)

    def forward(self, z, p_Zlx, parent=None):
        z = einops.rearrange(
            z,
            "b (c h w) -> (b h w) c",
            h=self.side_dim,
            w=self.side_dim,
            c=self.n_channels,
        )
        z_hat, rates, r_logs, r_other = super().forward(z, p_Zlx, parent)

        z_hat = einops.rearrange(
            z_hat,
            "(b h w) c -> b (c h w)",
            h=self.side_dim,
            w=self.side_dim,
            c=self.n_channels,
        )
        rates = einops.reduce(
            rates, "(b h w) -> b", "sum", h=self.side_dim, w=self.side_dim
        )

        # before you were taking a mean over a batch that is larger than the real one => simply multiply
        r_logs = {k: v * self.side_dim ** 2 for k, v in r_logs.items()}

        return z_hat, rates, r_logs, r_other
