import torch
import math
import compressai
import einops
from .helpers import atleast_ndim, BASE_LOG, kl_divergence
from .distributions import get_marginalDist


def get_coder(name, channels, p_ZlX, **kwargs):
    """Return the correct entropy coder."""

    if name == "entropy":
        return EntropyBottleneck(channels, **kwargs)
    if "MI_" in name:

        q_Z = get_marginalDist(
            kwargs.pop("prior_fam"), p_ZlX, **(kwargs.pop("prior_kwargs"))
        )
        return MIBottleneck(q_Z, **kwargs)
    else:
        raise ValueError(f"Unkown loss={name}.")


class Coder:
    """Base class for a coder, i.e. a model that perform entropy/MI coding."""

    is_can_compress = True  # wether can compress
    is_need_optimizer = False  # wether needs an additional optimizer

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    # returning the loss is necessary to work with `EntropyBottleneck`
    def forward(self, z_samples, p_Zlx):
        """Performs the compression and returns loss.

        Parameters
        ----------
        z_samples : torch.Tensor shape=[n_z_dim, batch_shape, z_dim]
            Representation to compress.

        p_Zlx : torch.Distribution
            Encoder which should be used to perform compression.
        
        Returns
        -------
        z_samples_hat : torch.Tensor shape=[n_z_dim, batch_shape, z_dim]
            Representation after compression.

        coding_loss : torch.Tensor shape=[n_z_dim, batch_shape]
            Loss that is incurred for the current representation

        coding_logs : dict
            Additional values to log.
        """
        raise NotImplementedError()

    def update(self, force):
        """Updates the coder before evaluation and checkpointing."""
        raise NotImplementedError()

    def compress(self, z):
        """Performs the actual compression. Note that this cannot contrain n_z as first dim."""
        raise NotImplementedError()

    def decompress(self, z_compressed):
        """Performs the actual decompression."""
        raise NotImplementedError()


class EntropyBottleneck(compressai.entropy_models.EntropyBottleneck, Coder):
    """
    Model that codes using the (approximate) entropy H[Z]. 

    Notes
    -----
    - The encoder `p_Zlx` should be deterministic here (i.e. Delta distribution).
    - THe additional optimization is only needed for compression and not during training (because
    during training uses uniform noise)
    """

    is_need_optimizer = True  # wether needs an additional IS_NEED_OPTIMIZER

    def forward(self, z, _):
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

        # - log q(z). shape :  [n_z_dim, batch_shape]
        neg_log_q_z = -torch.log(q_z).sum(-1)

        logs = dict(H_q_Z=neg_log_q_z.mean() / math.log(BASE_LOG), H_ZlX=0)

        return z_hat, neg_log_q_z, logs

    def compress(self, z):
        z = einops.rearrange(z, "b (c e1 e2) -> b c e1 e2", e1=1, e2=1)
        return super().compress(z)

    def decompress(self, z_compressed, shape):
        z_compressed = super().decompress(z_compressed, [1, 1])
        z_compressed = einops.rearrange(
            z_compressed, "b c e1 e2 -> b (c e1 e2)", e1=1, e2=1
        )
        return z_compressed


class MIBottleneck(torch.nn.Module, Coder):
    """
    Model that codes using the (approximate) mutual information I[Z,X]. 

    Notes
    -----
    - This is always correct, but in the deterministic case I[Z,X] = H[Z] so it's easier to use
    `EntropyBottleneck`.

    Parameters
    ----------
    q_Z : nn.Module
        Prior to use for compression
    """

    is_can_compress = False  # wether can compress

    def __init__(self, q_Z):
        super().__init__()
        self.q_Z = q_Z

    def update(self, force):
        pass

    def forward(self, z, p_Zlx):
        # batch shape: [] ; event shape: [z_dim]
        q_Z = self.q_Z()  # conditioning on nothing to get the prior

        # E_x[KL[p(Z|x) || q(z)]]. shape: [n_z_samples, batch_size]
        kl = kl_divergence(p_Zlx, q_Z, z_samples=z, is_reduce=False)

        z_hat = z

        logs = dict(
            I_q_ZX=kl.mean() / math.log(BASE_LOG),
            H_ZlX=p_Zlx.entropy().mean(0) / math.log(BASE_LOG),
        )
        # upper bound on H[Z] (cross entropy)
        logs["H_q_Z"] = logs["I_q_ZX"] + logs["H_ZlX"]

        return z_hat, kl, logs

