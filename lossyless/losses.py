import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import einops
from .helpers import kl_divergence, BASE_LOG, undo_normalization

__all__ = ["get_loss"]


def get_loss(name, **kwargs):
    if name == "direct":
        return DirectLoss(**kwargs)

    elif name == "selfsup":
        return SelfSupLoss(**kwargs)

    else:
        raise ValueError(f"Unkown loss={name}.")


class DirectLoss(nn.Module):
    """Computes the loss using the direct variational bound.
    
    Parameters
    ----------
    beta : float, optional
        Factor by which to multiply the KL divergence.

    is_img_out : bool, optional
        Whether the model is predicting an image.

    dataset : str, optional
        Name of the dataset, used to undo normalization.
    """

    def __init__(self, beta=1, is_img_out=False, dataset=None):
        super().__init__()
        self.beta = beta
        self.is_img_out = is_img_out
        self.dataset = dataset

    def forward(self, Y_hat, targets, p_Zlx, q_Z, z_samples, entropy_coder):
        n_z = z_samples.size(0)

        # E_x[KL[p(Z|x) || q(z)]]. shape: [n_z_samples, batch_size]
        kl = kl_divergence(p_Zlx, q_Z, z_samples=z_samples, is_reduce=False)

        # breakpoint()

        # -log p(y|z). shape: [n_z_samples, batch_size]
        # make sure same size targets and pred
        Y_hat = einops.rearrange(Y_hat, "z b ... -> (z b) ...")
        targets = einops.repeat(targets, "b ... -> (z b) ...", z=n_z)
        if not self.is_img_out:
            n_log_p_ylz = F.cross_entropy(Y_hat, targets, reduction="none")
        else:
            if targets.shape[-3] == 1:
                # black white image => uses categorical distribution.
                n_log_p_ylz = F.binary_cross_entropy_with_logits(
                    Y_hat, targets, reduction="none"
                )
            elif targets.shape[-3] == 3:
                # this is just to ensure that images are in [0,1] and compared in unormalized space
                # black white doesn't need because no normalization, and uses `_with_logits` for stability
                Y_hat, targets = undo_normalization(Y_hat, targets, self.dataset)
                # color image => uses gaussian distribution
                n_log_p_ylz = F.mse_loss(Y_hat, targets, reduction="none")
            else:
                raise ValueError(
                    f"shape={targets.shape} does not seem to come from an image"
                )
        #! mathematically should take a sum (log prod proba -> sum log proba), but  usually people take mean
        n_log_p_ylz = einops.reduce(
            n_log_p_ylz, "(z b) ... -> z b", reduction="sum", z=n_z
        )

        # loss. shape: [n_z_samples, batch_size]
        loss = n_log_p_ylz + self.beta * kl

        # tightens bound using IWAE: log 1/k sum exp(loss). shape: [batch_size]
        if z_samples.size(0) > 1:
            tight_loss = torch.logsumexp(loss, 0) - math.log(n_z)
        else:
            tight_loss = loss.squeeze(0)

        # E_x[loss]. shape: [1]
        tight_loss = tight_loss.mean(0)

        logs = dict(
            loose_loss=loss.mean() / math.log(BASE_LOG),
            n_log_p_ylz=n_log_p_ylz.mean() / math.log(BASE_LOG),
            kl=kl.mean() / math.log(BASE_LOG),
            loss=tight_loss / math.log(BASE_LOG),
            H_ZlX=p_Zlx.entropy().mean(0) / math.log(BASE_LOG),
        )
        # upper bound on H[Z] (cross entropy)
        logs["H_q_Z"] = logs["kl"] + logs["H_ZlX"]

        return tight_loss, logs


# TODO
# SelfSupLoss
