import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import einops
from .helpers import BASE_LOG, undo_normalization

__all__ = ["get_loss"]


def get_loss(name, **kwargs):
    if name == "direct":
        return DirectLoss(**kwargs)

    elif name == "selfsup":
        return SelfSupLoss(**kwargs)

    else:
        raise ValueError(f"Unkown loss={name}.")


class Loss(nn.Module):
    """Base class for compression losses."""

    def __init__(self, beta=1, **kwargs):
        super().__init__()
        self.beta = beta

    def forward(self, Y_hat, targets, coding_loss):
        n_z = Y_hat.size(0)

        distortion_loss, logs = self.get_distortion(Y_hat, targets)

        # loss. shape: [n_z_samples, batch_size]
        loss = distortion_loss + self.beta * coding_loss

        # tightens bound using IWAE: log 1/k sum exp(loss). shape: [batch_size]
        if n_z > 1:
            tight_loss = torch.logsumexp(loss, 0) - math.log(n_z)
        else:
            tight_loss = loss.squeeze(0)

        # E_x[loss]. shape: []
        tight_loss = tight_loss.mean(0)

        logs.update(
            dict(
                loose_loss=loss.mean() / math.log(BASE_LOG),
                loss=tight_loss / math.log(BASE_LOG),
            )
        )

        return tight_loss, logs

    def get_distortion(self, Y_hat, targets):
        raise NotImplementedError()


class DirectLoss(Loss):
    """Computes the loss using the direct variational bound.
    
    Parameters
    ----------
    is_img_out : bool, optional
        Whether the model is predicting an image.

    dataset : str, optional
        Name of the dataset, used to undo normalization.

    is_classification : str, optional
        Wether you should perform classification instead of regression. It is not used if 
        `is_img_out=True`, in which cross entropy is used only if the image is black and white.

    kwargs :
        Additional arguments 
    """

    def __init__(
        self, is_img_out=False, dataset=None, is_classification=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.is_img_out = is_img_out
        self.dataset = dataset
        self.is_classification = is_classification

    def get_distortion(self, Y_hat, targets):
        n_z = Y_hat.size(0)

        # -log p(yi|zi). shape: [n_z_samples, batch_size, *y_shape]
        #! all of the following should really be written in a single line using log_prob where P_{Y|Z}
        # is an actual conditional distribution (categorical if cross entropy, and gaussian for mse),
        # but this might be less understandable for usual deep learning + less numberically stable
        Y_hat = einops.rearrange(Y_hat, "z b ... -> (z b) ...")
        targets = einops.repeat(targets, "b ... -> (z b) ...", z=n_z)
        if not self.is_img_out:
            if self.is_classification:
                neg_log_q_ylz = F.cross_entropy(Y_hat, targets, reduction="none")
            else:
                neg_log_q_ylz = F.mse_loss(Y_hat, targets, reduction="none")
        else:
            if targets.shape[-3] == 1:
                # black white image => uses categorical distribution, with logits for stability
                neg_log_q_ylz = F.binary_cross_entropy_with_logits(
                    Y_hat, targets, reduction="none"
                )
            elif targets.shape[-3] == 3:
                # this is just to ensure that images are in [0,1] and compared in unormalized space
                # black white doesn't need because uses logits
                Y_hat, targets = undo_normalization(Y_hat, targets, self.dataset)
                # color image => uses gaussian distribution
                neg_log_q_ylz = F.mse_loss(Y_hat, targets, reduction="none")
            else:
                raise ValueError(
                    f"shape={targets.shape} does not seem to come from an image"
                )

        # -log p(y|z). shape: [n_z_samples, batch_size]
        #! mathematically should take a sum (log prod proba -> sum log proba), but usually people take mean
        neg_log_q_ylz = einops.reduce(
            neg_log_q_ylz, "(z b) ... -> z b", reduction="sum", z=n_z
        )

        logs = dict(H_q_YlZ=neg_log_q_ylz.mean() / math.log(BASE_LOG))

        return neg_log_q_ylz, logs


# TODO
# SelfSupLoss
