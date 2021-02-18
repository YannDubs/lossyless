import math

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from .architectures import get_Architecture
from .distributions import Deterministic, DiagGaussian
from .helpers import (BASE_LOG, UnNormalizer, is_colored_img, kl_divergence,
                      mse_or_crossentropy_loss)

__all__ = ["get_distortion_estimator"]


def get_distortion_estimator(mode, p_ZlX, **kwargs):
    if mode == "direct":
        return DirectDistortion(**kwargs)

    elif mode == "contrastive":
        return ContrastiveDistortion(p_ZlX=p_ZlX, **kwargs)

    else:
        raise ValueError(f"Unkown disotrtion.mode={mode}.")


class DirectDistortion(nn.Module):
    """Computes the loss using an direct variational bound (i.e. trying to predict an other variable).

    Parameters
    ----------
    z_dim : int
        Dimensionality of the representation.

    y_shape : tuple of int or int
        Shape of Y.

    arch : str, optional
        Architecture of the decoder. See `get_Architecture`.

    arch_kwargs : dict, optional
        Additional parameters to `get_Architecture`.

    dataset : str, optional
        Name of the dataset, used to undo normalization.

    is_classification : str, optional
        Wether you should perform classification instead of regression. It is not used if
        `is_img_out=True`, in which cross entropy is used only if the image is black and white.

    n_classes_multilabel : list of int, optional
        In the multilabel multiclass case but with varying classes, the model cannot predict the
        correct shape as tensor (as each label have different associated target size) as a result it
        predicts everything in a single flattened predictions. `n_labels` is a list of the number
        of classes for each labels. This should only be given if the targets and predictions are
        flattened.

    is_sum_over_tasks : bool, optional
        Whether to sum all task loss rather than average.

    is_normalized : bool, optional
        Whether the data is normalized. This is important to know whether needs to be unormalized
        when comparing in case you are reconstructing the input. Currently only works for colored
        images.

    data_mode : {"image","distribution"}, optional      
        Mode of the data input.
    """

    def __init__(
        self,
        z_dim,
        y_shape,
        arch=None,
        arch_kwargs=dict(complexity=2),
        dataset=None,
        is_classification=True,
        n_classes_multilabel=None,
        is_sum_over_tasks=False,
        is_normalized=True,
        data_mode="image",
        name=None,  # in case you are directly using cfg of architecture. This is a placeholder
    ):
        super().__init__()
        self.dataset = dataset
        self.is_classification = is_classification
        self.is_img_out = data_mode == "image"

        if arch is None:
            arch = "cnn" if self.is_img_out else "mlp"
        Decoder = get_Architecture(arch, **arch_kwargs)
        self.q_YlZ = Decoder(z_dim, y_shape)
        self.n_classes_multilabel = n_classes_multilabel
        self.is_sum_over_tasks = is_sum_over_tasks
        self.is_normalized = is_normalized

        if self.is_normalized:
            if self.is_img_out:
                self.unnormalizer = UnNormalizer(self.dataset)
            else:
                raise NotImplementedError(
                    "Can curently only deal with normalized data if it's an image."
                )

    def forward(self, z_hat, aux_target, _):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[n_z, batch_size, z_dim]
            Reconstructed representations.

        aux_target : Tensor shape=[batch_size, *aux_shape]
            Targets to predict.

        Returns
        -------
        distortions : torch.Tensor shape=[n_z_dim, batch_shape]
            Estimates distortion.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """

        n_z = z_hat.size(0)

        flat_z_hat = einops.rearrange(z_hat, "n_z b d -> (n_z b) d")

        # shape: [n_z_samples*batch_size, *y_shape]
        Y_hat = self.q_YlZ(flat_z_hat)  # Y_hat is suff statistic of q_{Y|z}

        aux_target = einops.repeat(aux_target, "b ... -> (n_z b) ...", n_z=n_z)

        # -log p(yi|zi). shape: [n_z_samples, batch_size, *y_shape]
        #! all of the following should really be written in a single line using log_prob where P_{Y|Z}
        # is an actual conditional distribution (categorical if cross entropy, and gaussian for mse),
        # but this might be less understandable for usual deep learning + less numberically stable
        if self.is_img_out:
            if is_colored_img(aux_target):
                if self.is_normalized:
                    # compare in unormalized space
                    aux_target = self.unnormalizer(aux_target)

                # output is linear but data is in 0,1 => for a better comparison put in [-,1]
                Y_hat = torch.sigmoid(Y_hat)

                # color image => uses gaussian distribution
                neg_log_q_ylz = F.mse_loss(Y_hat, aux_target, reduction="none")
            else:
                # black white image => uses categorical distribution, with logits for stability
                neg_log_q_ylz = F.binary_cross_entropy_with_logits(
                    Y_hat, aux_target, reduction="none"
                )

                # but for saving you still want the image in [0,1]
                Y_hat = torch.sigmoid(Y_hat)

        elif self.n_classes_multilabel:
            assert self.is_classification
            cum_cls = 0
            neg_log_q_ylz = 0
            for i, n_classes in enumerate(self.n_classes_multilabel):
                cum_cls_new = cum_cls + n_classes
                neg_log_q_ylz = neg_log_q_ylz + F.cross_entropy(
                    Y_hat[:, cum_cls:cum_cls_new], aux_target[:, i], reduction="none"
                )
                cum_cls = cum_cls_new

            if not self.is_sum_over_tasks:
                n_tasks = len(self.n_classes_multilabel)
                neg_log_q_ylz = neg_log_q_ylz / n_tasks

        else:  # normal pred
            neg_log_q_ylz = mse_or_crossentropy_loss(
                Y_hat,
                aux_target,
                self.is_classification,
                is_sum_over_tasks=self.is_sum_over_tasks,
            )

        # -log p(y|z). shape: [n_z_samples, batch_size]
        #! mathematically should take a sum (log prod proba -> sum log proba), but usually people take mean
        neg_log_q_ylz = einops.reduce(
            neg_log_q_ylz, "(z b) ... -> z b", reduction="sum", z=n_z
        )

        # T for auxilary task to distinguish from task Y
        logs = dict(H_q_TlZ=neg_log_q_ylz.mean() / math.log(BASE_LOG))

        other = dict()
        # for plotting (note that they are already unormalized)
        other["Y_hat"] = Y_hat[0].detach().cpu()
        other["Y"] = aux_target[0].detach().cpu()

        return neg_log_q_ylz, logs, other


class ContrastiveDistortion(nn.Module):
    """Computes the loss using contrastive variational bound (i.e. with positive and negative examples).

    Notes
    -----
    - Only works for `Deterministic` (or tensors) or `DiagGaussian` distribution. In latter case uses a
    distributional infoNCE (i.e. with KL divergence).
    - Distributional InfoNCE is memory heavy because copy tensors. TODO: Use torch.as_strided
    - Never uses samples z_hat.
    - parts of code taken from https://github.com/lucidrains/contrastive-learner

    Parameters
    ----------
    p_ZlX : CondDist
        Instantiated conditional distribution. Used to represent all the other positives.

    temperature : float, optional
        Temperature scaling in InfoNCE.

    is_symmetric : bool, optional
        Whether to use symmetric logits in the case of probabilistic InfoNCE.

    weight : float, optional
        By how much to weight the denominator in InfoNCE. In [1] this corresponds to 1/alpha of
        reweighted CPC. We can show that this is still a valid lower bound if set to at max
        len(train_dataset) / (2*batch_size-1), and should thus be set to that.

    is_cosine : bool, optional
        Whether to use cosine similarity instead of dot products fot the logits of deterministic functions.
        This seems necessary for training, probably because if not norm of Z matters++ and then
        large loss in entropy bottleneck.

    is_invariant : bool, optional
        Want to be invariant => same orbit / positive element. If not then the positive is the element itself
        and the image on the same orbit is a negative (so batch is doubled size). Using augmented image
        as negatives is necesary to ensure the representations are not invariant. THis is used
        for nce (but not ince).

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self,
        p_ZlX,
        temperature=0.1,
        is_symmetric=True,
        is_cosine=True,
        weight=1,
        is_invariant=True,
    ):
        super().__init__()
        self.p_ZlX = p_ZlX
        self.temperature = temperature
        self.is_symmetric = is_symmetric
        self.is_cosine = is_cosine
        self.weight = weight
        self.is_invariant = is_invariant

    def forward(self, z_hat, x_pos, p_Zlx):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[n_z, batch_size, z_dim]
            Reconstructed representations.

        x_pos : Tensor shape=[batch_size, *x_shape]
            Other positive inputs., i.e., input on the same orbit.

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoded distribution of Z.

        Returns
        -------
        distortions : torch.Tensor shape=[n_z_dim, batch_shape]
            Estimates distortion.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        batch_size = z_hat.size(1)
        new_batch_size = 2 * batch_size
        device = z_hat.device

        # Distribution for positives. batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlp = self.p_ZlX(x_pos)

        # shape: [new_batch_size,new_batch_size]
        logits = self.compute_logits(p_Zlx, p_Zlp)
        logits /= self.temperature

        if self.is_invariant:
            mask = ~torch.eye(new_batch_size, device=device).bool()
            # select all but current.
            n_classes = new_batch_size - 1  # n classes in clf
            logits = logits[mask].view(new_batch_size, n_classes)

            # infoNCE is essentially the same as a softmax (where wights are other z => try to predict
            # index of positive z). The label is the index of the positive z, which for each z is the
            # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
            # you masked select all but the current z. arange takes care of idx of z which increases
            arange = torch.arange(batch_size, device=device)
            pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)
        else:
            n_classes = new_batch_size  # all examples ( do not remove current)
            # use NCE which is not invariant to transformations =>
            # the positive example is the example  itself
            pos_idx = torch.arange(new_batch_size, device=device)

        if self.weight != 1:
            # TODO CHECK correct (And without self.is_invariant )
            # you want to multiply \sum e(negative) in the denomiator by to_mult
            to_mult = (n_classes - (1 / self.weight)) / (n_classes - 1)
            # equivalent: add log(to_mult) to every negative logits
            # equivalent: add - log(to_mult) to positive logits
            to_add = -math.log(to_mult)
            to_add = to_add * torch.ones_like(logits[:, 0:1])  # correct shape
            logits.scatter_add_(1, pos_idx.unsqueeze(1), to_add)
            # when testing use `logits.gather(1, pos_idx.unsqueeze(1))` to see pos

        # I[Z,f(M(X))] = E[ log \frac{(N-1) exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log softmax(z^Tz_p) ]
        # = log(N-1) - E[ crossentropy(z^Tz, p) ]
        # = \hat{H}[f(M(X))] - \hat{H}[f(M(X))|Z]
        hat_H_m = math.log(self.weight * n_classes)
        hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="mean")

        logs = dict(I_q_zm=(hat_H_m - hat_H_mlz) / math.log(BASE_LOG))
        other = dict()

        return hat_H_mlz, logs, other

    def compute_logits(self, p_Zlx, p_Zlp):
        if isinstance(p_Zlx, DiagGaussian):
            # use probabilistic InfoNCE
            # not pytorch's kl divergence because want fast computation (kl between every and every element in batch)
            mu_x, sigma_x = p_Zlx.base_dist.loc, p_Zlx.base_dist.scale
            mu_p, sigma_p = p_Zlp.base_dist.loc, p_Zlp.base_dist.scale

            # shape: [2*batch_size, z_dim]
            mus = torch.cat([mu_x, mu_p], dim=0)
            sigmas = torch.cat([sigma_x, sigma_p], dim=0)

            N = mus.size(0)

            # TODO: Should use torch.as_strided
            # shape: [4*batch_size**2, z_dim]
            # repeat: [1,2,3] -> [1,2,3,1,2,3,1,2,3]
            r_mus = mus.repeat(N, 1)
            r_sigmas = sigmas.repeat(N, 1)
            # repeat interleave: [1,2,3] -> [1,1,1,2,2,2,3,3,3]
            i_mus = mus.repeat_interleave(N, dim=0)
            i_sigmas = sigmas.repeat_interleave(N, dim=0)

            # batch shape: [4*batch_size**2] event shape: [z_dim]
            r_gaus = DiagGaussian(r_mus, r_sigmas)
            i_gaus = DiagGaussian(i_mus, i_sigmas)

            # all possible pairs of KLs. shape: [2*batch_size, 2*batch_size]
            kls = kl_divergence(r_gaus, i_gaus).view(N, N)
            logits = -kls  # logits needs to be larger when more similar

            if self.is_symmetric:
                # equivalent to (kl(p||q) + kl(q||p))/2 for each pair
                logits = (logits + logits.T) / 2

        else:
            # if you are using a deterministic representation then use standard InfoNCE
            # because if not KL is none differentiable
            if isinstance(p_Zlx, Deterministic):
                # shape: [batch_size,z_dim]
                z = p_Zlx.base_dist.loc
                z_pos = p_Zlp.base_dist.loc
            else:
                # suppose outputs are tensors of shape: [n_z,batch_size,z_dim]
                z = p_Zlx.squeeze(0)
                z_pos = p_Zlp.squeeze(0)

            # shape: [2*batch_size,z_dim]
            zs = torch.cat([z, z_pos], dim=0)

            if self.is_cosine:
                zs = F.normalize(zs, dim=1, p=2)

            # shape: [2*batch_size,2*batch_size]
            logits = zs @ zs.T

        return logits
