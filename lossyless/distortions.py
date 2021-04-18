import logging
import math

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from .architectures import MLP, get_Architecture
from .distributions import Deterministic, DiagGaussian
from .helpers import (
    BASE_LOG,
    UnNormalizer,
    gather_from_gpus,
    is_colored_img,
    kl_divergence,
    prediction_loss,
    weights_init,
)

logger = logging.getLogger(__name__)

__all__ = ["get_distortion_estimator"]


def get_distortion_estimator(mode, p_ZlX, **kwargs):
    if mode == "direct":
        return DirectDistortion(**kwargs)

    elif mode == "contrastive":
        return ContrastiveDistortion(p_ZlX=p_ZlX, **kwargs)

    elif mode == "lossyZ":
        return LossyZDistortion(**kwargs)

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

    is_normalized : bool, optional
        Whether the data is normalized. This is important to know whether needs to be unormalized
        when comparing in case you are reconstructing the input. Currently only works for colored
        images.

    data_mode : {"image","distribution"}, optional
        Mode of the data input.

    kwargs :
        Additional arguments to `prediction_loss`.
    """

    def __init__(
        self,
        z_dim,
        y_shape,
        arch=None,
        arch_kwargs=dict(complexity=2),
        dataset=None,
        is_normalized=True,
        data_mode="image",
        name=None,  # in case you are directly using cfg of architecture. This is a placeholder
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.is_img_out = data_mode == "image"

        if arch is None:
            arch = "cnn" if self.is_img_out else "mlp"
        Decoder = get_Architecture(arch, **arch_kwargs)
        self.q_YlZ = Decoder(z_dim, y_shape)
        self.is_normalized = is_normalized
        self.kwargs = kwargs

        if self.is_normalized:
            if self.is_img_out:
                self.unnormalizer = UnNormalizer(self.dataset)
            else:
                raise NotImplementedError(
                    "Can curently only deal with normalized data if it's an image."
                )

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

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
        else:  # normal pred
            neg_log_q_ylz = prediction_loss(Y_hat, aux_target, **self.kwargs)

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
        Temperature scaling in InfoNCE. Recommended less than 1.

    is_train_temperature : bool, optional
        Whether to treat the temperature as a parameter. Uses the same sceme as CLIP.
        If true then `temperature` becomes the lower bound on temperature.

    is_symmetric : bool, optional
        Whether to use symmetric logits in the case of probabilistic InfoNCE.

    effective_batch_size : float, optional
        Effective batch size to use for estimating InfoNCE. Larger means that more variance but less bias,
        but if too large can become not a lower bound anymore. In [1] this is (m+1)/(2*alpha), where
        +1 and / 2 comes from the fact that talking about batch size rather than sample size.
        If `None` will use the standard unweighted `effective_batch_size`. Another good possibility
        is `effective_batch_size=len_dataset` which ensures that least bias while still lower bound.

    is_cosine : bool, optional
        Whether to use cosine similarity instead of dot products fot the logits of deterministic functions.
        This seems necessary for training, probably because if not norm of Z matters++ and then
        large loss in entropy bottleneck. Recommended True.

    is_invariant : bool, optional
        Want to be invariant => same orbit / positive element. If not then the positive is the element itself
        and the image on the same orbit is a negative (so batch is doubled size). Using augmented image
        as negatives is necesary to ensure the representations are not invariant. THis is used
        for nce (but not ince).

    is_already_featurized : bool, optional
        Whether the posivite examples are already featurized => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already featuized.

    is_batch_neg : bool, optional
        Whether to treat all the examples in the batch also as negatives. This double the nubmer
        of negatives and should thus be used when possible. CLIP treats images and text separately
        and thus does not do that.

    is_project : bool, optional
        Whether to use a porjection head. True seems to work better.

    project_kwargs : dict, optional
        Additional arguments to `Projector` in case `is_project`. Noe that is `out_shape` is <= 1
        it will be a percentage of z_dim.

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self,
        p_ZlX,
        temperature=0.01,
        is_train_temperature=True,
        is_symmetric=True,
        is_cosine=True,
        effective_batch_size=None,
        is_invariant=True,
        is_already_featurized=False,
        is_batch_neg=True,
        is_project=True,
        project_kwargs={"mode": "mlp", "out_shape": 128},
    ):
        super().__init__()
        self.p_ZlX = p_ZlX if not is_already_featurized else Deterministic
        self.temperature = temperature
        self.is_train_temperature = is_train_temperature
        self.is_symmetric = is_symmetric
        self.is_cosine = is_cosine
        self.effective_batch_size = effective_batch_size
        self.is_invariant = is_invariant
        self.is_batch_neg = is_batch_neg
        self.is_project = is_project
        if self.is_project:
            z_dim = self.p_ZlX.out_dim
            if project_kwargs["out_shape"] <= 1:
                project_kwargs["out_shape"] = max(
                    10, int(z_dim * project_kwargs["out_shape"])
                )

            Projector = get_Architecture(**project_kwargs)
            self.projector = Projector(z_dim)
        else:
            self.projector = torch.nn.Identity()

        if self.is_train_temperature:
            # Same initialization as clip
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

        if self.is_train_temperature:
            # Same initialization as clip
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

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
        n_z, batch_size, z_dim = z_hat.shape

        # Distribution for positives. batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlp = self.p_ZlX(x_pos)

        # shape: [2 * batch_size, 2 * batch_size * world_size]
        logits = self.compute_logits(p_Zlx, p_Zlp)

        # shape: [2 * batch_size]
        hat_H_mlz, logs = self.compute_loss(logits)

        # shape: [n_z, batch_size]
        hat_H_mlz = (hat_H_mlz[:batch_size] + hat_H_mlz[batch_size:]) / 2
        hat_H_mlz = einops.repeat(hat_H_mlz, "bz -> n_z bz", n_z=n_z)

        other = dict()

        return hat_H_mlz, logs, other

    def compute_logits(self, p_Zlx, p_Zlp):

        if isinstance(p_Zlx, DiagGaussian):
            assert not self.is_project

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

        elif isinstance(p_Zlx, Deterministic):

            # if you are using a deterministic representation then use standard InfoNCE
            # because if not KL is none differentiable
            # z,z_pos. shape: [batch_size,z_dim]
            z = p_Zlx.mean
            z_pos = p_Zlp.mean

            z = self.projector(z)
            z_pos = self.projector(z_pos)

            # shape: [2*batch_size,z_dim]
            zs = torch.cat([z, z_pos], dim=0)

            if self.is_cosine:
                # Note: simclr head projector already normalizes, but can issue normalize twice
                zs = F.normalize(zs, dim=1, p=2)

            # shape: [2*batch_size, 2*batch_size]
            logits = zs @ zs.T

            # NB: in the case of CLIP this is actually slightly different because we also compare
            # the logits images with images and text with text. We do so to double number of negatives.

        else:
            raise ValueError(f"Unkown type(p_Zlx)={type(p_Zlx)}.")

        # collect all negatives on different devices.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            list_logits = gather_from_gpus(logits)
            curr_gpu = torch.distributed.get_rank()
            curr_logits = list_logits[curr_gpu]
            other_logits = torch.cat(
                list_logits[:curr_gpu] + list_logits[curr_gpu + 1 :], dim=-1
            )

            # shape: [2*batch_size, 2*batch_size * world_size]
            logits = torch.cat([curr_logits, other_logits], dim=1)

        return logits

    def compute_loss(self, logits):
        n_classes = logits.size(1)
        new_batch_size = logits.size(0)
        batch_size = new_batch_size // 2
        device = logits.device

        if not self.is_invariant:
            # use NCE which is not invariant to transformations =>
            # the positive example is the example  itself
            pos_idx = torch.arange(new_batch_size, device=device)

        elif self.is_batch_neg:
            # whether the rest of the batch should also be viewed as negative examples
            # this doubles the number of negatives and should be used for image SSL

            # select all but current example.
            mask = ~torch.eye(new_batch_size, device=device).bool()
            n_to_add = n_classes - new_batch_size
            ones = torch.ones(new_batch_size, n_to_add, device=device).bool()
            # shape: [2*batch_size, 2*batch_size * world_size]
            mask = torch.cat([mask, ones], dim=1)
            n_classes -= 1  # remove the current example due to masking
            logits = logits[mask].view(new_batch_size, n_classes)

            # infoNCE is essentially the same as a softmax (where wights are other z => try to predict
            # index of positive z). The label is the index of the positive z, which for each z is the
            # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
            # you masked select all but the current z. arange takes care of idx of z which increases
            arange = torch.arange(batch_size, device=device)
            pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)

        elif not self.is_batch_neg:
            #! does not currently work with multi GPU
            # TODO test if necessary for CLIP and if not remove
            # does not use the batch as negative examples. This is what CLIP does: images have
            # to discriminate which text is associated with it and vis versa, but do not have to
            # discriminate betweeen the images and the text !
            logits_img2text = logits[:batch_size, batch_size:]
            logits_text2img = logits[batch_size:, :batch_size]
            logits = torch.cat([logits_img2text, logits_text2img], dim=0)
            arange = torch.arange(batch_size, device=device)
            pos_idx = torch.cat([arange, arange], dim=0)
            n_classes = batch_size

        if self.effective_batch_size is not None:
            # want the reweighting so that as if the batchsize was entire dataset
            # so number of negatives would be 2*detaset - 1 (one being beign current)
            effective_n_classes = 2 * self.effective_batch_size - 1

            # you want to multiply \sum exp(negative) in the denominator by to_mult
            # so that they have an effective weight of `effective_n_classes - 1`
            # -1 comes from the fact that only the negatives
            to_mult = (effective_n_classes - 1) / (n_classes - 1)

            # equivalent: add log(to_mult) to every negative logits
            # equivalent: add - log(to_mult) to positive logits
            to_add = -math.log(to_mult)
            to_add = to_add * torch.ones_like(logits[:, 0:1])  # correct shape
            logits.scatter_add_(1, pos_idx.unsqueeze(1), to_add)
            # when testing use `logits.gather(1, pos_idx.unsqueeze(1))` to see pos
        else:
            effective_n_classes = n_classes

        if self.is_train_temperature:
            temperature = 1 / torch.clamp(
                self.logit_scale.exp(), max=1 / self.temperature
            )
        else:
            temperature = self.temperature

        logits = logits / temperature

        # I[Z,f(M(X))] = E[ log \frac{(N-1) exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log softmax(z^Tz_p) ]
        # = log(N-1) - E[ crossentropy(z^Tz, p) ]
        # = \hat{H}[f(M(X))] - \hat{H}[f(M(X))|Z]
        hat_H_m = math.log(effective_n_classes)
        hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="none")

        logs = dict(
            I_q_zm=(hat_H_m - hat_H_mlz.mean()) / math.log(BASE_LOG),
            hat_H_m=hat_H_m / math.log(BASE_LOG),
            n_negatives=n_classes,
        )

        return hat_H_mlz, logs


class LossyZDistortion(nn.Module):
    """Computes the distortion by simply trying to reconstruct the given Z => Lossy compression of the
    representation without looking at X. Uses Mikowsky distance between z and z_hat.

    Parameters
    ----------
    p_norm : float, optional
        Which Lp norm to use for computing the distance.
    """

    def __init__(self, p_norm=1):
        super().__init__()
        self.distance = nn.PairwiseDistance(p=p_norm)

    def forward(self, z_hat, _, p_Zlx):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[n_z, batch_size, z_dim]
            Reconstructed representations. WIll take

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoded distribution of Z. Will take the mean to get the target z.

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

        # shape=[n_z*batch_size, z_dim]
        flat_z_hat = einops.rearrange(z_hat, "n_z b d -> (n_z b) d")

        # shape=[n_z*batch_size, z_dim]
        z_mean = p_Zlx.base_dist.mean
        z_mean = einops.repeat(z_mean, "b d -> (n_z b) d", n_z=n_z)

        # shape=[n_z*batch_size]
        dist = self.distance(flat_z_hat, z_mean)

        # shape: [n_z, batch_size]
        dist = einops.rearrange(dist, "(n_z b)  -> n_z b", n_z=n_z)

        logs = dict()
        other = dict()

        return dist, logs, other
