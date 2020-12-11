import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import einops
from .helpers import BASE_LOG, undo_normalization, kl_divergence
from .architectures import get_Architecture
from .distributions import Deterministic, DiagGaussian

__all__ = ["get_distortion_estimator"]


def get_distortion_estimator(name, p_ZlX, **kwargs):
    if name == "direct":
        return DirectDistortion(**kwargs)

    elif name == "contrastive":
        return ContrastiveDistortion(p_ZlX=p_ZlX, **kwargs)

    else:
        raise ValueError(f"Unkown loss={name}.")


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
    """

    def __init__(
        self,
        z_dim,
        y_shape,
        arch=None,
        arch_kwargs=dict(complexity=2),
        dataset=None,
        is_classification=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.is_classification = is_classification
        self.is_img_out = (not isinstance(y_shape, int)) and (len(y_shape) == 3)

        if arch is None:
            arch = "cnn" if self.is_img_out else "mlp"
        Decoder = get_Architecture(arch, **arch_kwargs)
        self.q_YlZ = Decoder(z_dim, y_shape)

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
        if not self.is_img_out:
            if self.is_classification:
                neg_log_q_ylz = F.cross_entropy(Y_hat, aux_target, reduction="none")
            else:
                neg_log_q_ylz = F.mse_loss(Y_hat, aux_target, reduction="none")
        else:
            if aux_target.shape[-3] == 1:
                # black white image => uses categorical distribution, with logits for stability
                neg_log_q_ylz = F.binary_cross_entropy_with_logits(
                    Y_hat, aux_target, reduction="none"
                )
            elif aux_target.shape[-3] == 3:
                # this is just to ensure that images are in [0,1] and compared in unormalized space
                # black white doesn't need because uses logits
                Y_hat, aux_target = undo_normalization(Y_hat, aux_target, self.dataset)
                # color image => uses gaussian distribution
                neg_log_q_ylz = F.mse_loss(Y_hat, aux_target, reduction="none")
            else:
                raise ValueError(
                    f"shape={aux_target.shape} does not seem to come from an image"
                )

        # -log p(y|z). shape: [n_z_samples, batch_size]
        #! mathematically should take a sum (log prod proba -> sum log proba), but usually people take mean
        neg_log_q_ylz = einops.reduce(
            neg_log_q_ylz, "(z b) ... -> z b", reduction="sum", z=n_z
        )

        logs = dict(H_q_YlZ=neg_log_q_ylz.mean() / math.log(BASE_LOG))

        other = dict()
        if self.is_img_out:
            # for image plotting
            other["rec_img"] = Y_hat[0].detach().cpu()
            other["real_img"] = aux_target[0].detach().cpu()

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

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in 
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self, p_ZlX, temperature=0.1, is_symmetric=True, is_cosine=True, weight=1
    ):
        super().__init__()
        self.p_ZlX = p_ZlX
        self.temperature = temperature
        self.is_symmetric = is_symmetric
        self.is_cosine = is_cosine
        self.weight = weight

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
        N = 2 * batch_size
        device = z_hat.device

        # Distribution for positives. batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlp = self.p_ZlX(x_pos)

        # shape: [2*batch_size,2*batch_size]
        logits = self.compute_logits(p_Zlx, p_Zlp)

        mask = ~torch.eye(N, device=device).bool()
        logits /= self.temperature

        # select all but current. shape: [2*batch_size,2*batch_size-1]
        logits = logits[mask].view(N, N - 1)

        # infoNCE is essentially the same as a softmax (where wights are other z => try to predict
        # index of positive z). The label is the index of the positive z, which for each z is the
        # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
        # you masked select all but the current z. arange takes care of idx of z which increases
        arange = torch.arange(batch_size, device=device)
        pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)

        if self.weight != 1:
            # you want to multiply \sum e(negative) in the denomiator by to_mult
            to_mult = (N - 1 - 1 / self.weight) / (N - 2)
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
        hat_H_m = math.log(self.weight * (N - 1))
        hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="mean")

        logs = dict(I_zm=(hat_H_m - hat_H_mlz) / math.log(BASE_LOG))
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
