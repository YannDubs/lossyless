from functools import partial
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torchvision
import math

import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal, MixtureSameFamily
import einops

from .helpers import weights_init, batch_flatten, batch_unflatten, prod, Delta

__all__ = ["CondDist", "get_marginalDist"]

### CONDITIONAL DISTRIBUTIONS ###
class CondDist(nn.Module):
    """Return the (uninstantiated) correct CondDist.

    Parameters
    ----------
    in_shape : tuple of int

    out_dim : int

    Architecture : nn.Module
        Module to be instantiated by `Architecture(in_shape, out_dim)`.

    family : {"gaussian","uniform"}
        Family of the distribution (after conditioning), this can be easily extandable to any
        distribution in `torch.distribution`.

    kwargs :
        Additional arguments to the `Family`.
    """

    def __init__(self, in_shape, out_dim, Architecture, family, **kwargs):
        super().__init__()

        if family == "diaggaussian":
            self.Family = DiagGaussian
        elif family == "deterministic":
            self.Family = Deterministic
        else:
            raise ValueError(f"Unkown family={family}.")

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.kwargs = kwargs

        self.mapper = Architecture(in_shape, out_dim * self.Family.n_param)

        self.reset_parameters()

    def forward(self, x):
        """Compute the distribution conditioned on `X`.

        Parameters
        ----------
        Xx: torch.Tensor, shape: [batch_size, *in_shape]
            Input on which to condition the output distribution.

        Return
        ------
        p(.|x) : torch.Distribution, batch shape: [batch_size] event shape: [out_dim]
        """

        # shape: [batch_size, out_dim * n_param]
        suff_param = self.mapper(x)

        # batch shape: [batch_size] ; event shape: [out_dim]
        p__lx = self.Family.from_suff_param(suff_param, **self.kwargs)

        return p__lx

    def reset_parameters(self):
        weights_init(self)


class Distributions:
    """Base class for distributions that can be instantiated with joint suff stat."""

    n_param = None  # needs to be defined in each class

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_suff_param(cls, concat_suff_params, **kwargs):
        """Initialize the distribution using the concatenation of sufficient parameters (output of NN)."""
        # shape: [batch_size, -1] * n_param
        suff_params = einops.rearrange(
            concat_suff_params, "b (z p) -> b z p", p=cls.n_param
        ).unbind(-1)
        suff_params = cls.preprocess_suff_params(*suff_params)
        return cls(*suff_params, **kwargs)

    @classmethod
    def preprocess_suff_params(cls, *suff_params):
        """Preprocesses parameters outputed from network (usually to satisfy some constraints)."""
        return suff_params

    def detach(self):
        """Detaches all the parameters."""
        raise NotImplementedError()


class DiagGaussian(Distributions, Independent):
    """Gaussian with diagonal covariance."""

    n_param = 2
    min_std = 1e-5

    def __init__(self, diag_loc, diag_scale):
        super().__init__(Normal(diag_loc, diag_scale), 1)
        self.min_std

    @classmethod
    def preprocess_suff_params(cls, diag_loc, diag_log_var):
        # usually exp()**0.5, but you don't want to explose
        diag_scale = F.softplus(diag_log_var) + cls.min_std
        return diag_loc, diag_scale

    def detach(self):
        loc = self.base_dist.loc.detach()
        scale = self.base_dist.scale.detach()
        return DiagGaussian(loc, scale)


class Deterministic(Distributions, Independent):
    """Delta function distribution (i.e. no stochasticity)."""

    n_param = 1

    def __init__(self, param):
        super().__init__(Delta(param), 1)

    def detach(self):
        loc = self.base_dist.loc.detach()
        return Deterministic(loc)


### MARGINAL DISTRIBUTIONS ###


def get_marginalDist(family, cond_dist, **kwargs):
    """Return an approximate marginal distribution.

    Notes
    -----
    - Marginal ditsributions are Modules that TAKE NO ARGUMENTS and return the correct distribution
    as they are modules, they ensure that parameters are on the correct device.
    """
    if family == "unitgaussian":
        marginal = MarginalUnitGaussian(cond_dist.out_dim, **kwargs)
    elif family == "vamp":
        marginal = MarginalVamp(cond_dist.out_dim, cond_dist, **kwargs)
    else:
        raise ValueError(f"Unkown family={family}.")
    return marginal


class MarginalUnitGaussian(nn.Module):
    """Mean 0 covariance 1 Gaussian."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

        self.register_buffer("loc", torch.tensor([0.0] * self.out_dim))
        self.register_buffer("scale", torch.tensor([1.0] * self.out_dim))

    def forward(self):
        return Independent(Normal(self.loc, self.scale), 1)


class MarginalDiagGaussian(nn.Module):
    """Trained Gaussian with diag covariance."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.loc = nn.Parameter(torch.tensor([0.0] * self.out_dim))
        self.scale = nn.Parameter(torch.tensor([0.0] * self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.loc, -0.05, 0.05)
        nn.init.uniform_(self.scale, -0.05, 0.05)

    def forward(self):
        return Independent(Normal(self.loc, self.scale), 1)


class MarginalVamp(nn.Module):
    """
    Trained Gaussian using VampPrior [1], i.e. approximates the REAL marginal using
    pseudoinputs p(z) ~= 1/k \sum p(z|u^k) where x^k are trained.

    Parameters
    ----------
    out_dim : int
        Size event.

    p_ZlX : CondDist
        Instantiated conditional distribution.

    is_train_mixture : bool, optional
        Whether to train the mixture model. This was not considered in [1],

    References
    ---------
    [1] Tomczak, Jakub, and Max Welling. "VAE with a VampPrior." International Conference on
    Artificial Intelligence and Statistics. PMLR, 2018.
    """

    def __init__(self, out_dim, p_ZlX, n_pseudo=64, is_train_mixture=True):
        super().__init__()
        self.out_dim = out_dim
        self.p_ZlX = p_ZlX
        self.n_pseudo = n_pseudo
        self.is_train_mixture = is_train_mixture

        self.pseudoinputs = None
        self.mixtures = torch.nn.Parameter(
            torch.ones(self.n_pseudo), requires_grad=self.is_train_mixture
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.mixtures, 1)
        self.pseudoinputs = None  # set_pseudoinput should always be called after reset

    def set_pseudoinput_(self, X):
        """Set the initial pseudo inputs to be like the data."""
        self.pseudoinputs = torch.nn.Parameter(X)

    def forward(self):

        # batch shape: [n_pseudo] ; event shape: [z_dim]
        p_Zlu = self.p_ZlX(self.pseudoinputs)

        # p_Z. batch shape: [] ; event shape: [z_dim]
        mixtures = Categorical(logits=self.mixtures)
        p_Z = MixtureSameFamily(mixtures, p_Zlu)

        return p_Z
