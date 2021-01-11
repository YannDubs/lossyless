from torch import nn

import numpy as np
from torch.nn.utils.rnn import PackedSequence
import torch
import itertools
from torchvision.datasets import CIFAR10
from torchvision import transforms as transform_lib
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from numbers import Number
import contextlib
import random
from functools import reduce
import operator

import einops


BASE_LOG = 2


# modified from https://github.com/skorch-dev/skorch/blob/92ae54b/skorch/utils.py#L106
def to_numpy(X):
    """Convert tensors,list,tuples,dataframes to numpy arrays."""
    if isinstance(X, np.ndarray):
        return X

    # the sklearn way of determining pandas dataframe
    if hasattr(X, "iloc"):
        return X.values

    if isinstance(X, (tuple, list)):
        return np.array(X)

    if not isinstance(X, (torch.Tensor, PackedSequence)):
        raise TypeError(f"Cannot convert {type(X)} to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def concatenate(arr):
    """Concatenate (axis=0) an iterable of data, assuming all example have same type"""
    if isinstance(arr[0], np.ndarray):
        return np.concatenate(arr)
    elif isinstance(arr[0], torch.Tensor):
        return torch.cat(arr)
    elif isinstance(arr[0], (list, tuple, set)):
        T = type(arr[0])
        return T(itertools.chain.from_iterable(arr))
    elif isinstance(arr[0], dict):
        return dict(itertools.chain.from_iterable(d.items() for d in arr))
    else:
        raise ValueError(f"Don't know how to concatenate data of type {type(arr[0])}.")


def set_seed(seed):
    """Set the random seed."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


@contextlib.contextmanager
def tmp_seed(seed):
    """Context manager to use a temporary random seed with `with` statement."""
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state()

    set_seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            # if seed is None do as if no tmp_seed
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(torch_cuda_state)


def weights_init(module, nonlinearity="relu"):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    # loop over direct children (not grand children)
    for m in module.children():

        # all standard layers
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass

        elif isinstance(m, nn.BatchNorm2d):
            try:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            except AttributeError:  # affine = False
                pass

        elif hasattr(m, "reset_parameters"):  # if has a specific reset
            m.reset_parameters()
            #! don't go in grand children because you might have specifc weights you don't want to reset

        else:
            weights_init(m, nonlinearity=nonlinearity)  # go to grand children


def batch_flatten(x):
    """Batch wise flattenting of an array."""
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def batch_unflatten(x, shape):
    """Revert `batch_flatten`."""
    return x.reshape(*shape[:-1], -1)


def prod(iterable):
    """Take product of iterable like."""
    return reduce(operator.mul, iterable, 1)


def mean(array):
    """Take mean of array like."""
    return sum(array) / len(array)


def is_pow2(n):
    """Check if a number is a power of 2."""
    return (n != 0) and (n & (n - 1) == 0)


def get_lr_scheduler(optimizer, name, epochs=None, decay_factor=None, **kwargs):
    """Return the correct learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to wrap.

    name : {None, "expdecay"}U{any torch lr_scheduler}
        Name of the optimizer to use. "expdecay" uses an exponential decay scheduler where the lr
        is decayed by `decay_factor` during training. Needs to be given `epochs`. If another `str`
        it must be a `torch.optim.lr_scheduler` in which case the arguments are given by `kwargs`.

    epochs : int, optional
        Number of epochs during training.

    decay_factor : int, optional
        By how much to reduce learning rate during training. Only if `name = "expdecay"`.

    kwargs :
        Additional arguments to any `torch.optim.lr_scheduler`.

    """
    if name is None:
        return None
    elif name == "expdecay":
        gamma = (1 / decay_factor) ** (1 / epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        return Scheduler(optimizer, **kwargs)


def kl_divergence(p, q, z_samples=None, is_lower_var=False, is_reduce=True):
    """Computes KL[p||q], analytically if possible but with MC."""
    try:
        kl_pq = torch.distributions.kl_divergence(p, q)

        if not is_reduce:
            kl_pq = einops.repeat(kl_pq, "... -> z ...", z=z_samples.size(0))

    except NotImplementedError:
        # removes the event shape
        log_q = q.log_prob(z_samples)
        log_p = p.log_prob(z_samples)
        if is_lower_var:
            # http://joschu.net/blog/kl-approx.html
            log_r = log_q - log_p
            # KL[p||q] = (r−1) - logr
            kl_pq = log_r.exp() - 1 - log_r
        else:
            # KL[p||q] = E_p[log p] - E_p[log q]
            kl_pq = log_p - log_q

        if is_reduce:
            kl_pq = kl_pq.mean(0)

    return kl_pq


# TODO add galaxy
MEANS = dict(imagenet=[0.485, 0.456, 0.406], cifar10=[0.4914009, 0.48215896, 0.4465308])
STDS = dict(
    imagenet=[0.229, 0.224, 0.225], cifar10=[0.24703279, 0.24348423, 0.26158753]
)


def get_normalization(Dataset):
    """Return corrrect normalization given dataset class."""
    if "cifar10" in Dataset.__name__.lower():
        return transform_lib.Normalize(mean=MEANS["cifar10"], std=STDS["cifar10"])
    # TODO add galaxy
    else:
        raise ValueError(f"Uknown mean and std for {Dataset}.")


def undo_normalization(Y_hat, targets, dataset):
    """Undo transformation of predicted and target images given dataset name.
    Used to ensure nice that can be used for plotting generated images."""

    # images are in [0,1] due to `ToTensor` so can use sigmoid to ensure output is also in [0,1]
    Y_hat = torch.sigmoid(Y_hat)

    if Y_hat.size(-3) == 3:
        # only normalized if color
        mean, std = MEANS[dataset], STDS[dataset]
        denormalize = transform_lib.Normalize(
            [-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
        )
        Y_hat, targets = denormalize(Y_hat), denormalize(targets)

    return Y_hat, targets


def atleast_ndim(x, ndim):
    """Reshapes a tensor so that it has at least n dimensions."""
    if x is None:
        return None
    return x.view(list(x.shape) + [1] * (ndim - x.ndim))


# modified from: http://docs.pyro.ai/en/stable/_modules/pyro/distributions/delta.html#Delta
class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Parameters
    ----------
    v: torch.Tensor
        The single support element.

    log_density: torch.Tensor, optional
        An optional density for this Delta. This is useful to keep the class of :class:`Delta`
        distributions closed under differentiable transformation.

    event_dim: int, optional
        Optional event dimension.
    """

    has_rsample = True
    arg_constraints = {"loc": constraints.real, "log_density": constraints.real}
    support = constraints.real

    def __init__(self, loc, log_density=0.0, validate_args=None):
        self.loc, self.log_density = broadcast_all(loc, log_density)

        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()

        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super().__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = list(sample_shape) + list(self.loc.shape)
        return self.loc.expand(shape)

    def log_prob(self, x):
        log_prob = (x == self.loc).type(x.dtype).log()
        return log_prob + self.log_density
