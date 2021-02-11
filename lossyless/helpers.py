import contextlib
import itertools
import operator
import random
import sys
import time
from collections import OrderedDict
from functools import reduce
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

import einops
import torch
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from torch import nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torchvision import transforms as transform_lib

BASE_LOG = 2


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(
                module, module
            )
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(
                module, to_use, module
            )
            raise ImportError(error)


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start


def rename_keys_(dictionary, renamer):
    """Rename the keys in a dictionary using the renamer."""
    for old, new in renamer.items():
        dictionary[new] = dictionary.pop(old)


def dict_mean(dicts):
    """Average a list of dictionary."""
    means = {}
    for key in dicts[0].keys():
        means[key] = sum(d[key] for d in dicts) / len(dicts)
    return means


def orderedset(l):
    """Return a list of unique elements."""
    # could use list(dict.fromkeys(l)) in python 3.6+
    return [k for k, v in OrderedDict.fromkeys(l).items()]


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


MEANS = dict(
    imagenet=[0.485, 0.456, 0.406],
    cifar10=[0.4914009, 0.48215896, 0.4465308],
    galaxy64=[0.03341029, 0.04443058, 0.05051352],
    galaxy128=[0.03294565, 0.04387402, 0.04995899],
)
STDS = dict(
    imagenet=[0.229, 0.224, 0.225],
    cifar10=[0.24703279, 0.24348423, 0.26158753],
    galaxy64=[0.06985303, 0.07943781, 0.09557958],
    galaxy128=[0.07004886, 0.07964786, 0.09574898],
)


class Normalizer:
    def __init__(self, dataset):
        super().__init__()
        try:
            self.normalizer = transform_lib.Normalize(
                mean=MEANS[dataset], std=STDS[dataset]
            )
        except:
            self.normalizer = None

    def __call__(self, x):
        if x.size(-3) != 3 and self.normalizer is None:
            # if not colored and wasn't in dict
            return x

        return self.normalizer(x)


class UnNormalizer:
    def __init__(self, dataset):
        super().__init__()
        try:
            mean, std = MEANS[dataset], STDS[dataset]
            self.unnormalizer = transform_lib.Normalize(
                [-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            )
        except:
            self.unnormalizer = None

    def __call__(self, x):
        if x.size(-3) != 3 and self.unnormalizer is None:
            # if not colored and wasn't in dict
            return x

        return self.unnormalizer(x)


def get_normalization(Dataset):
    """Return corrrect normalization given dataset class."""
    if "cifar10" in Dataset.__name__.lower():
        return Normalizer("cifar10")
    elif "galaxy" in Dataset.__name__.lower():
        return Normalizer("galaxy64")
        # todo: different means for different resolution
        # return Normalizer("galaxy129")
    else:
        raise ValueError(f"Uknown mean and std for {Dataset}.")


def undo_normalization(Y_hat, targets, dataset):
    """Undo transformation of predicted and target images given dataset name.
    Used to ensure nice that can be used for plotting generated images."""

    # images are in [0,1] due to `ToTensor` so can use sigmoid to ensure output is also in [0,1]
    Y_hat = torch.sigmoid(Y_hat)

    unnormalizer = UnNormalizer(dataset)

    return unnormalizer(Y_hat), unnormalizer(targets)


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


def setup_grid(range_lim=4, n_pts=1000, device=torch.device("cpu")):
    """
    Return a tensor `xy` of 2 dimensional points (x,y) that span an entire grid [-range_lim,range_lim]
    with `n_pts` discretizations.
    """
    x = torch.linspace(-range_lim, range_lim, n_pts, device=device)
    y = torch.linspace(-range_lim, range_lim, n_pts, device=device)
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack((xx, yy), dim=-1)
    return xy.transpose(0, 1)  # indexing="xy"


def plot_density(p, n_pts=1000, range_lim=0.7, figsize=(7, 7), title=None, ax=None):
    """Plot the density of a distribution `p`."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    xy = setup_grid(range_lim=range_lim, n_pts=n_pts)

    ij = xy.transpose(0, 1)  # put as indexing="ij" more natural for indexing
    left, right, down, up = ij[0, 0, 0], ij[-1, 0, 0], ij[0, 0, 1], ij[0, -1, 1]
    data_p = torch.exp(p.log_prob(xy)).cpu().data

    vmax = data_p.max()
    ax.imshow(
        data_p,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=vmax,
        extent=(left, right, down, up),
        origin="lower",
    )

    ax.axis("image")
    ax.grid(False)
    ax.set_xlim(left, right)
    ax.set_ylim(down, up)
    ax.set_xlabel("Source dim. 1")
    ax.set_ylabel("Source dim. 2")

    if title is not None:
        ax.set_title(title)


def mse_or_crossentropy_loss(Y_hat, y, is_classification, is_sum_over_tasks=False):
    """Compute the cross entropy for multilabel clf tasks or MSE for regression"""

    if is_classification:
        loss = F.cross_entropy(Y_hat, y.long(), reduction="none")
    else:
        loss = F.mse_loss(Y_hat, y, reduction="none")

    if not is_sum_over_tasks:
        n_tasks = prod(Y_hat[0, 0, ...].shape)
        loss = loss / n_tasks  # takes an average over tasks

    batch_size = loss.size(0)
    loss = loss.view(batch_size, -1).sum(keepdim=True, dim=-1)

    return loss


def get_lr_scheduler(optimizer, mode, epochs=None, decay_factor=None, **kwargs):
    """Return the correct learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to wrap.

    mode : {None, "expdecay"}U{any torch lr_scheduler}
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
    if mode is None:
        return None
    elif mode == "expdecay":
        gamma = (1 / decay_factor) ** (1 / epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        Scheduler = getattr(torch.optim.lr_scheduler, mode)
        return Scheduler(optimizer, **kwargs)


def get_optimizer(parameters, mode, is_lars=False, **kwargs):
    """Return an inistantiated optimizer.

    Parameters
    ----------
    optimizer : {"gdn"}U{any torch.optim optimizer}
        Optimizer to use.mode

    is_lars : bool, optional
        Whether to use a LARS optimizer which can improve when using large batch sizes.

    kwargs : 
        Additional arguments to the optimzier.
    """
    Optimizer = getattr(torch.optim, mode)
    optimizer = Optimizer(parameters, **kwargs)
    if is_lars:
        optimizer = LARSWrapper(optimizer)
    return optimizer


def append_optimizer_scheduler_(
    hparams_opt, hparams_sch, parameters, optimizers, schedulers
):
    """Return the correct optimzier and scheduler."""
    optimizer = get_optimizer(parameters, hparams_opt.mode, **hparams_opt.kwargs)
    optimizers += [optimizer]

    for mode in hparams_sch.modes:
        sch_kwargs = hparams_sch.kwargs.get(mode, {})
        scheduler = get_lr_scheduler(optimizer, mode, **sch_kwargs)
        schedulers += [scheduler]

    return optimizers, schedulers
