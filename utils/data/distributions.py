from functools import partial
import math

import torch
import torch.distributions as dist
from torch.utils.data import Dataset
import numpy as np

from lossyless.helpers import tmp_seed, BASE_LOG
from .helpers import differential_entropy, discrete_entropy, rotate, int_or_ratio
from .base import LossylessDataset, LossylessDataModule

__all__ = ["BananaDataModule"]

### Base Classes ###
class LossylessDistributionDataset(LossylessDataset, Dataset):
    """Base class for 2D distribution datasets used for lossy compression but lossless predicitons.

    Note
    ----
    - target is max invariant.

    Parameters
    -----------
    distribution : torch.Distribution
        Main distribution to sample from.

    length : int, optional
        Size of the dataset.

    equivalence : {"rotation","y_translation","x_translation",None}, optional
        Equivalence relationship with respect to which to be invariant.

    decimals : float or None, optional
        Number of decimals to keep. If `None` keeps as much as python can. This is useful
        to have better estimates of the entropies.

    seed : int or None, optional
        Seed to force deterministic dataset (if int). This is especially useful when using
        `reload_dataloaders_every_epoch` but you only want to reload only the training set.

    kwargs:
        Additional arguments to `LossylessDataset`.
    """

    def __init__(
        self,
        distribution,
        length=1024000,
        equivalence="rotation",
        decimals=5,
        seed=None,
        **kwargs,
    ):
        super().__init__(equivalence=equivalence, seed=seed, **kwargs)

        self.length = length
        self.distribution = distribution
        self.equivalence = equivalence
        self.decimals = decimals
        self.data, self.targets = self.get_n_data_Mxs(length)

        # precompute quantiles which are for sampling quivalence action
        self.min_x, self.min_y = self.data.quantile(0.1, dim=0)
        self.max_x, self.max_y = self.data.quantile(0.9, dim=0)

    def get_x_target_Mx(self, index):
        Mx = self.targets[index]
        return self.data[index], Mx, Mx

    def get_representative(self, Mx):
        if self.equivalence == "y_translation":
            return torch.cat([Mx, torch.zeros_like(Mx)], dim=-1)

        if self.equivalence == "rotation":
            # use the 7.5 o'clock  representative to make it more clear in the banana distribution
            left_rep = torch.cat([-Mx, torch.zeros_like(Mx)], dim=-1)
            return rotate(left_rep, 45)

        elif self.equivalence == "x_translation":
            return torch.cat([torch.zeros_like(Mx), Mx], dim=-1)

        elif self.equivalence is None:
            return Mx

        else:
            raise ValueError(f"Unkown equivalence={self.equivalence}.")

    def get_max_var(self, x, Mx):
        # Mx is a regression task so one a possible max variant is simply predicting x
        return x

    def sample_equivalence_action(self):
        def action_translation(rep, min_ax, max_ax, axis):
            # random translation on * scale
            delta = max_ax - min_ax
            jitter = torch.rand_like(rep) * delta + min_ax
            x, y = jitter.chunk(2, dim=-1)
            xy = (x, y * 0) if axis == 0 else (x * 0, y)  # only jitter correct axis
            jitter_ax = torch.cat(xy, dim=-1)
            return rep + jitter_ax

        def action_rotation(rep):
            # random rotation
            angle = torch.rand(1) * 360
            return rotate(rep, angle)

        if self.equivalence == "rotation":
            return action_rotation

        elif self.equivalence == "y_translation":
            return partial(
                action_translation, min_ax=self.min_y, max_ax=self.max_y, axis=1
            )

        elif self.equivalence == "x_translation":
            return partial(
                action_translation, min_ax=self.min_x, max_ax=self.max_x, axis=0
            )

        elif self.equivalence is None:
            return lambda x: x

        else:
            raise ValueError(f"Unkown equivalence={self.equivalence}.")

    @property
    def entropies(self):
        if hasattr(self, "_entropies"):
            return self._entropies  # if precomputed

        entropies = {}
        n_samples = int(1e6)
        x_samples, Mx_samples = self.get_n_data_Mxs(n_samples)
        x_samples, Mx_samples = x_samples.numpy(), Mx_samples.numpy()

        # analytic computations
        entropies["true h[X]"] = self.distribution.entropy().item() / math.log(BASE_LOG)

        # sample estimates
        # continuous (i.e. estimate as continuous even though we only have discrete due to float-point)
        entropies["h[X]"] = differential_entropy(x_samples, base=BASE_LOG)
        entropies["h[M(X)]"] = differential_entropy(Mx_samples, base=BASE_LOG)
        # discrete (i.e. treat it as discrete as we only have discrete values)
        entropies["H[X]"] = discrete_entropy(x_samples, base=BASE_LOG)
        entropies["H[M(X)]"] = discrete_entropy(Mx_samples, base=BASE_LOG)

        entropies["h[Y]"] = entropies["h[M(X)]"]
        entropies["H[Y]"] = entropies["H[M(X)]"]

        self._entropies = entropies
        return entropies

    @property
    def is_clf_x_t_Mx(self):
        return dict(input=False, target=False, max_inv=False)

    @property
    def shapes_x_t_Mx(self):
        return dict(input=(2,), target=(1,), max_inv=(1,), max_var=(2,))

    def get_n_data_Mxs(self, length):
        """Return an array for the examples and correcponding max inv ofa given length."""
        with tmp_seed(self.seed):
            data = self.distribution.sample([length])
            # the targets will be derivative from max invariant => the important is Mx
            Mxs = self.max_invariant(data)

        if self.decimals is not None:
            data = np.around(data, decimals=self.decimals)
            Mxs = np.around(Mxs, decimals=self.decimals)

        return data, Mxs

    def max_invariant(self, samples):
        """Apply the maximal invariant M(x) to the last dim."""
        if self.equivalence == "rotation":
            return samples.norm(2, dim=-1, keepdim=True)  # L2 norm
        elif self.equivalence == "y_translation":
            return samples.chunk(2, dim=-1)[0]  # max inv is x coord
        elif self.equivalence == "x_translation":
            return samples.chunk(2, dim=-1)[1]  # max inv is y coord
        elif self.equivalence is None:
            return samples  # max inv is x itself
        else:
            raise ValueError(f"Unkown equivalence={self.equivalence}.")

    def __len__(self):
        return self.length


class DistributionDataModule(LossylessDataModule):
    def get_train_dataset(self, **dataset_kwargs):
        # will resample at every training epoch so you want a changing seed
        dataset_kwargs["seed"] = None
        dataset = self.Dataset(**dataset_kwargs)
        return dataset

    def get_val_dataset(self, **dataset_kwargs):
        dataset_kwargs["seed"] = None
        dataset_kwargs["length"] = int_or_ratio(self.val_size, len(self.train_dataset))

        dataset = self.Dataset(**dataset_kwargs)
        return dataset

    def get_test_dataset(self, **dataset_kwargs):
        # fix seed for reproducability
        dataset_kwargs["seed"] = self.seed

        test_size = self.val_size if self.test_size is None else self.test_size
        dataset_kwargs["length"] = int_or_ratio(test_size, len(self.train_dataset))

        dataset = self.Dataset(**dataset_kwargs)
        return dataset

    def prepare_data(self):
        pass  # no download

    @property
    def distribution(self):
        return self.train_dataset.distribution


### Banana Distribution ###
class BananaTransform(dist.Transform):
    """Transform from gaussian to banana."""

    def __init__(self, curvature, factor=10):
        super().__init__()
        self.bijective = True
        self.curvature = curvature
        self.factor = factor

    def _call(self, x):
        shift = torch.zeros_like(x)
        shift[..., 1] = self.curvature * (torch.pow(x[..., 0], 2) - self.factor ** 2)
        return x + shift

    def _inverse(self, y):
        shift = torch.zeros_like(y)
        shift[..., 1] = self.curvature * (torch.pow(y[..., 0], 2) - self.factor ** 2)
        return y - shift

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class RotateTransform(dist.Transform):
    """Rotate a distribution from `angle` degrees."""

    def __init__(self, angle):
        super().__init__()
        self.bijective = True
        self.angle = angle

    def _call(self, x):
        return rotate(x, self.angle)

    def _inverse(self, y):
        return rotate(y, -self.angle)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class BananaDistribution(dist.TransformedDistribution):
    """2D banana distribution.

    Parameters
    ----------
    curvature : float, optional
        Controls the strength of the curvature of the banana-shape.

    factor : float, optional
        Controls the elongation of the banana-shape.

    location : torch.Tensor, optional
        Controls the location of the banana-shape.

    angles : float, optional
        Controls the angle rotation of the banana-shape.

    scale : float, optional
        Rescales the entire distribution (while keeping entropy of underlying distribution correct)
        This is useful to make sure that the inputs during training are not too large / small.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        curvature=0.05,
        factor=6,
        location=torch.as_tensor([-1.5, -2.0]),
        angle=-40,
        scale=1 / 2,
    ):
        std = torch.as_tensor([factor * scale, scale])
        base_dist = dist.Independent(dist.Normal(loc=torch.zeros(2), scale=std), 1)

        transforms = dist.ComposeTransform(
            [
                BananaTransform(curvature / scale, factor=factor * scale),
                RotateTransform(angle),
                dist.AffineTransform(location * scale, 1),
            ]
        )
        super().__init__(base_dist, transforms)

        self.curvature = curvature
        self.factor = factor
        self.rotate = rotate

    def entropy(self):
        return self.base_dist.entropy()  # log det is zero => same entropy


class BananaDataset(LossylessDistributionDataset):
    def __init__(self, **kwargs):
        super().__init__(distribution=BananaDistribution(), **kwargs)


class BananaDataModule(DistributionDataModule):
    @property
    def Dataset(self):
        return BananaDataset
