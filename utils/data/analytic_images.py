import logging
import math
import os

import scipy

import torch
from lossyless.helpers import BASE_LOG
from torchvision.datasets import MNIST
from utils.estimators import discrete_entropy

from .helpers import RotationAction, ScalingAction, TranslationAction
from .images import LossylessImgDataset, TorchvisionDataModule

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyticMnistDataModule",
]


class LossylessImgAnalyticDataset(LossylessImgDataset):
    """Base class for image datasets with action entropies that can be computed analytically.

    Note
    ----
    - THe target is not the actual classificatoin but the worst case task, i.e., the maximal invariant.
    These datasets should thus only be used to compute rate distortion curves
    in terms of the invariant distortion H[M(X)|Z], i.e. worst case tasks. If you want rate-task 
    prediction curves you should really be using `LossylessImgDataset`.

    Parameters
    -----------
    equivalence : set of {"rotation","y_translation","x_translation","scale"}, optional
        List of equivalence relationship with respect to which to be invariant.

    n_action_per_equiv : int, optional
        Number of actions for each equivalence. Typically 4 or 8.

    dist_actions : {"uniform","bell"}, optional
        Distribution to use for all actions. "bell" is a symmetric beta binomial distribution.

    kwargs:
        Additional arguments to `LossylessCLFDataset` and `LossylessDataset` and `LossylessImgDataset`.
    """

    def __init__(
        self, *args, n_action_per_equiv=8, dist_actions="uniform", **kwargs,
    ):
        self.n_action_per_equiv = n_action_per_equiv
        self.dist_actions = dist_actions

        if self.dist_actions == "uniform":
            self.rv_A = scipy.stats.randint(0, self.n_action_per_equiv)
        elif self.dist_actions == "bell":
            b = 100  # larger means more peaky (less variance)
            self.rv_A = scipy.stats.betabinom(self.n_action_per_equiv - 1, b, b)
        else:
            raise ValueError(
                f"Unkown `dist_actions={dist_actions}` should be `uniform` or `bell`."
            )

        super().__init__(*args, **kwargs)

    def get_x_target_Mx(self, index):
        # don't return the target only the max_inv
        img, _ = self.get_img_target(index)
        img = self.base_tranform(img)
        img = self.aug_transform(img)
        max_inv = index
        return img, max_inv, max_inv

    @property
    def augmentations(self):
        shape = self.shapes_x_t_Mx["input"]
        return {
            "rotation": RotationAction(self.rv_A, max_angle=60),
            "y_translation": TranslationAction(
                self.rv_A, dim=1, max_trnslt=shape[1] // 8
            ),
            "x_translation": TranslationAction(
                self.rv_A, dim=0, max_trnslt=shape[2] // 8
            ),
            "scale": ScalingAction(self.rv_A, max_scale=1.2),
        }

    @property
    def entropies(self):
        if hasattr(self, "_entropies"):
            return self._entropies  # if precomputed

        entropies = {}

        entropies["H[Y]"] = discrete_entropy(self.targets, base=BASE_LOG)
        # Marginal entropy can only be computed on training set by treating dataset as the real uniform rv
        entropies["train H[M(X)]"] = math.log(len(self), BASE_LOG)

        # Data augmentation are applied independently so H[X|M(X)]=\sum_i H[A_i] = k * H[A]
        # where A_i is the r.v. for sampling the ith action, which is the same for each action A_i = A_j
        entropies["H[X|M(X)]"] = len(self.equivalence) * self.rv_A.entropy()
        entropies["H[X|M(X)]"] /= math.log(BASE_LOG)

        entropies["train H[M(X)]"] = math.log(len(self), BASE_LOG)
        entropies["train H[X]"] = entropies["train H[M(X)]"] + entropies["H[X|M(X)]"]

        self._entropies = entropies
        return entropies

    # TODO clean max_var for multi label multi clf
    def get_max_var(self, x, Mx):
        # the max_var for analytic transforms Mx and the indices of the transform you sampled
        # e.g. if you rotate and translate it will be multilabel prediction of the rotation index
        # and translation index IN ADDITION to the Mx => ensure that it is classification
        # like for the maximal invariant but not invariant anymore => can use VIB loss
        max_var = [Mx]
        for trnsf in self.aug_transform.transforms:
            # all analytic transforms store the last index they sampled
            max_var.append(trnsf.i)

        # we would want a list of targets, where the first element is max_inv, then other elements
        # are the indices of the transformation that you sampled. But that cannot be put in a batch
        # as Mx does not usually have same size as `self.n_action_per_equiv`. So we flatten everything
        # and will be unflattened when computing the loss
        return torch.as_tensor(max_var)

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx

        # the target will be the max_inv
        shapes["target"] = shapes["max_inv"]

        if self.additional_target == "max_var":
            # flattened max var (see `get_max_var`)
            shapes["max_var"] = (sum(self.shape_max_var),)

        return shapes

    @property
    def shape_max_var(self):
        """Actual number of elements for each prediction task when `additional_target=max_var`."""
        shapes = super().shapes_x_t_Mx
        mx_shape = shapes["max_inv"]

        if len(mx_shape) > 1:
            raise NotImplementedError(
                f"Can only work with vector max_inv when using max_var, but shape={mx_shape}."
            )

        mv_shape = list(mx_shape) + [self.n_action_per_equiv] * len(self.equivalence)
        return tuple(mv_shape)


# to make an analytic dataset just rewrite the same as an image dataset without redefining
# the shape of the target
class AnalyticMnistDataset(LossylessImgAnalyticDataset, MNIST):
    FOLDER = "MNIST"

    # avoid duplicates by saving once at "MNIST" rather than at multiple  __class__.__name__
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.FOLDER, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.FOLDER, "processed")

    @property
    def shapes_x_t_Mx(self):
        shapes = super(AnalyticMnistDataset, self).shapes_x_t_Mx
        shapes["input"] = (1, 32, 32)
        #! ONLY DIFFERENCE IS THAT WE COMMENT / RM THAT LINE
        # shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = MNIST.__getitem__(self, index)
        return img, target


class AnalyticMnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return AnalyticMnistDataset
