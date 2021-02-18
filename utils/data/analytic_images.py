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
        img, _, max_inv = super().get_x_target_Mx(index)
        target = max_inv
        return img, target, max_inv

    @property
    def augmentations(self):
        shape = self.shapes_x_t_Mx["input"]
        return dict(
            PIL={
                "rotation": RotationAction(self.rv_A, max_angle=60),
                "y_translation": TranslationAction(
                    self.rv_A, dim=1, max_trnslt=shape[1] // 8
                ),
                "x_translation": TranslationAction(
                    self.rv_A, dim=0, max_trnslt=shape[2] // 8
                ),
                "scale": ScalingAction(self.rv_A, max_scale=1.2),
            },
            tensor={},
        )

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

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx

        # the target will be the max_inv
        shapes["target"] = shapes["max_inv"]

        return shapes


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

    @property
    def is_train(self):
        return self.train

    @property
    def dataset_name(self):
        return "MNIST"


class AnalyticMnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return AnalyticMnistDataset
