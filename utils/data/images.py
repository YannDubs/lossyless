import logging
from PIL import Image
import abc

import torch

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import os
import torch
import math
import scipy
from torch.utils.data import random_split
from torchvision import transforms as transform_lib
from torchvision.transforms import (
    ColorJitter,
    RandomErasing,
    RandomAffine,
    RandomRotation,
)

from lossyless.helpers import (
    get_normalization,
    BASE_LOG,
)

from .base import LossylessCLFDataset, LossylessDataModule
from .helpers import (
    discrete_entropy,
    RotationAction,
    TranslationAction,
    ScalingAction,
    int_or_ratio,
)


logger = logging.getLogger(__name__)

__all__ = [
    "Cifar10DataModule",
    "MnistDataModule",
    "FashionMnistDataModule",
    "GalaxyDataModule",
]


### HELPERS ###


### Base Classes ###
class LossylessImgDataset(LossylessCLFDataset):
    """Base class for image datasets used for lossy compression but lossless predicitons.

    Parameters
    -----------
    equivalence : set of str, optional
        List of equivalence relationship with respect to which to be invariant. 

    is_augment_val : bool, optional
        Whether to augment the validation + test set.

    is_normalize : bool, optional
        Whether to normalize the input images. Only for colored images. If True, you should ensure
        that `MEAN` and `STD` and `get_normalization` and `undo_normalization` in `lossyless.helpers` 
        can normalize your data.

    kwargs:
        Additional arguments to `LossylessCLFDataset` and `LossylessDataset`.
    """

    def __init__(
        self, *args, equivalence={}, is_augment_val=False, is_normalize=True, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.equivalence = equivalence
        self.is_augment_val = is_augment_val
        self.is_normalize = is_normalize

        self.base_tranform = self.get_base_transform()  # base transform
        self.aug_transform = self.get_aug_transform()  # real augmentation

    @abc.abstractmethod
    def get_img_target(self, index):
        """Return the unaugmented image (in PIL format) and target."""
        ...

    def get_x_target_Mx(self, index):
        """Return the correct example, target, and maximal invariant."""
        img, target = self.get_img_target(index)
        img = self.base_tranform(img)
        img = self.aug_transform(img)
        return img, target, index

    @property
    def augmentations(self):
        shape = self.shapes_x_t_Mx["input"]
        return {
            "rotation": RandomRotation(60),
            "y_translation": RandomAffine(0, translate=(0, shape[2])),
            "x_translation": RandomAffine(0, translate=(shape[1], 0)),
            "scale": RandomAffine(0, scale=(0.8, 1.2)),
            "color": ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            "erasing": RandomErasing(value=0.5),
        }

    def sample_equivalence_action(self):
        trnsfs = []

        for equiv in self.equivalence:
            if equiv in self.augmentations:
                trnsfs += [self.augmentations[equiv]]
            else:
                raise ValueError(f"Unkown `equivalence={equiv}`.")

        return transform_lib.Compose(trnsfs)

    def get_representative(self, index):
        notaug_img, _ = self.get_img_target(index)
        notaug_img = self.base_tranform(notaug_img)
        return notaug_img

    def get_max_var(self, x, Mx):
        raise NotImplementedError(
            "`max_var` can only be used imsges that are from `LossylessImgAnalyticDataset`,"
        )

    @property
    def entropies(self):
        if hasattr(self, "_entropies"):
            return self._entropies  # if precomputed

        entropies = {}

        entropies["H[Y]"] = discrete_entropy(self.targets, base=BASE_LOG)
        # Marginal entropy can only be computed on training set by treating dataset as the real uniform rv
        entropies["train H[M(X)]"] = math.log(len(self), BASE_LOG)

        self._entropies = entropies
        return entropies

    def get_base_transform(self):
        """Return the base transform, ie train or test."""
        shape = self.shapes_x_t_Mx["input"]
        trnsfs = [
            transform_lib.Resize((shape[1], shape[2])),
            transform_lib.ToTensor(),
        ]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            trnsfs += [get_normalization(type(self))]

        return transform_lib.Compose(trnsfs)

    def get_aug_transform(self):
        """Return the augmentations transorms."""

        if self.is_augment_val or self.train:
            return self.sample_equivalence_action()
        else:
            return transform_lib.Compose([])  # identity

    def __len__(self):
        return len(self.data)

    @property
    def is_color(self):
        shape = self.shapes_x_t_Mx["input"]
        return shape[0] == 3

    @property
    def is_clf_x_t_Mx(self):
        return dict(input=not self.is_color, target=True, max_inv=True)

    @property
    def shapes_x_t_Mx(self):
        #! In each child should assign "input" and "target"
        return dict(max_inv=(len(self),))


class LossylessImgAnalyticDataset(LossylessImgDataset):
    """Base class for image datasets with action entropies that can be computed analytically.

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

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        # first dim of max_var will be the max_inv, then other dims will correspond
        # to the index of the transformation that you sampled
        max_var = list(shapes["max_inv"])
        max_var += [self.n_action_per_equiv] * len(self.equivalence)
        shapes["max_var"] = tuple(max_var)
        return shapes

    def get_max_var(self, x, Mx):
        # the max_var for analytic transforms is the index of the transform you sampled
        # if you rotate and translate it will be multilabel prediction of the rotation index
        # and translation index IN ADDITION to the Mx => ensure that it is classification
        # like for the maximal invariant but not invariant anymore => can use VIB loss
        max_var = [Mx]
        for trnsf in self.aug_transform.transforms:
            # all analytic transforms store the last index they sampled
            max_var.append(trnsf.i)
        return max_var


### Torchvision Models ###

# Base class for data module for torchvision models.
class TorchvisionDataModule(LossylessDataModule):
    def get_train_valid_dataset(self, **dataset_kwargs):
        dataset = self.Dataset(
            self.data_dir, train=True, download=False, **self.dataset_kwargs,
        )

        n_val = int_or_ratio(self.val_size, len(dataset))
        train, valid = random_split(
            dataset,
            [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )

        return train, valid

    def get_train_dataset(self, **dataset_kwargs):
        train, _ = self.get_train_valid_dataset(**dataset_kwargs)
        return train

    def get_valid_dataset(self, **dataset_kwargs):
        _, valid = self.get_train_valid_dataset(**dataset_kwargs)
        return valid

    def get_test_dataset(self, **dataset_kwargs):
        test = self.Dataset(
            self.data_dir, train=False, download=False, **self.dataset_kwargs,
        )
        return test

    def prepare_data(self):
        self.Dataset(self.data_dir, train=True, download=True, **self.dataset_kwargs)
        self.Dataset(self.data_dir, train=False, download=True, **self.dataset_kwargs)


## Analytic datasets ##
# MNIST #
class MnistDataset(LossylessImgAnalyticDataset, MNIST):
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
        shapes = super(MnistDataset, self).shapes_x_t_Mx
        shapes["input"] = (1, 32, 32)
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = MNIST.__getitem__(self, index)
        return img, target


class MnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return MnistDataset


# Fasion MNIST #
class FashionMnistDataset(LossylessImgAnalyticDataset, FashionMNIST):
    FOLDER = "FashionMNIST"

    def get_img_target(self, index):
        img, target = FashionMNIST.__getitem__(self, index)
        return img, target

    @property
    def shapes_x_t_Mx(self):
        shapes = super(FashionMnistDataset, self).shapes_x_t_Mx
        shapes["input"] = (1, 32, 32)
        shapes["target"] = (10,)
        return shapes

    processed_folder = MnistDataset.processed_folder
    raw_folder = MnistDataset.raw_folder


class FashionMnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return FashionMnistDataset


# Cifar10 #
class Cifar10Dataset(LossylessImgAnalyticDataset, CIFAR10):
    @property
    def shapes_x_t_Mx(self):
        shapes = super(Cifar10Dataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 32, 32)
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = CIFAR10.__getitem__(self, index)
        return img, target


class Cifar10DataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return Cifar10Dataset


## Non Analytic datasets ##

# Imagenet #
# TODO

### Torchvision Models ###

# Galaxy Zoo #

# TODO @karen: modify as desired all those methods
# TODO we should also add the mean and std of "galaxy" in `lossyless.helpers` to normalize the data
class GalaxyDataset(LossylessImgAnalyticDataset):
    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # do as needed. For best compatibility with the framework
        # self.data should contain the downloaded data in tensor or numpy form
        # self.targets should contain the targets
        # self.train should say whether training

        # example:
        self.download()
        self.data, self.targets = torch.load("path")

    def download(self):
        # if there's link such that we can run something like
        #     subprocess.check_call(["curl", "-L", self.urls["train"], "--output", save_path])
        #     with zipfile.ZipFile(save_path) as zf: zf.extractall(self.dir)
        # would be nice. But people will have to download Imagenet in any case to replicate results
        # so we can also tell them to download that one.
        ...

    @property
    def augmentations(self):
        shape = self.shapes_x_t_Mx["input"]
        return {
            "rotation": RandomRotation(60),
            "y_translation": RandomAffine(0, translate=(0, shape[2])),
            "x_translation": RandomAffine(0, translate=(shape[1], 0)),
            "scale": RandomAffine(0, scale=(0.8, 1.2)),
            "color": ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            "erasing": RandomErasing(value=0.5),
        }

    @property
    def is_clf_x_t_Mx(self):
        is_clf = super(GalaxyDataset, self).is_clf_x_t_Mx
        # input should be true is using log loss for reconstruction (typically MNIST) and False if MSE (typically colored images)
        is_clf["input"] = True
        # target should be True if log loss (ie classification) and False if MSE (ie regression)
        is_clf["target"] = False
        return is_clf

    @property
    def shapes_x_t_Mx(self):
        shapes = super(GalaxyDataset, self).shapes_x_t_Mx
        # input is shape image
        shapes["input"] = (3, 64, 64)
        # target is shape of target. This will depend as to if we are using classfication or regression
        # in regression mode (as we said) `target=(3,)` means that there are 6 values to predict
        # (it's equivalent to `target=(1,1,1)` it depends how the targets are formatted)
        # for classification `target=(3,)` means 3-class classification and `target=(3,2)` means
        # multi label classification, one with 3-classes and one with 2-classes
        # (I still have to implement multilabel classification but will be soon)
        shapes["target"] = (6,)
        return shapes

    def get_img_target(self, index):
        # change as needed but something like that
        img = self.images[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # don't apply transformation yet, it's done for you

        return img, target


class GalaxyDataModule(LossylessDataModule):
    @property
    def Dataset(self):
        return GalaxyDataset

    # helper function for splitting train and valid
    def get_train_valid_dataset(self, **dataset_kwargs):
        dataset = self.Dataset(
            self.data_dir, train=True, download=False, **self.dataset_kwargs,
        )

        # use the following if there's no validation set predefined
        n_val = int_or_ratio(self.val_size, len(dataset))
        train, valid = random_split(
            dataset,
            [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )

        return train, valid

    def get_train_dataset(self, **dataset_kwargs):
        train, _ = self.get_train_valid_dataset(**dataset_kwargs)
        return train

    def get_valid_dataset(self, **dataset_kwargs):
        # if there's a validation set then do what you need here
        _, valid = self.get_train_valid_dataset(**dataset_kwargs)
        return valid

    def get_test_dataset(self, **dataset_kwargs):
        test = self.Dataset(
            self.data_dir, train=False, download=False, **self.dataset_kwargs,
        )
        return test

    def prepare_data(self):
        # this is where the downlading should happen if we can if not just put `pass`
        self.Dataset(self.data_dir, train=True, download=True, **self.dataset_kwargs)
        self.Dataset(self.data_dir, train=False, download=True, **self.dataset_kwargs)
