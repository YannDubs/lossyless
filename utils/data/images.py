import logging
from PIL import Image

import torch

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import os
import subprocess
import zipfile
import glob
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

__all__ = ["Cifar10DataModule", "MnistDataModule", "FashionMnistDataModule"]


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

    kwargs:
        Additional arguments to `LossylessCLFDataset` and `LossylessDataset`.
    """

    def __init__(
        self, *args, equivalence={}, is_augment_val=False, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.equivalence = equivalence
        self.is_augment_val = is_augment_val

        self.val_tranform = self.get_val_transform()  # only val
        self.transform = self.get_transform()  # current transforms
        self._tranform = self.transform  # copy current transforms

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

    def get_representative(self, Mx):
        # Mx is the index of the underlying image
        self.transform = self.val_tranform  # deactivate transforms
        notaug_img, _ = super().__getitem__(Mx)
        self.transform = self._tranform  # reactivate transforms
        return notaug_img

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

    def get_val_transform(self):
        """Return the transform for validation set."""
        shape = self.shapes_x_t_Mx["input"]
        trnsfs = [
            transform_lib.Resize((shape[1], shape[2])),
            transform_lib.ToTensor(),
        ]

        if shape[0] == 3:
            # only normalize colored images
            trnsfs += [get_normalization(self.Dataset)]

        return transform_lib.Compose(trnsfs)

    def get_transform(self):
        """Return the current transforms."""
        trnsfs = []
        trnsfs.append(self.get_val_transform())

        if self.is_augment_val or self.train:
            trnsfs.append(self.sample_equivalence_action())

        return transform_lib.Compose(trnsfs)

    def __len__(self):
        return len(self.data)


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
        self.dataset_test = self.Dataset(
            self.data_dir, train=False, download=False, **self.dataset_kwargs,
        )

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
    def is_clf_x_t_Mx(self):
        return dict(input=True, target=True, max_inv=True)

    @property
    def shapes_x_t_Mx(self):
        return dict(input=(1, 32, 32), target=(10,), max_inv=(len(self),))

    def get_x_target_Mx(self, index):
        img, target = MNIST.__getitem__(self, index)
        Mx = index
        return img, target, Mx


class MnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return MnistDataset


# Fasion MNIST #
class ToyFashionMnistDataset(LossylessImgAnalyticDataset, FashionMNIST):
    FOLDER = "FashionMNIST"

    def get_x_target_Mx(self, index):
        img, target = MNIST.__getitem__(self, index)
        Mx = index
        return img, target, Mx

    processed_folder = MnistDataset.processed_folder
    raw_folder = MnistDataset.raw_folder
    is_clf_x_t_Mx = MnistDataset.is_clf_x_t_Mx
    shapes_x_t_Mx = MnistDataset.shapes_x_t_Mx


class FashionMnistDataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return ToyFashionMnistDataset


# Cifar10 #
class Cifar10Dataset(LossylessImgAnalyticDataset, CIFAR10):
    @property
    def is_clf_x_t_Mx(self):
        return dict(input=False, target=True, max_inv=True)

    @property
    def shapes_x_t_Mx(self):
        return dict(input=(3, 32, 32), target=(10,), max_inv=(len(self),))

    def get_x_target_Mx(self, index):
        img, target = CIFAR10.__getitem__(self, index)
        Mx = index
        return img, target, Mx


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
class GalaxyDataset(LossylessImgAnalyticDataset):
    def __init__(
        self,
        data_root,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # do as needed. For best compatibility with the framework
        # self.data should contain the downloaded data in tensor or numpy form
        # self.targets should contain the targets

        # example:
        self.data_root = data_root
        self.download(self.data_root)
        self.data, self.targets = torch.load("path")

    def download(self, data_root):

        data_dir = os.path.join(data_root, "galaxyzoo")

        def unpack_all_zips():
            for f, file in enumerate(
                glob.glob(os.path.join(data_dir, "*.zip"))):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                    os.remove(file)
                    print("{} completed. Progress: {}/6".format(file, f))

        filename = "galaxy-zoo-the-galaxy-challenge.zip"

        # check if data was already downloaded
        if os.path.exists(os.path.join(data_root,filename)):
            # continue unpacking files just in case this got interrupted
            unpack_all_zips()
            return
        # check if user has access to the kaggle API otherwise link instructions
        try:
            import kaggle
        except Exception as e:
            print(e)
            print("The download of the Galaxy dataset failed. Make sure you "
                  "followed the steps in https://github.com/Kaggle/kaggle-api.")

        # download the dataset
        bashCommand = "kaggle competitions download -c " \
                      "galaxy-zoo-the-galaxy-challenge -p {}".format(data_root)
        subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

        # unpack the data
        with zipfile.ZipFile(os.path.join(data_root,filename), 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        unpack_all_zips()


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
        # target should be True if log loss (ie classification) and False if MSE (ie regression)
        # input should be true is using log loss for reconstruction (typically MNIST) and False if MSE (typically colored images)
        # leave max inv true (because ``max_inv` classifyies the indices)
        return dict(input=True, target=False, max_inv=True)

    @property
    def shapes_x_t_Mx(self):
        # input is shape image
        # target is shape of target. This will depend as to if we are using classfication or regression
        # in regression mode (as we said) `target=(3,)` means that there are 6 values to predict
        # (it's equivalent to `target=(1,1,1)` it depends how the targets are formatted)
        # for classification `target=(3,)` means 3-class classification and `target=(3,2)` means
        # multi label classification, one with 3-classes and one with 2-classes
        # (I still have to implement multilabel classification but will be soon)
        return dict(input=(3, 32, 32), target=(6,), max_inv=(len(self),))

    def get_x_target_Mx(self, index):
        # change as needed but something like that
        img = self.images[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        Mx = index
        return img, target, Mx


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
        self.dataset_test = self.Dataset(
            self.data_dir, train=False, download=False, **self.dataset_kwargs,
        )

    def prepare_data(self):
        # this is where the downlading should happen if we can if not just put `pass`
        self.Dataset(self.data_dir, train=True, download=True, **self.dataset_kwargs)
        self.Dataset(self.data_dir, train=False, download=True, **self.dataset_kwargs)
