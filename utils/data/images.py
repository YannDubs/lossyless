import abc
import glob
import logging
import math
import os
import subprocess
import zipfile

from PIL import Image

import torch
from lossyless.helpers import BASE_LOG, get_normalization
from torch.utils.data import random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import (
    ColorJitter,
    RandomAffine,
    RandomErasing,
    RandomRotation,
)
from utils.estimators import discrete_entropy

from .base import LossylessCLFDataset, LossylessDataModule
from .helpers import int_or_ratio

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
        max_inv = index
        return img, target, max_inv

    @property
    def augmentations(self):
        return {
            "rotation": RandomRotation(60),
            "y_translation": RandomAffine(0, translate=(0.1, 0.1)),
            "x_translation": RandomAffine(0, translate=(0.1, 0.1)),
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


### Torchvision Models ###
# Base class for data module for torchvision models.
class TorchvisionDataModule(LossylessDataModule):
    def get_train_val_dataset(self, **dataset_kwargs):
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
        train, _ = self.get_train_val_dataset(**dataset_kwargs)
        return train

    def get_val_dataset(self, **dataset_kwargs):
        _, valid = self.get_train_val_dataset(**dataset_kwargs)
        return valid

    def get_test_dataset(self, **dataset_kwargs):
        test = self.Dataset(
            self.data_dir, train=False, download=False, **dataset_kwargs,
        )
        return test

    def prepare_data(self):
        self.Dataset(self.data_dir, train=True, download=True, **self.dataset_kwargs)
        self.Dataset(self.data_dir, train=False, download=True, **self.dataset_kwargs)

    @property
    def mode(self):
        return "image"


# MNIST #
class MnistDataset(LossylessImgDataset, MNIST):
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
class FashionMnistDataset(LossylessImgDataset, FashionMNIST):
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
class Cifar10Dataset(LossylessImgDataset, CIFAR10):
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


# Imagenet #
# TODO

### Non Torchvision Models ###

# Galaxy Zoo #

# TODO @karen: modify as desired all those methods
# TODO we should also add the mean and std of "galaxy" in `lossyless.helpers` to normalize the data
# TODO add config for galaxy in config/data with good defaults
class GalaxyDataset(LossylessImgDataset):
    def __init__(
        self, data_root, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # do as needed. For best compatibility with the framework
        # self.data should contain the downloaded data in tensor or numpy form
        # self.targets should contain the targets
        # self.train should say whether training

        # example:
        self.data_root = data_root
        self.download(self.data_root)
        self.data, self.targets = torch.load("path")

    def download(self, data_root):

        data_dir = os.path.join(data_root, "galaxyzoo")

        def unpack_all_zips():
            for f, file in enumerate(glob.glob(os.path.join(data_dir, "*.zip"))):
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                    os.remove(file)
                    print("{} completed. Progress: {}/6".format(file, f))

        filename = "galaxy-zoo-the-galaxy-challenge.zip"

        # check if data was already downloaded
        if os.path.exists(os.path.join(data_root, filename)):
            # continue unpacking files just in case this got interrupted
            unpack_all_zips()
            return
        # check if user has access to the kaggle API otherwise link instructions
        try:
            import kaggle
        except Exception as e:
            print(e)
            print(
                "The download of the Galaxy dataset failed. Make sure you "
                "followed the steps in https://github.com/Kaggle/kaggle-api."
            )

        # download the dataset
        bashCommand = (
            "kaggle competitions download -c "
            "galaxy-zoo-the-galaxy-challenge -p {}".format(data_root)
        )
        subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

        # unpack the data
        with zipfile.ZipFile(os.path.join(data_root, filename), "r") as zip_ref:
            zip_ref.extractall(data_dir)

        unpack_all_zips()

    @property
    def augmentations(self):
        return {
            "rotation": RandomRotation(60),
            "y_translation": RandomAffine(0, translate=(0.1, 0.1)),
            "x_translation": RandomAffine(0, translate=(0.1, 0.1)),
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
        # in regression mode (as we said) then you should stack all labels.
        # e.g. if the are 37 different regression tasks use `target=(1,37)` which says that there are 37
        # one dimensional tasks (it's the same as `target=(37,)` but averages over 6 rather than sum)
        #
        # for classification something like `target=(2,37)` means 2-class classification for 37
        # labels  (note that I use cross entropy rather than binary cross entropy. it shouldn't matter
        # besides a little more parameters right ? )
        shapes["target"] = (1, 2)
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
    def get_train_val_dataset(self, **dataset_kwargs):
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
        train, _ = self.get_train_val_dataset(**dataset_kwargs)
        return train

    def get_val_dataset(self, **dataset_kwargs):
        # if there's a validation set then do what you need here
        _, valid = self.get_train_val_dataset(**dataset_kwargs)
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

    @property
    def mode(self):
        return "image"
