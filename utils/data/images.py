import abc
import copy
import glob
import logging
import math
import os
import random
import subprocess
import zipfile
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from lossyless.helpers import BASE_LOG, Normalizer, check_import, to_numpy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, STL10,
                                  CocoCaptions, ImageFolder, ImageNet)
from torchvision.transforms import (CenterCrop, ColorJitter, Compose, Lambda,
                                    RandomAffine, RandomApply, RandomCrop,
                                    RandomErasing, RandomGrayscale,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    RandomRotation, RandomVerticalFlip, Resize,
                                    ToPILImage, ToTensor)
from utils.helpers import remove_rf

from .augmentations import (CIFAR10Policy, EquivariantRandomRotation,
                            EquivariantRandomResizedCrop, ImageNetPolicy,
                            get_finetune_augmentations,
                            get_simclr_augmentations)
from .base import LossylessDataModule, LossylessDataset
from .helpers import (Caltech101BalancingWeights, Pets37BalancingWeights,
                      download_url, image_loader, int_or_ratio, npimg_resize,
                      unzip)

try:
    import kaggle
except (ImportError, OSError):
    pass

try:
    import cv2  # only used for galaxy so skip if not needed
except ImportError:
    pass

try:
    import tensorflow_datasets as tfds  # only used for tfds data
except ImportError:
    pass

try:
    import pycocotools  # only used for coco
except ImportError:
    pass

try:
    import clip  # only used for coco
except ImportError:
    pass

EXIST_DATA = "data_exist.txt"
logger = logging.getLogger(__name__)

__all__ = [
    "Cifar10DataModule",
    "Cifar100DataModule",
    "STL10DataModule",
    "STL10UnlabeledDataModule",
    "Food101DataModule",
    "Sun397DataModule",
    "Cars196DataModule",
    "Pets37DataModule",
    "PCamDataModule",
    "Caltech101DataModule",
    "MnistDataModule",
    "GalaxyDataModule",
    "ImagenetDataModule",
    "CocoClipDataModule",
]


### HELPERS ###


### Base Classes ###
class LossylessImgDataset(LossylessDataset):
    """Base class for image datasets used for lossy compression but lossless predicitons.

    Parameters
    -----------
    equivalence : set of str, optional
        List of equivalence relationship with respect to which to be invariant (or equivariant).

    p_augment : float, optional
        Probability (in [0,1]) of applying the entire augmentation.

    is_augment_val : bool, optional
        Whether to augment the validation + test set.

    val_equivalence : set of str, optional
        List of equivalence relationship with respect to which to be invariant during evaluation.

    is_normalize : bool, optional
        Whether to normalize the input images. Only for colored images. If True, you should ensure
        that `MEAN` and `STD` and `get_normalization` and `undo_normalization` in `lossyless.helpers`
        can normalize your data.

    base_resize : {"resize","upscale_crop_eval", None}, optional
        What resizing to apply. If "resize" uses the same standard resizing during train and test.
        If "scale_crop_eval" then during test first up scale to 1.1*size and then center crop (this
        is used by SimCLR). If "clip" during test first resize such that smallest side is
        224 size and then center crops, during training ensures that image is first rescaled to smallest
        side of 256, also ensures that using "clip" normalization. If None does not perform any resizing.

    curr_split : str, optional
        Which data split you are considering.

    kwargs:
        Additional arguments to `LossylessCLFDataset` and `LossylessDataset`.
    """

    def __init__(
        self,
        *args,
        equivalence={},
        p_augment=1.0,
        is_augment_val=False,
        val_equivalence={},
        is_normalize=True,
        base_resize="resize",
        curr_split="train",
        **kwargs,
    ):

        self.base_resize = base_resize
        super().__init__(*args, is_normalize=is_normalize, **kwargs)
        self.equivalence = self.validate_equivalence(equivalence)
        self.val_equivalence = self.validate_equivalence(val_equivalence)
        self.is_augment_val = True if len(self.val_equivalence) > 0 else is_augment_val
        self.p_augment = p_augment
        self.curr_split = curr_split

    @property
    def curr_split(self):
        """Return the current split."""
        return self._curr_split

    @curr_split.setter
    def curr_split(self, value):
        """Update the current split. Also reloads correct transformation as they are split dependent."""
        self._curr_split = value

        # when updating the split has to also reload the transformations as
        self.base_tranform = self.get_base_transform()

        # these are invariances => only act on X
        self.PIL_aug, self.tensor_aug = self.get_curr_augmentations(self.augmentations)

        # these are equivariances => also act on Y
        self.joint_PIL_aug, self.joint_tensor_aug = self.get_curr_augmentations(
            self.joint_augmentations
        )

    @property
    def is_train(self):
        """Whether considering training split."""
        return self.curr_split == "train"

    @abc.abstractmethod
    def get_img_target(self, index):
        """Return the unaugmented image (in PIL format) and target."""
        ...

    @property
    @abc.abstractmethod
    def dataset_name(self):
        """Name of the dataset."""
        ...

    @classmethod  # class method property does not work before python 3.9
    def get_available_splits(cls):
        return ["train", "test"]

    def validate_equivalence(self, equivalence):
        """Check that all given augmentations are available."""
        available = dict(
            **self.augmentations["PIL"],
            **self.augmentations["tensor"],
            **self.joint_augmentations["PIL"],
            **self.joint_augmentations["tensor"],
        ).keys()
        for equiv in equivalence:
            if equiv not in available:
                raise ValueError(f"Unkown `equivalence={equiv}`.")
        return equivalence

    def get_x_target_Mx(self, index):
        """Return the correct example, target, and maximal invariant."""
        img, target = self.get_img_target(index)

        img = self.PIL_aug(img)
        img, target = self.joint_PIL_aug((img, target))

        img = self.base_tranform(img)

        img = self.tensor_aug(img)
        img, target = self.joint_tensor_aug((img, target))

        max_inv = index

        return img, target, max_inv

    @property
    def augmentations(self):
        """
        Return a dictionary of dictionaries containing all possible augmentations of interest.
        first dictionary say which kind of data they act on.
        """
        shape = self.shapes["input"]

        return dict(
            PIL={
                "rotation": RandomRotation(45),
                "360_rotation": RandomRotation(360),
                "D4_group": Compose(
                    [
                        RandomHorizontalFlip(p=0.5),
                        RandomVerticalFlip(p=0.5),
                        RandomApply([RandomRotation((90, 90))], p=0.5),
                    ]
                ),
                "y_translation": RandomAffine(0, translate=(0, 0.25)),
                "x_translation": RandomAffine(0, translate=(0.25, 0)),
                "shear": RandomAffine(0, shear=25),
                "scale": RandomAffine(0, scale=(0.6, 1.4)),
                "color": RandomApply(
                    [
                        ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                        )
                    ],
                    p=0.8,
                ),
                "gray": RandomGrayscale(p=0.2),
                "hflip": RandomHorizontalFlip(p=0.5),
                "vflip": RandomVerticalFlip(p=0.5),
                "resize_crop": RandomResizedCrop(
                    size=(shape[1], shape[2]), scale=(0.3, 1.0), ratio=(0.7, 1.4)
                ),
                "resize": Resize(min(shape[1], shape[2]), interpolation=Image.BICUBIC),
                "crop": RandomCrop(size=(shape[1], shape[2])),
                "auto_cifar10": CIFAR10Policy(),
                "auto_imagenet": ImageNetPolicy(),
                "simclr_cifar10": get_simclr_augmentations("cifar10", shape[-1]),
                "simclr_imagenet": get_simclr_augmentations("imagenet", shape[-1]),
                "simclr_finetune": get_finetune_augmentations(shape[-1]),
            },
            tensor={"erasing": RandomErasing(value=0.5),},
        )

    @property
    def joint_augmentations(self):
        """
        Return a dictortionary of dictionaries containing all possible augmentations of interest.
        first dictionary say which kind of data they act on. Augmentations for (img,label).
        """
        shape = self.shapes["input"]

        return dict(
            PIL={
                "equiv_rotation_0": EquivariantRandomRotation(equivariant_degrees=10, invariant_degrees=5, p=0),
                "equiv_rotation_0.05": EquivariantRandomRotation(equivariant_degrees=10, invariant_degrees=5, p=0.05),
                "equiv_rotation_0.2": EquivariantRandomRotation(equivariant_degrees=10, invariant_degrees=5, p=0.2),
                "equiv_rotation_0.5": EquivariantRandomRotation(equivariant_degrees=10, invariant_degrees=5, p=0.5),
                "equiv_resize_crop_0": EquivariantRandomResizedCrop(size=(shape[1], shape[2]),
                                                                    equivariant_scale=(0.3, 1.0),
                                                                    invariant_scale=(0.5, 1.0), ratio=(0.7, 1.4),
                                                                    p=0.0),
                "equiv_resize_crop_0.05": EquivariantRandomResizedCrop(size=(shape[1], shape[2]),
                                                                       equivariant_scale=(0.3, 1.0),
                                                                       invariant_scale=(0.5, 1.0), ratio=(0.7, 1.4),
                                                                       p=0.05),
                "equiv_resize_crop_0.2": EquivariantRandomResizedCrop(size=(shape[1], shape[2]),
                                                                      equivariant_scale=(0.3, 1.0),
                                                                      invariant_scale=(0.5, 1.0), ratio=(0.7, 1.4),
                                                                      p=0.2),
                "equiv_resize_crop_0.5": EquivariantRandomResizedCrop(size=(shape[1], shape[2]),
                                                                      equivariant_scale=(0.3, 1.0),
                                                                      invariant_scale=(0.5, 1.0), ratio=(0.7, 1.4),
                                                                      p=0.5),
            },
            tensor={},
        )

    def get_equiv_x(self, x, index):
        # to load equivalent image can load a same index (transformations will be different)
        img, _, __ = self.get_x_target_Mx(index)
        return img

    def get_representative(self, index):
        notaug_img, _ = self.get_img_target(index)
        notaug_img = self.base_tranform(notaug_img)
        return notaug_img

    def get_base_transform(self):
        """Return the base transform, ie train or test."""
        shape = self.shapes["input"]

        trnsfs = []

        if self.base_resize == "resize":
            trnsfs += [Resize((shape[1], shape[2]))]
        elif self.base_resize == "upscale_crop_eval":
            if not self.is_train:
                # this is what simclr does : first upscale by 10% then center crop
                trnsfs += [
                    Resize((int(shape[1] * 1.1), int(shape[2] * 1.1))),
                    CenterCrop((shape[1], shape[2])),
                ]
        elif self.base_resize == "clip":
            if not self.is_train:
                trnsfs += [
                    # resize smallest to 224
                    Resize(224, interpolation=Image.BICUBIC),
                    CenterCrop((224, 224)),
                ]
        elif self.base_resize is None:
            pass  # no resizing
        else:
            raise ValueError(f"Unkown base_resize={self.base_resize }")

        trnsfs += [ToTensor()]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            # raise if can't normalize because you specifically gave `is_normalize`
            trnsfs += [self.normalizer()]

        return Compose(trnsfs)

    def normalizer(self):
        dataset_name = "clip" if self.base_resize == "clip" else self.dataset_name
        #! normalization for clip will not affect plotting => if using vae with clip will look wrong
        return Normalizer(dataset_name, is_raise=True)

    def get_augmentations(self, augmentations):
        """Return the augmentations transorms (tuple for PIL and tensor)."""
        if (not self.is_train) and (len(self.val_equivalence) > 0):
            equivalence = self.val_equivalence
        else:
            equivalence = self.equivalence

        PIL_augment, tensor_augment = [], []
        for equiv in equivalence:
            if equiv in augmentations["PIL"]:
                PIL_augment += [augmentations["PIL"][equiv]]
            elif equiv in augmentations["tensor"]:
                tensor_augment += [augmentations["tensor"][equiv]]

        PIL_augment = RandomApply(PIL_augment, p=self.p_augment)
        tensor_augment = RandomApply(tensor_augment, p=self.p_augment)
        return PIL_augment, tensor_augment

    def get_curr_augmentations(self, augmentations):
        """Return the current augmentations transorms (tuple for PIL and tensor)."""
        if self.is_augment_val or self.is_train:
            PIL_augment, tensor_augment = self.get_augmentations(augmentations)
            return PIL_augment, tensor_augment
        else:
            identity = Compose([])
            return identity, identity

    def __len__(self):
        return len(self.data)

    @property
    def is_color(self):
        shape = self.shapes["input"]
        return shape[0] == 3

    @property
    def is_clfs(self):
        return dict(input=not self.is_color, target=True)

    @property
    def shapes(self):
        #! In each child should assign "input" and "target"
        shapes = dict()

        if self.base_resize == "clip":
            # when using clip the shape should always be 224x224
            shapes["input"] = (3, 224, 224)

        return shapes


class LossylessImgDataModule(LossylessDataModule):
    def get_train_val_dataset(self, **dataset_kwargs):
        dataset = self.Dataset(
            self.data_dir, download=False, curr_split="train", **dataset_kwargs,
        )

        n_val = int_or_ratio(self.val_size, len(dataset))
        train, valid = random_split(
            dataset,
            [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # ensure that you can change the validation dataset without impacting train
        valid.dataset = copy.deepcopy(valid.dataset)
        valid.dataset.curr_split = "validation"

        return train, valid

    def get_train_dataset(self, **dataset_kwargs):
        if "validation" in self.Dataset.get_available_splits():
            train = self.Dataset(
                self.data_dir, curr_split="train", download=False, **dataset_kwargs,
            )
        else:
            # if there is no valdation split will compute it on the fly
            train, _ = self.get_train_val_dataset(**dataset_kwargs)
        return train

    def get_val_dataset(self, **dataset_kwargs):
        if "validation" in self.Dataset.get_available_splits():
            valid = self.Dataset(
                self.data_dir,
                curr_split="validation",
                download=False,
                **dataset_kwargs,
            )
        else:
            # if there is no validation split will compute it on the fly
            _, valid = self.get_train_val_dataset(**dataset_kwargs)
        return valid

    def get_test_dataset(self, **dataset_kwargs):
        test = self.Dataset(
            self.data_dir, curr_split="test", download=False, **dataset_kwargs,
        )
        return test

    def prepare_data(self):
        for split in self.Dataset.get_available_splits():
            self.Dataset(
                self.data_dir, curr_split=split, download=True, **self.dataset_kwargs
            )

    @property
    def mode(self):
        return "image"


### Torchvision Datasets ###

# MNIST #
class MnistDataset(LossylessImgDataset, MNIST):
    FOLDER = "MNIST"

    def __init__(self, *args, curr_split="train", **kwargs):
        is_train = curr_split == "train"
        super().__init__(*args, curr_split=curr_split, train=is_train, **kwargs)

    # avoid duplicates by saving once at "MNIST" rather than at multiple  __class__.__name__
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.FOLDER, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.FOLDER, "processed")

    @property
    def shapes(self):
        shapes = super(MnistDataset, self).shapes
        shapes["input"] = shapes.get("input", (1, 32, 32))
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = MNIST.__getitem__(self, index)
        return img, target

    @property
    def dataset_name(self):
        return "MNIST"


class MnistDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return MnistDataset


# Cifar10 #
class Cifar10Dataset(LossylessImgDataset, CIFAR10):
    def __init__(self, *args, curr_split="train", **kwargs):
        is_train = curr_split == "train"
        super().__init__(*args, curr_split=curr_split, train=is_train, **kwargs)

    @property
    def shapes(self):
        shapes = super(Cifar10Dataset, self).shapes
        shapes["input"] = shapes.get("input", (3, 32, 32))
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = CIFAR10.__getitem__(self, index)
        return img, target

    @property
    def dataset_name(self):
        return "CIFAR10"


class Cifar10DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Cifar10Dataset


# Cifar100 #
class Cifar100Dataset(LossylessImgDataset, CIFAR100):
    def __init__(self, *args, curr_split="train", **kwargs):
        is_train = curr_split == "train"
        super().__init__(*args, curr_split=curr_split, train=is_train, **kwargs)

    @property
    def shapes(self):
        shapes = super(Cifar100Dataset, self).shapes
        shapes["input"] = shapes.get("input", (3, 32, 32))
        shapes["target"] = (100,)
        return shapes

    def get_img_target(self, index):
        img, target = CIFAR100.__getitem__(self, index)
        return img, target

    @property
    def dataset_name(self):
        return "CIFAR100"


class Cifar100DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Cifar100Dataset


# STL10 #
class STL10Dataset(LossylessImgDataset, STL10):
    def __init__(self, *args, curr_split="train", **kwargs):
        super().__init__(*args, curr_split=curr_split, split=curr_split, **kwargs)

    @property
    def shapes(self):
        shapes = super(STL10Dataset, self).shapes
        shapes["input"] = shapes.get("input", (3, 96, 96))
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index):
        img, target = STL10.__getitem__(self, index)
        return img, target

    @property
    def dataset_name(self):
        return "STL10"


class STL10DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return STL10Dataset


# STL10 Unlabeled #
class STL10UnlabeledDataset(STL10Dataset):
    def __init__(self, *args, curr_split="train", **kwargs):
        curr_split = "unlabeled" if curr_split == "train" else curr_split
        super().__init__(*args, curr_split=curr_split, **kwargs)


class STL10UnlabeledDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return STL10UnlabeledDataset


# Imagenet #
class ImageNetDataset(LossylessImgDataset, ImageNet):
    def __init__(
        self,
        root,
        *args,
        curr_split="train",
        base_resize="upscale_crop_eval",
        download=None,  # for compatibility
        **kwargs,
    ):
        if os.path.isdir(path.join(root, "imagenet256")):
            # use 256 if alreeady resized
            data_dir = path.join(root, "imagenet256")
        elif os.path.isdir(path.join(root, "imagenet")):
            data_dir = path.join(root, "imagenet")
        else:
            raise ValueError(
                f"Imagenet data folder (imagenet256 or imagenet) not found in {root}."
                "This has to be installed manually as download is not available anymore."
            )

        # imagenet test set is not available so it is standard to use the val split as test
        split = "val" if curr_split == "test" else curr_split

        super().__init__(
            data_dir,
            *args,
            curr_split=curr_split,  # goes to lossyless
            split=split,  # goes to iamgenet
            base_resize=base_resize,
            **kwargs,
        )

    @property
    def shapes(self):
        shapes = super(ImageNetDataset, self).shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (1000,)
        return shapes

    def get_img_target(self, index):
        img, target = ImageNet.__getitem__(self, index)
        return img, target

    @property
    def dataset_name(self):
        return "ImageNet"

    def __len__(self):
        return ImageNet.__len__(self)


class ImagenetDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return ImageNetDataset


### Tensorflow Datasets Modules ###
class TensorflowBaseDataset(LossylessImgDataset, ImageFolder):
    """Base class for tensorflow-datasets.

    Notes
    -----
    - By default will load the datasets in a format usable by CLIP.
    - Only works for square cropping for now.

    Parameters
    ----------
    root : str or Path
        Path to directory for saving data.

    split : str, optional
        Split to use, depends on data but usually ["train","test"]

    download : bool, optional
        Whether to download the data if it is not existing.

    kwargs :
        Additional arguments to `LossylessImgDataset` and `ImageFolder`.

    class attributes
    ----------------
    min_size : int, optional
        Resizing of the smaller size of an edge to a certain value. If `None` does not resize.
        Recommended for images that will be always rescaled to a smaller version (for memory gains).
        Only used when downloading.
    """

    min_size = 256

    def __init__(
        self, root, curr_split="train", download=True, base_resize="clip", **kwargs,
    ):
        check_import("tensorflow_datasets", "TensorflowBaseDataset")

        self.root = root
        self._curr_split = curr_split  #  for get dir (but cannot set curr_split yet)

        if download and not self.is_exist_data:
            self.download()

        super().__init__(
            root=self.get_dir(self.curr_split),
            base_resize=base_resize,
            curr_split=curr_split,
            **kwargs,
        )
        self.root = root  # overwirte root for which is currently split folder

    def get_dir(self, split=None):
        """Return the main directory or the one for a split."""
        main_dir = Path(self.root) / self.dataset_name
        if split is None:
            return main_dir
        else:
            return main_dir / split

    @property
    def is_exist_data(self):
        """Whether the data is available."""
        is_exist = True
        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            is_exist &= check_file.is_file()
        return is_exist

    def download(self):
        """Download the data."""
        tfds_splits = [self.to_tfds_split(s) for s in self.get_available_splits()]
        tfds_datasets, metadata = tfds.load(
            name=self.dataset_name,
            batch_size=1,
            data_dir=self.root,
            as_supervised=True,
            split=tfds_splits,
            with_info=True,
        )
        np_datasets = tfds.as_numpy(tfds_datasets)
        metadata.write_to_directory(self.get_dir())

        for split, np_data in zip(self.get_available_splits(), np_datasets):
            split_path = self.get_dir(split)
            remove_rf(split_path, not_exist_ok=True)
            split_path.mkdir()
            for i, (x, y) in enumerate(tqdm(np_data)):
                if self.min_size is not None:
                    x = npimg_resize(x, self.min_size)

                x = x.squeeze()  # given as batch of 1 (and squeeze if single channel)
                target = y.squeeze().item()

                label_name = metadata.features["label"].int2str(target)
                label_name = label_name.replace(" ", "_")
                label_name = label_name.replace("/", "")

                label_dir = split_path / label_name
                label_dir.mkdir(exist_ok=True)

                img_file = label_dir / f"{i}.jpeg"
                Image.fromarray(x).save(img_file)

        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            check_file.touch()

        # remove all downloading files
        remove_rf(Path(metadata.data_dir))

    def get_img_target(self, index):
        img, target = ImageFolder.__getitem__(self, index)
        return img, target

    def __len__(self):
        return ImageFolder.__len__(self)

    def to_tfds_split(self, split):
        """Change from a lossyless split to a tfds split."""

        if split == "validation" and ("validation" not in self.get_available_splits()):
            # when there is no validation set then the validation will come from training set
            split = "train"

        return split

    @property
    @abc.abstractmethod
    def dataset_name(self):
        """Name of datasets to load, this should be the same as found at `www.tensorflow.org/datasets/catalog/`."""
        ...


# Food101 #
class Food101Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (101,)
        return shapes

    @property
    def dataset_name(self):
        return "food101"

    def to_tfds_split(self, split):
        # validation comes from train
        renamer = dict(train="train", test="validation", validation="train")
        return renamer[split]

    @classmethod
    def get_available_splits(cls):
        return ["train", "validation"]


class Food101DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Food101Dataset


# Sun 397 #
class Sun397Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (397,)
        return shapes

    @property
    def dataset_name(self):
        return "sun397"

    @classmethod
    def get_available_splits(cls):
        return ["train", "test", "validation"]


class Sun397DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Sun397Dataset


# Cars #
class Cars196Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (196,)
        return shapes

    @property
    def dataset_name(self):
        return "cars196"


class Cars196DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Cars196Dataset


# Patch Camelyon #
class PCamDataset(TensorflowBaseDataset):
    min_size = None

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 96, 96))
        shapes["target"] = (2,)
        return shapes

    @property
    def dataset_name(self):
        return "patch_camelyon"

    @classmethod
    def get_available_splits(cls):
        return ["train", "test", "validation"]


class PCamDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return PCamDataset


# note: not using flowers 102 dataset due to
# https://github.com/tensorflow/datasets/issues/3022

# Pets 37 #
class Pets37Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (37,)
        return shapes

    @property
    def dataset_name(self):
        return "oxford_iiit_pet"


class Pets37DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Pets37Dataset

    @property
    def balancing_weights(self):
        return Pets37BalancingWeights  # should compute mean acc per class


# Caltech 101 #
class Caltech101Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = (102,)  # ?!? there are 102 classes in caltech 101
        return shapes

    @property
    def dataset_name(self):
        return "caltech101"


class Caltech101DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Caltech101Dataset

    @property
    def balancing_weights(self):
        return Caltech101BalancingWeights  # should compute mean acc per class


### Other Datasets ###
class ExternalImgDataset(LossylessImgDataset):
    """Base class for external datasets that are neither torchvision nor tensorflow. Images will be
    saved as jpeg.

    Parameters
    ----------
    root : str or Path
        Base path to directory for saving data.

    download : bool, optional
        Whether to download the data if it does not exist.

    kwargs :
        Additional arguments to `LossylessImgDataset`.

    class attributes
    ----------------
    min_size : int, optional
        Resizing of the smaller size of an edge to a certain value. If `None` does not resize.
        Recommended for images that will be always rescaled to a smaller version (for memory gains).
        Only used when downloading.
    """

    min_size = 256
    required_packages = []

    def __init__(
        self, root, *args, download=True, curr_split="train", **kwargs,
    ):
        for p in self.required_packages:
            check_import(p, type(self).__name__)

        self.root = Path(root)
        if download and not self.is_exist_data:
            self.download_extract()
            self.preprocess()

        self.load_data_(curr_split)
        self.length = len(list(self.get_dir(curr_split).glob("*.jpeg")))

        super().__init__(
            *args, curr_split=curr_split, **kwargs,
        )

    def get_dir(self, split=None):
        """Return the main directory or the one for a split."""
        if split == "validation":
            split = "train"  # validation split comes from train

        main_dir = Path(self.root) / self.dataset_name
        if split is None:
            return main_dir
        else:
            return main_dir / split

    @property
    def is_exist_data(self):
        """Whether the data is available."""
        is_exist = True
        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            is_exist &= check_file.is_file()
        return is_exist

    def download_extract(self):
        logger.info(f"Downloading {self.dataset_name} ...")

        data_dir = self.get_dir()
        remove_rf(data_dir, not_exist_ok=True)
        data_dir.mkdir(parents=True)

        self.download(data_dir)

        logger.info(f"Extracting {self.dataset_name} ...")

        zips = list(data_dir.glob("*.zip"))
        # while loop for recursing in case zips of zips
        while len(zips) > 0:
            for filename in zips:
                logger.info(f"Unzipping {filename}")
                unzip(filename)
            zips = list(data_dir.glob("*.zip"))

        logger.info(f"{self.dataset_name} successfully pre-processed.")

    def preprocess(self):
        for split in self.get_available_splits():
            logger.info(f"Preprocessing {self.dataset_name} split={split}.")
            split_path = self.get_dir(split)

            remove_rf(split_path, not_exist_ok=True)
            split_path.mkdir()

            to_rm = self.preprocess_split(split)

            check_file = split_path / EXIST_DATA
            check_file.touch()

            for f in to_rm:
                # remove all files and directories that are not needed
                remove_rf(f)

    def __len__(self):
        return self.length

    @classmethod
    def get_available_splits(cls):
        return ["test", "train"]

    @property
    def preprocessing_resizer(self):
        """Resizing function for preprocessing step."""
        if self.min_size is None:
            return Compose([])
        else:
            return Resize(self.min_size)

    @abc.abstractmethod
    def download(self, data_dir):
        """Actual downloading of the dataset to `data_dir`."""
        ...

    @abc.abstractmethod
    def preprocess_split(self, split):
        """Preprocesses the current split, and return all the files that can be removed fpr that split."""
        ...

    @abc.abstractmethod
    def load_data_(self, split):
        """Loads data if needed."""
        ...


# Galaxy Zoo #
class GalaxyDataset(ExternalImgDataset):
    """Galaxy Zoo 2 Dataset.

    Notes
    -----
    - See challenge: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
    - Winning solution on kaggle (and tricks for augmentations):
      http://benanne.github.io/2014/04/05/galaxy-zoo.html
    - The images will be center cropped to 256x256 after downloading, this is slightly larger than
    the 207 used by winning strategy (so that can be used by standard SSL models).

    Parameters
    ----------
    resolution : int, optional
        Square size of the image to work with. The winning strategy uses resizes to 69 and then
        crops to 45, so effectively works with 45.

    arg, kwargs :
        Arguments to `LossylessImgDataset`.
    """

    required_packages = ["cv2", "kaggle"]

    split_to_root = dict(test="images_test_rev1", train="images_training_rev1",)

    def __init__(
        self,
        *args,
        resolution=128,
        is_normalize=False,  # do not normalize because little variance (black ++)
        **kwargs,
    ):
        self.resolution = resolution

        # kaggle is not straightforward to install => add more details about error
        try:
            import kaggle
        except Exception as e:
            logger.critical(
                "Cannot import Kaggle which is needed for GalaxyDataset. Make sure you "
                "followed the steps in https://github.com/Kaggle/kaggle-api."
            )
            raise e

        super().__init__(
            *args, is_normalize=is_normalize, **kwargs,
        )

    def download(self, data_dir):
        kaggle.api.competition_download_files(
            "galaxy-zoo-the-galaxy-challenge", path=data_dir, quiet=False
        )

    def preprocess_split(self, split):

        data_dir = self.get_dir()
        split_path = self.get_dir(split)

        # SAVE IDs
        raw_paths = list(data_dir.glob(f"{self.split_to_root[split]}/*.jpg"))
        ids = [int(f.stem) for f in raw_paths]
        np.save(data_dir / f"{split}_ids", ids)

        # SAVE LABELS
        if split == "train":
            # no test labels
            df = pd.read_csv(data_dir / "training_solutions_rev1.csv")
            targets = [
                df.loc[df["GalaxyID"] == id].values[:, 1:].astype("float32")
                for id in ids
            ]
            np.save(data_dir / "train_targets", np.array(targets))

        # resize to all images if needed
        for i, path in enumerate(tqdm(raw_paths)):
            img = image_loader(path)
            img = self.preprocessing_resizer(img)
            img.save(split_path / f"{i}th_img.jpeg")

        files_to_rm = [data_dir / self.split_to_root[split]]
        return files_to_rm

    def load_data_(self, split):
        data_dir = self.get_dir()
        if split == "test":
            # We do not have test targets bc kaggle holds them back. We will
            # later need the image IDs to make a submission file that will be
            # evaluated via the kaggle api.
            self.ids = np.load(data_dir / f"{split}_ids.npy")
        else:
            self.targets = np.load(data_dir / f"{split}_targets.npy")

    @property
    def is_clfs(self):
        is_clf = super().is_clfs
        is_clf["target"] = False  # treated as regression task
        return is_clf

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = (3, self.resolution, self.resolution)
        # (1,37) instead of (37,) because those are 37 different tasks => want mean/std over tasks
        shapes["target"] = (1, 37)
        return shapes

    def get_img_target(self, index):
        split_path = self.get_dir(self.curr_split)
        img = image_loader(split_path / f"{index}th_img.jpeg")
        return img, self.targets[index]

    @property
    def dataset_name(self):
        return "galaxy"

    @property
    def augmentations(self):
        # TODO remove if we don't end up using those
        augmentations = super().augmentations

        # these are the augmentations used in kaggle
        PIL_update = {
            # in kaggle authors translate 69x69 images by /pm 4 pixel = 11.6%
            "y_translation": RandomAffine(0, translate=(0, 0.116)),
            "x_translation": RandomAffine(0, translate=(0.116, 0)),
            "scale": RandomAffine(0, scale=(1.0 / 1.3, 1.3)),
        }
        augmentations["PIL"].update(PIL_update)
        return augmentations


class GalaxyDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return GalaxyDataset


# MS Coco caption dataset with clip sentences #
class CocoClipDataset(ExternalImgDataset):
    """MSCOCO caption dataset where the captions are featurized by CLIP.

    Parameters
    ----------
    args, kwargs :
        Additional arguments to `LossylessImgDataset`.
    """

    required_packages = ["pycocotools", "clip"]

    urls = [
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/train2017.zip",
    ]

    # test annotation are not given => use val instead
    split_to_root = dict(test="val2017", train="train2017",)
    split_to_annotate = dict(
        test="annotations/captions_val2017.json",
        train="annotations/captions_train2017.json",
    )

    def __init__(
        self, *args, base_resize="clip", **kwargs,
    ):
        super().__init__(
            *args, base_resize=base_resize, **kwargs,
        )

    def download(self, data_dir):
        for url in self.urls:
            logger.info(f"Downloading {url}")
            download_url(url, data_dir)

    def preprocess_split(self, split):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        entire_model, _ = clip.load("ViT-B/32", device)
        to_pil = ToPILImage()

        split_path = self.get_dir(split)

        old_root = self.get_dir() / self.split_to_root[split]
        old_annotate = self.get_dir() / self.split_to_annotate[split]

        dataset = CocoCaptions(
            root=old_root,
            annFile=old_annotate,
            transform=Compose([self.preprocessing_resizer, ToTensor()]),
            target_transform=Lambda(lambda texts: clip.tokenize([t for t in texts])),
        )

        with torch.no_grad():
            for i, (images, texts) in enumerate(
                tqdm(DataLoader(dataset, batch_size=1, num_workers=0))
            ):
                image = to_pil(images.squeeze(0))
                text_in = texts.squeeze(0).to(device)
                text_features = to_numpy(entire_model.encode_text(text_in))

                image.save(split_path / f"{i}th_img.jpeg")
                np.save(split_path / f"{i}th_features.npy", text_features)

        files_to_rm = [old_root, old_annotate]
        return files_to_rm

    @property
    def shapes(self):
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 224, 224))
        shapes["target"] = None  # no classification
        return shapes

    def get_img_target(self, index):
        split_path = self.get_dir(self.curr_split)
        img = image_loader(split_path / f"{index}th_img.jpeg")
        return img, -1  # target -1 means missing for torcvhvision (see stl10)

    def get_equiv_x(self, x, index):
        # to get an x from the same equivalence class just return one the texts (already featurized)
        split_path = self.get_dir(self.curr_split)
        text_features = np.load(split_path / f"{index}th_features.npy")

        # index to select (multiple possible sentences)
        selected_idx = torch.randint(text_features.shape[0] - 1, (1,)).item()

        return text_features[selected_idx]

    def load_data_(self, curr_split):
        # no data needed to be loaded
        pass

    @property
    def dataset_name(self):
        return "coco_captions"


class CocoClipDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return CocoClipDataset
