import abc
import glob
import logging
import math
import os
import subprocess
import zipfile
from os import path
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from lossyless.helpers import BASE_LOG, Normalizer, check_import
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, STL10, ImageFolder,
                                  ImageNet)
from torchvision.transforms import (ColorJitter, Compose, RandomAffine,
                                    RandomApply, RandomErasing,
                                    RandomGrayscale, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation,
                                    RandomVerticalFlip)
from tqdm import tqdm
from utils.estimators import discrete_entropy
from utils.helpers import remove_rf

from .augmentations import (CIFAR10Policy, ImageNetPolicy,
                            get_finetune_augmentations,
                            get_simclr_augmentations)
from .base import LossylessDataModule, LossylessDataset
from .helpers import int_or_ratio, npimg_resize

try:
    import cv2  # only used for galaxy so skip if not needed
except ImportError:
    pass

try:
    import tensorflow_datasets as tfds  # only used for tfds data
except ImportError:
    pass


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
    "Flowers102DataModule",
    "MnistDataModule",
    "GalaxyDataModule",
    "ImagenetDataModule",
]


### HELPERS ###


### Base Classes ###
class LossylessImgDataset(LossylessDataset):
    """Base class for image datasets used for lossy compression but lossless predicitons.

    Parameters
    -----------
    equivalence : set of str, optional
        List of equivalence relationship with respect to which to be invariant.

    p_augment : float, optional
        Probability (in [0,1]) of applying the entire augmentation.

    is_augment_val : bool, optional
        Whether to augment the validation + test set.

    is_normalize : bool, optional
        Whether to normalize the input images. Only for colored images. If True, you should ensure
        that `MEAN` and `STD` and `get_normalization` and `undo_normalization` in `lossyless.helpers`
        can normalize your data.

    base_resize : {"resize","crop_eval", "upscale_crop_eval", None}, optional
        What resizing to apply. If "resize" uses the same standard resizing during train and test.
        If "crop_eval" during test first resize such that smallest side is correct size and then
        center crops but nothing at training time (this is used by CLIP). If "scale_crop_eval"
        then during test first up scale to 1.1*size and then center crop (this is used by SimCLR).
        If None does not perform any resizing.

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
        is_normalize=True,
        base_resize="resize",
        curr_split="train",
        **kwargs,
    ):
        super().__init__(*args, is_normalize=is_normalize, **kwargs)
        self.equivalence = equivalence
        self.is_augment_val = is_augment_val
        self.base_resize = base_resize
        self.curr_split = curr_split
        self.p_augment = p_augment

        self.base_tranform = self.get_base_transform()
        self.PIL_augment, self.tensor_augment = self.get_curr_augmentations()

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
    def get_available_splits(self):
        return ["train", "test"]

    def get_x_target_Mx(self, index):
        """Return the correct example, target, and maximal invariant."""
        img, target = self.get_img_target(index)
        img = self.PIL_augment(img)
        img = self.base_tranform(img)
        img = self.tensor_augment(img)
        max_inv = index
        return img, target, max_inv

    @property
    def augmentations(self):
        """
        Return a dictionary of dictionaries containing all possible augmentations of interest.
        first dictionary say which kind of data they act on.
        """
        shape = self.shapes_x_t_Mx["input"]
        return dict(
            PIL={
                "rotation": RandomRotation(30),
                "y_translation": RandomAffine(0, translate=(0, 0.1)),
                "x_translation": RandomAffine(0, translate=(0.1, 0)),
                "shear": RandomAffine(0, shear=10),
                "scale": RandomAffine(0, scale=(0.8, 1.2)),
                "rotation++": RandomRotation(45),
                "360_rotation": RandomRotation(360),
                "y_translation++": RandomAffine(0, translate=(0, 0.25)),
                "x_translation++": RandomAffine(0, translate=(0.25, 0)),
                "shear++": RandomAffine(0, shear=25),
                "scale++": RandomAffine(0, scale=(0.6, 1.4)),
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
                "auto_cifar10": CIFAR10Policy(),
                "auto_imagenet": ImageNetPolicy(),
                # NB you should use those 3 also at eval time
                "simclr_cifar10": get_simclr_augmentations("cifar10", shape[-1]),
                "simclr_imagenet": get_simclr_augmentations("imagenet", shape[-1]),
                "simclr_finetune": get_finetune_augmentations(),
            },
            tensor={"erasing": RandomErasing(value=0.5),},
        )

    def get_equiv_x(self, x, index):
        # to load equivalent image can load a same index (transformations will be different)
        img, _, __ = self.get_x_target_Mx(index)
        return img

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

        trnsfs = []

        if self.base_resize == "resize":
            trnsfs += [transform_lib.Resize((shape[1], shape[2]))]
        elif self.base_resize == "upscale_crop_eval":
            if not self.is_train:
                # this is what simclr does : first upscale by 10% then center crop
                trnsfs += [
                    transform_lib.Resize((int(shape[1] * 1.1), int(shape[2] * 1.1))),
                    transform_lib.CenterCrop((shape[1], shape[2])),
                ]
        elif self.base_resize == "crop_eval":
            if not self.is_train:
                # this is what CLIP does : scale small side and then center crop
                trnsfs += [
                    transform_lib.Resize(
                        (shape[1], shape[2]), interpolation=Image.BICUBIC
                    ),
                    transform_lib.CenterCrop((shape[1], shape[2])),
                ]
        elif self.base_resize is None:
            pass  # no resizing
        else:
            raise ValueError(f"Unkown base_resize={self.base_resize }")

        trnsfs = [transform_lib.ToTensor()]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            # raise if can't normalize because you specifrically gave `is_normalize`
            trnsfs += [self.normalizer()]

        return transform_lib.Compose(trnsfs)

    def normalizer(self):
        return Normalizer(self.dataset_name, is_raise=True)

    def get_augmentations(self):
        """Return the augmentations transorms (tuple for PIL and tensor)."""
        PIL_augment, tensor_augment = [], []
        for equiv in self.equivalence:
            if equiv in self.augmentations["PIL"]:
                PIL_augment += [self.augmentations["PIL"][equiv]]
            elif equiv in self.augmentations["tensor"]:
                tensor_augment += [self.augmentations["tensor"][equiv]]
            else:
                raise ValueError(f"Unkown `equivalence={equiv}`.")

        PIL_augment = RandomApply(Compose(PIL_augment),p=self.p_augment)
        tensor_augment = RandomApply(Compose(tensor_augment),p=self.p_augment)
        return PIL_augment, tensor_augment

    def get_curr_augmentations(self):
        """Return the current augmentations transorms (tuple for PIL and tensor)."""
        if self.is_augment_val or self.is_train:
            PIL_augment, tensor_augment = self.get_augmentations()
            return PIL_augment, tensor_augment
        else:
            identity = transform_lib.Compose([])
            return identity, identity

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
        valid.dataset.curr_split = "validation"

        return train, valid

    def get_train_dataset(self, **dataset_kwargs):
        train, _ = self.get_train_val_dataset(**dataset_kwargs)
        return train

    def get_val_dataset(self, **dataset_kwargs):
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
    def shapes_x_t_Mx(self):
        shapes = super(MnistDataset, self).shapes_x_t_Mx
        shapes["input"] = (1, 32, 32)
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
    def shapes_x_t_Mx(self):
        shapes = super(Cifar10Dataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 32, 32)
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
    def shapes_x_t_Mx(self):
        shapes = super(Cifar100Dataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 32, 32)
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
    def shapes_x_t_Mx(self):
        shapes = super(STL10Dataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 96, 96)
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
        download=None,  # # for compatibility
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
    def shapes_x_t_Mx(self):
        shapes = super(ImageNetDataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 224, 224)
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
    """Base class for tensorflow-datasets. This will download using save as hdf5.

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

    n_pixels : int, optional
        Size of square crops to take.

    is_clip_normalization : bool, optional
        Whether to use default CLIP normalization (i.e. ~ real image) rather than dataset specific.
        This currently does not work with direct distortion when unormalization has to be done.

    class attributes
    ----------------
    min_size : int, optional
        Resizing of the smaller size of an edge to a certain value. If `None` does not resize.
        Recommended for images that will be always rescaled to a smaller version (for memory gains).
        Only used when downloading.
    """

    min_size = 256
    CHECK_FILENAME = "tfds_exist.txt"

    def __init__(
        self,
        root,
        curr_split="train",
        download=True,
        n_pixels=224,
        is_clip_normalization=True,
        base_resize="crop_eval",
        **kwargs,
    ):
        check_import("tensorflow_datasets", "TensorflowBaseDataset")

        self.root = root
        self.curr_split = curr_split
        self._length = None
        self.n_pixels = n_pixels
        self.is_clip_normalization = is_clip_normalization

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
            check_file = self.get_dir(split) / self.CHECK_FILENAME
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

                label_dir = split_path / label_name
                label_dir.mkdir(exist_ok=True)

                img_file = label_dir / f"{i}.jpeg"
                Image.fromarray(x).save(img_file)

        for split in self.get_available_splits():
            check_file = self.get_dir(split) / self.CHECK_FILENAME
            check_file.touch()

        # remove all downloading files
        remove_rf(Path(metadata.data_dir))

    def get_img_target(self, index):
        img, target = ImageFolder.__getitem__(self, index)
        return img, target

    def normalizer(self):
        data_normalize = "CLIP" if self.is_clip_normalization else self.dataset_name
        return Normalizer(data_normalize, is_raise=True)

    def __len__(self):
        return ImageFolder.__len__(self)

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["input"] = (3, self.n_pixels, self.n_pixels)
        #! in each child should assign target
        return shapes

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
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (101,)
        return shapes

    @property
    def dataset_name(self):
        return "food101"

    def to_tfds_split(self, split):
        # validation comes from train
        renamer = dict(train="train", test="validation", validation="train")
        return renamer[split]


class Food101DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Food101Dataset


# Sun 397 #
class Sun397Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (397,)
        return shapes

    @property
    def dataset_name(self):
        return "sun397"

    @classmethod
    def get_available_splits(self):
        return ["train", "test", "validation"]


class Sun397DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Sun397Dataset


# Cars #
class Cars196Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (196,)
        return shapes

    @property
    def dataset_name(self):
        return "cars196"


class Cars196DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Cars196Dataset


# Path Camelyon #
class PCamDataset(TensorflowBaseDataset):
    min_size = None

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (2,)
        return shapes

    @property
    def dataset_name(self):
        return "patch_camelyon"

    @property
    def available_split(self):
        return ["train", "test", "validation"]


class PCamDataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return PCamDataset


# Flowers 102 #
class Flowers102Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (102,)
        return shapes

    @property
    def dataset_name(self):
        return "oxford_flowers102"

    @property
    def available_split(self):
        return ["train", "test", "validation"]


class Flowers102DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Flowers102Dataset


# Pets 37 #
class Pets37Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (37,)
        return shapes

    @property
    def dataset_name(self):
        return "oxford_iiit_pet"

    @property
    def available_split(self):
        return ["train", "test"]


class Pets37DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Pets37Dataset


# Caltech 101 #
class Caltech101Dataset(TensorflowBaseDataset):
    min_size = 256

    @property
    def shapes_x_t_Mx(self):
        shapes = super().shapes_x_t_Mx
        shapes["target"] = (101,)
        return shapes

    @property
    def dataset_name(self):
        return "caltech101"

    @property
    def available_split(self):
        return ["train", "test"]


class Caltech101DataModule(LossylessImgDataModule):
    @property
    def Dataset(self):
        return Caltech101Dataset


### Other Datasets ###

# Galaxy Zoo #
class GalaxyDataset(LossylessImgDataset):
    def __init__(
        self,
        root: str,
        *args,
        curr_split: str = "train",
        download: bool = True,
        resolution: int = 64,
        **kwargs,
    ):
        check_import("cv2", "GalaxyDataset")

        self.root = root
        self.curr_split = curr_split
        data_dir = path.join(root, "galaxyzoo")
        self.resolution = resolution
        if download and not self.is_exist_data(data_dir):
            self.download(data_dir)
            self.preprocess(data_dir)

        self.load_data(data_dir, curr_split, resolution)

        super().__init__(*args, curr_split=curr_split, **kwargs)

    def is_exist_data(self, data_dir):
        # check if we already preprocessed and downloaded the data
        # this is a hacky check, we just check if the last file that this
        # routine yields exists
        return path.exists(path.join(data_dir, "test_images_128.npy"))

    def download(self, data_dir):
        def unpack_all_zips():
            for f, file in enumerate(glob.glob(path.join(data_dir, "*.zip"))):
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                    os.remove(file)
                    logger.info("{} completed. Progress: {}/6".format(file, f))

        filename = "galaxy-zoo-the-galaxy-challenge.zip"

        # check if data was already downloaded
        if path.exists(path.join(self.root, filename)):
            # continue unpacking files just in case this got interrupted or user
            # downloaded files manually. you never know :)
            unpack_all_zips()
            return
        # check if user has access to the kaggle API otherwise link instructions
        try:
            import kaggle
        except Exception as e:
            logger.critical(
                "Cannot import Kaggle which is needed for GalaxyDataset. Make sure you "
                "followed the steps in https://github.com/Kaggle/kaggle-api."
            )
            raise e

        logger.info("Downloading Galaxy ...")

        # download the dataset
        bashCommand = (
            "kaggle competitions download -c "
            "galaxy-zoo-the-galaxy-challenge -p {}".format(self.root)
        )
        process = subprocess.Popen(
            bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        out = process.communicate()[0].decode("utf-8")
        is_error = process.returncode != 0
        if is_error:
            logging.critical(
                f"{bashCommand} failed with outputs: {out}. \n "
                "Hint: don't forget to accept competition rules at "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/rules. "
            )

        # unpack the data
        with zipfile.ZipFile(os.path.join(self.root, filename), "r") as zip_ref:
            zip_ref.extractall(data_dir)

        unpack_all_zips()

    def preprocess(self, data_dir):

        # SAVE IMAGE IDs
        def get_image_ids(image_dir):
            raw_file_paths = glob.glob(path.join(data_dir, image_dir))
            ids = [int(f.split("/")[-1].split(".")[0]) for f in raw_file_paths]
            return ids, raw_file_paths

        # test set
        test_ids, test_paths = get_image_ids("images_test_rev1/*.jpg")
        np.save(path.join(data_dir, "test_ids"), test_ids)

        # train and valid set
        ids, paths = get_image_ids("images_training_rev1/*.jpg")
        # make fixed train / valid split
        num_valid = len(ids) // 10
        assert num_valid == 6157, (
            "Validation set is not the right size. " "Oeef. That is not good."
        )
        valid_ids, valid_paths = ids[:num_valid], paths[:num_valid]
        train_ids, train_paths = ids[num_valid:], paths[num_valid:]
        np.save(path.join(data_dir, "valid_ids"), valid_ids)
        np.save(path.join(data_dir, "train_ids"), train_ids)

        # SAVE TRAIN LABELS
        df = pd.read_csv(path.join(data_dir, "training_solutions_rev1.csv"))

        for split, ids in [("train", train_ids), ("valid", valid_ids)]:
            targets = [
                df.loc[df["GalaxyID"] == id].values[:, 1:].astype("float32")
                for id in ids
            ]
            np.save(path.join(data_dir, split + "_targets"), np.array(targets))

        # PRE-PROCESSING IMAGES
        ORIG_SHAPE = (424, 424)
        CROP_SIZE = (384, 384)
        x1 = (ORIG_SHAPE[0] - CROP_SIZE[0]) // 2
        y1 = (ORIG_SHAPE[1] - CROP_SIZE[1]) // 2

        def get_image(path, out_shape):
            x = cv2.imread(path)
            x = x[x1 : x1 + CROP_SIZE[0], y1 : y1 + CROP_SIZE[1]]
            x = cv2.resize(x, dsize=out_shape, interpolation=cv2.INTER_LINEAR)
            x = np.transpose(x, (2, 0, 1))
            return x

        for out_shape in [(64, 64), (128, 128)]:
            res = str(out_shape[0])

            for (split, raw_paths) in [
                ("train", train_paths),
                ("valid", valid_paths),
                ("test", test_paths),
            ]:
                preprocessed_images = []
                for i, p in enumerate(raw_paths):
                    logger.info(
                        "Processed {}/{} images in {} split with resolution "
                        "{}x{}.".format(i, len(raw_paths), split, res, res)
                    )
                    preprocessed_images.append(get_image(p, out_shape))

                out = np.array(preprocessed_images)
                out_path = split + "_images_" + res
                np.save(path.join(data_dir, out_path), out)
                if split == "train":
                    out = out.astype("float32") / 255.0
                    mean = np.mean(out, axis=(0, 2, 3))
                    np.save(path.join(data_dir, out_path + "_mean"), mean)
                    std = np.std(out, axis=(0, 2, 3))
                    np.save(path.join(data_dir, out_path + "_std"), std)

        logger.info("Galaxy data successfully pre-processed.")

    def load_data(self, data_dir, split, resolution):
        imgs = np.load(
            path.join(data_dir, "{}_images_{}.npy".format(split, resolution))
        )
        self.data = imgs.astype("float32") / 255.0

        if not split == "test":
            self.targets = np.load(path.join(data_dir, split + "_targets.npy"))
        else:
            # We do not have test targets bc kaggle holds them back. We will
            # later need the image IDs to make a submission file that will be
            # evaluated via the kaggle api.
            self.ids = np.load(path.join(data_dir, split + "_ids.npy"))

    @property
    def is_clf_x_t_Mx(self):
        is_clf = super(GalaxyDataset, self).is_clf_x_t_Mx
        # input should be true is using log loss for reconstruction (typically MNIST) and False if MSE (typically colored images)
        is_clf["input"] = False
        # target should be True if log loss (ie classification) and False if MSE (ie regression)
        is_clf["target"] = False
        return is_clf

    @property
    def shapes_x_t_Mx(self):
        shapes = super(GalaxyDataset, self).shapes_x_t_Mx
        # input is shape image
        shapes["input"] = (3, self.resolution, self.resolution)
        # target is shape of target. This will depend as to if we are using classfication or regression
        # in regression mode (as we said) then you should stack all labels.
        # e.g. if the are 37 different regression tasks use `target=(1,37)` which says that there are 37
        # one dimensional tasks (it's the same as `target=(37,)` but averages over 37 rather than sum)
        #
        # for classification something like `target=(2,37)` means 2-class classification for 37
        # labels  (note that I use cross entropy rather than binary cross entropy. it shouldn't matter
        # besides a little more parameters right ? )
        shapes["target"] = (1, 37)
        return shapes

    def get_img_target(self, index):
        # change as needed but something like that
        img = self.data[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # equivalent: Image.fromarray((img.transpose(1,2,0)*255).astype(np.uint8))
        img = transform_lib.functional.to_pil_image(torch.from_numpy(img), None)

        # don't apply transformation yet, it's done for you

        return img, target

    @property
    def dataset_name(self):
        return f"galaxy{self.resolution}"

    @property
    def augmentations(self):
        # TODO remove if we don't end up using those
        augmentations = super().augmentations()

        # these are the augmentations used in kaggle
        PIL_update ={
                # in kaggle authors translate 69x69 images by /pm 4 pixel = 11.6%
                "y_translation": RandomAffine(0, translate=(0, 0.116)),
                "x_translation": RandomAffine(0, translate=(0.116, 0)),
                "scale": RandomAffine(0, scale=(1.0/1.3, 1.3)),
            }
        augmentations["PIL"].update(PIL_update)
        return augmentations


class GalaxyDataModule(LossylessDataModule):
    @property
    def Dataset(self):
        return GalaxyDataset

    def get_train_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, curr_split="train", download=False, **dataset_kwargs,
        )

    def get_val_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, curr_split="valid", download=False, **dataset_kwargs,
        )

    def get_test_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, curr_split="test", download=False, **dataset_kwargs,
        )

    def prepare_data(self):
        for split in ["train", "valid", "test"]:
            self.Dataset(
                self.data_dir, curr_split=split, download=True, **self.dataset_kwargs,
            )

    @property
    def mode(self):
        return "image"
