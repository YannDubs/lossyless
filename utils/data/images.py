import abc
import glob
import logging
import math
import os
import subprocess
import zipfile
from os import path
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from lossyless.helpers import BASE_LOG, Normalizer, check_import
from torch.utils.data import random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageNet
from torchvision.transforms import (
    ColorJitter,
    RandomAffine,
    RandomErasing,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    RandomRotation,
)
from utils.estimators import discrete_entropy

from .augmentations import (
    CIFAR10Policy,
    ImageNetPolicy,
    SVHNPolicy,
    get_finetune_augmentations,
    get_simclr_augmentations,
)
from .base import LossylessCLFDataset, LossylessDataModule
from .helpers import int_or_ratio

try:
    import cv2  # only used for galaxy so skip if not needed
except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "Cifar10DataModule",
    "MnistDataModule",
    "FashionMnistDataModule",
    "GalaxyDataModule",
    "ImagenetDataModule",
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
        super().__init__(*args, is_normalize=is_normalize, **kwargs)
        self.equivalence = equivalence
        self.is_augment_val = is_augment_val

        self.base_tranform = self.get_base_transform()
        self.PIL_augment, self.tensor_augment = self.get_curr_augmentations()

    @property
    @abc.abstractmethod
    def is_train(self):
        """Whether considering training split."""
        ...

    @abc.abstractmethod
    def get_img_target(self, index):
        """Return the unaugmented image (in PIL format) and target."""
        ...

    @property
    @abc.abstractmethod
    def dataset_name(self):
        """Name of the dataset."""
        ...

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
        Return a dictortionary of dictionaries containing all possible augmentations of interest.
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
                "y_translation++": RandomAffine(0, translate=(0, 0.25)),
                "x_translation++": RandomAffine(0, translate=(0.25, 0)),
                "shear++": RandomAffine(0, shear=25),
                "scale++": RandomAffine(0, scale=(0.6, 1.4)),
                "color": ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                "hflip": RandomHorizontalFlip(),
                "resizecrop": RandomResizedCrop(size=(shape[1], shape[2])),
                "auto_cifar10": CIFAR10Policy(),
                "auto_imagenet": ImageNetPolicy(),
                "auto_svhn": SVHNPolicy(),
                # NB you should use those 3 also at eval time
                "simclr_cifar10": get_simclr_augmentations(
                    "cifar10", shape[-1], self.is_train
                ),
                "simclr_imagenet": get_simclr_augmentations(
                    "imagenet", shape[-1], self.is_train
                ),
                "simclr_finetune": get_finetune_augmentations(shape[-1], self.is_train),
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
        trnsfs = [
            transform_lib.Resize((shape[1], shape[2])),
            transform_lib.ToTensor(),
        ]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            # raise if can't normalize because you specifrically gave `is_normalize`
            trnsfs += [Normalizer(self.dataset_name, is_raise=True)]

        return transform_lib.Compose(trnsfs)

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

        return transform_lib.Compose(PIL_augment), transform_lib.Compose(tensor_augment)

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


### Torchvision Models ###
# Base class for data module for torchvision models.
class TorchvisionDataModule(LossylessDataModule):
    def get_train_val_dataset(self, **dataset_kwargs):
        dataset = self.Dataset(
            self.data_dir, train=True, download=False, **dataset_kwargs,
        )

        n_val = int_or_ratio(self.val_size, len(dataset))
        train, valid = random_split(
            dataset,
            [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )
        valid.train = False

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

    @property
    def is_train(self):
        return self.train

    @property
    def dataset_name(self):
        return "MNIST"


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

    @property
    def is_train(self):
        return self.train

    @property
    def dataset_name(self):
        return "FashionMNIST"

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

    @property
    def is_train(self):
        return self.train

    @property
    def dataset_name(self):
        return "CIFAR10"


class Cifar10DataModule(TorchvisionDataModule):
    @property
    def Dataset(self):
        return Cifar10Dataset


# Imagenet #
class ImageNetDataset(LossylessImgDataset, ImageNet):
    def __init__(self, root, *args, **kwargs):
        data_dir = path.join(root, "imagenet")
        super().__init__(data_dir, *args, **kwargs)

    @property
    def shapes_x_t_Mx(self):
        shapes = super(ImageNetDataset, self).shapes_x_t_Mx
        shapes["input"] = (3, 224, 224)
        shapes["target"] = (1000,)
        return shapes

    def get_img_target(self, index):
        img, target = ImageNet.__getitem__(self, index)
        return img, target

    def get_base_transform(self):
        # resizing is not just directly resizing but center cropping
        trnsfs = [
            transform_lib.Resize(256),
            transform_lib.CenterCrop(224),
            transform_lib.ToTensor(),
        ]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            # raise if can't normalize because you specifrically gave `is_normalize`
            trnsfs += [Normalizer(self.dataset_name, is_raise=True)]

        return transform_lib.Compose(trnsfs)

    @property
    def is_train(self):
        return self.split == "train"

    @property
    def dataset_name(self):
        return "ImageNet"

    def __len__(self):
        return ImageNet.__len__(self)


class ImagenetDataModule(LossylessDataModule):
    def get_train_dataset(self, **dataset_kwargs):
        train = self.Dataset(self.data_dir, split="train", **dataset_kwargs,)
        return train

    def get_val_dataset(self, **dataset_kwargs):
        valid = self.Dataset(self.data_dir, split="val", **dataset_kwargs,)
        return valid

    def get_test_dataset(self, **dataset_kwargs):
        # we do not have access to test set so use valid
        test = self.Dataset(self.data_dir, split="val", **dataset_kwargs,)
        return test

    def prepare_data(self):
        # ensure that model exists
        self.Dataset(self.data_dir, split="train", **self.dataset_kwargs)
        self.Dataset(self.data_dir, split="val", **self.dataset_kwargs)

    @property
    def mode(self):
        return "image"

    @property
    def Dataset(self):
        return ImageNetDataset


### Non Torchvision Models ###

# Galaxy Zoo #
class GalaxyDataset(LossylessImgDataset):
    def __init__(
        self,
        root: str,
        *args,
        split: str = "train",
        download: bool = True,
        resolution: int = 64,
        **kwargs,
    ):
        # TODO check if need to normalize (if not add is_normalize=False in cfg) because very low std
        check_import("cv2", "GalaxyDataset")

        self.root = root
        self.split = split
        data_dir = path.join(root, "galaxyzoo")
        self.resolution = resolution
        if download and not self.is_exist_data(data_dir):
            self.download(data_dir)
            self.preprocess(data_dir)

        self.load_data(data_dir, split, resolution)

        super().__init__(*args, **kwargs)

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
    def augmentations(self):
        return dict(
            PIL={
                "rotation": RandomRotation(360),
                # in kaggle authors translate 69x69 images by /pm 4 pixel = 11.6%
                "y_translation": RandomAffine(0, translate=(0, 0.116)),
                "x_translation": RandomAffine(0, translate=(0.116, 0)),
                "scale": RandomAffine(0, scale=(1.0/1.3, 1.3)),
                "hflip": RandomHorizontalFlip(0.5),
                "vflip": RandomVerticalFlip(0.5),
                # color transforms where not used originally but could be a good idea
                #"color": ColorJitter(
                #    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                #),
            },
            # tensor={"erasing": RandomErasing(value=0.5),},
        )

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
    def is_train(self):
        return self.split == "train"

    @property
    def dataset_name(self):
        return f"galaxy{self.resolution}"


class GalaxyDataModule(LossylessDataModule):
    @property
    def Dataset(self):
        return GalaxyDataset

    def get_train_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, split="train", download=False, **dataset_kwargs,
        )

    def get_val_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, split="valid", download=False, **dataset_kwargs,
        )

    def get_test_dataset(self, **dataset_kwargs):
        return self.Dataset(
            self.data_dir, split="test", download=False, **dataset_kwargs,
        )

    def prepare_data(self):
        for split in ["train", "valid", "test"]:
            self.Dataset(
                self.data_dir, split=split, download=True, **self.dataset_kwargs,
            )

    @property
    def mode(self):
        return "image"
