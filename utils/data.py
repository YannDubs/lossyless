from numpy.core.fromnumeric import shape
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    # MNISTDataModule, # currently not working #337
    # FashionMNISTDataModule, #337
)

# tmp #337
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pytorch_lightning import LightningDataModule
from torch.nn.functional import normalize

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms.functional import rotate
from unittest.mock import patch
import numpy as np
import os
import torch
import math
import random
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib

from lossyless.helpers import to_numpy, concatenate, tmp_seed, get_normalization

DATASETS_DICT = {
    "cifar10": "CIFAR10Module",
    "toymnist": "ToyMNISTModule",
    "toyfashionmnist": "ToyFashionMNISTModule",
}
DATASETS = list(DATASETS_DICT.keys())
DIR = os.path.abspath(os.path.dirname(__file__))


__all__ = ["get_datamodule"]


### HELPERS ###


def get_datamodule(datamodule):
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[datamodule])
    except KeyError:
        raise ValueError("Unkown datamodule: {}".format(datamodule))


def get_augmentations(augmentations=[], params={}):
    """Helper function that returns all image augmentations."""
    pil_augmentations = []
    tensor_augmentations = []

    if "erase" in augmentations:
        tensor_augmentations += [transform_lib.RandomErasing(params["erase"])]

    return pil_augmentations, tensor_augmentations


### BASE DATASET ###


class LossylessDatasetToyImg:
    """Base class for toy image datasets used for lossy compression but lossless predicitons.

    Notes
    -----
    - the dataset will be augmented A PRIORI with the augmentations, so it can use a lot of memory.
    - for computations of entropy, we assume that there is a unique way of constructing the final
    example. I.e. unique base example and unique data augmentation. This for example is not exactly
    the case in MNIST (where 6 can become 9's and rotations of 0 can give back the same 0) but is often
    a good enough simplifying approximation.
    - this is written with MNIST / fashion mnist in mind, but is modular enough to work with any 
    dataset by overiding a few methods. For example you would have to define a new property `data`
    and `targets` is that is not the name of the iamges and labels.

    Parameters
    -----------
    additional_target : {"aug_img", "img", "new_img", "aug_idx", "idx", "target", None}, optional
        Additional target to append to the target. `"aug_img"` is the input image (i.e. augmented),
        `"img"` is the base image (orbit representative). `"new_img"` is another random image on the same
        orbit. `"aug_idx"` is the actual index. `"idx"` is the base index (maximal invariant). "target"
        uses agin the target (i.e. duplicate).

    n_inputs : int, optional
        Number of images (from same orbit) for each example in a batch.
    
    n_per_target : int, optional 
        Number of examples to keep per label.

    targets_drop : list, optional
        Targets to drop (needs to be same type as the targets).

    n_rotations : int, optional
        Order of the discrete rotation group (i.e. cyclic group by multiples of 360/n_rotation).
        It n_rotations != 2 or 4, then it is best to use images with a black background to make sure 
        that the transformation is indeed a group (because the padding will be black).

    n_luminosity : int, optional
        Order fo the discrete group of luminosity changes (i.e. cyclic group which adds 
        1/n_luminosity modulo 1).  
    
    seed : int, optional
        Pseudo random seed.
    """

    def __init__(
        self,
        *args,
        additional_target=None,
        n_inputs=1,
        n_per_target=None,
        targets_drop=[],
        n_rotations=1,
        n_luminosity=1,
        seed=123,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.additional_target = additional_target
        self.n_inputs = n_inputs
        self.n_per_target = n_per_target
        self.targets_drop = targets_drop
        self.n_rotations = n_rotations
        self.n_luminosity = n_luminosity
        self.seed = seed

        if self.n_per_target is not None:
            self.keep_n_idcs_per_target_(n_per_target)

        self.drop_targets_(self.targets_drop)

        self.noaug_length = len(self)
        with tmp_seed(self.seed):
            self.augment_rotations_()  # dataset is increased to contain rotations
            self.augment_luminosity_()  # dataset is increased to contain luminosity
        self.aug_factor = len(self) // self.noaug_length

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        targets = [target]

        notaugmented_idx = index % self.noaug_length
        if self.additional_target is None:
            targets += [None]  # just so that all the code is the same
        elif self.additional_target == "aug_img":
            targets += [img]
        elif self.additional_target == "img":
            notaug_img, _ = super().__getitem__(notaugmented_idx)
            targets += [notaug_img]
        elif self.additional_target == "new_img":
            k_jump_orbit = random.randint(1, self.aug_factor - 1)
            sampled_idx = index + self.noaug_length * k_jump_orbit
            new_img, _ = super().__getitem__(sampled_idx)
            targets += [new_img]
        elif self.additional_target == "aug_idx":
            targets += [index]
        elif self.additional_target == "idx":
            targets += [notaugmented_idx]
        elif self.additional_target == "target":
            # duplicate but makes coder simpler
            targets += [target]
        else:
            raise ValueError(f"Unkown self.additional_target={self.additional_target}")

        # if self.n_inputs > 1:
        #     add_imgs = []
        #     for _ in range(self.n_inputs):
        #         k_jump_orbit = random.randint(1, self.aug_factor - 1)
        #         sampled_idx = index + self.noaug_length * k_jump_orbit
        #         other_img, _ = super().__getitem__(sampled_idx)
        #         add_imgs += other_img
        #     img = [img] + add_imgs

        return img, targets

    @property
    def entropies(self):
        return {
            "X": math.log(len(self), base=2),
            "G_rot": math.log(self.n_rotations, base=2),
            "G_lum": math.log(self.n_luminosity, base=2),
            "C": math.log(len(self.noaug_length), base=2),
        }

    def augment_rotations_(self):
        """Augment the dataset with rotations."""
        self.augment_(rotate, 360, self.n_rotations)

    def augment_luminosity_(self):
        """Augment the dataset with luminosity."""
        # byte tensor takes care of modulo arithmetic
        self.augment_(lambda x, a: x + int(a), 256, self.n_luminosity)

        #! has to overide in case you are not working with uint8, e.g, `(x + int(a)) % 256`
        if not (
            isinstance(self.data, torch.Tensor) and (self.data.dtype == torch.uint8)
        ):
            raise NotImplementedError("Luminosity only implemented for torch.uint8.")

    def augment_(self, f_augment, max_n, n):
        """Augment the dataset with a cyclic finite group of order n and group action f_augment."""
        if n == 1:
            return

        data_list = [f_augment(self.data, i * max_n / n) for i in range(n)]
        target_list = [self.targets for _ in range(n)]
        self.data = concatenate(data_list)
        self.targets = concatenate(target_list)

    def keep_n_idcs_per_target_(self, n):
        """Keep only `n` example for each target inplace.

        Parameters
        ----------
        n : int
            Number of examples to keep per label.
        """
        np_trgts = to_numpy(self.targets)
        list_idcs_keep = [np.nonzero(np_trgts == i)[0][:n] for i in np.unique(np_trgts)]
        idcs_keep = np.sort(np.concatenate(list_idcs_keep))
        self.keep_indcs_(list(idcs_keep))

    def drop_targets_(self, targets):
        """Drops targets inplace.

        Parameters
        ----------
        targets : list
            Targets to drop (needs to be same type as the targets).
        """
        for target in targets:
            np_trgts = to_numpy(self.targets)
            idcs_keep = np.nonzero(np_trgts != target)[0]
            self.keep_indcs_(list(idcs_keep))

    def keep_indcs_(self, indcs):
        """Keep the given indices inplace.

        Parameters
        ----------
        indcs : array-like int
            Indices to keep. If multiplicity larger than 1 then will duplicate  the data.
        """
        self.data = self.data[indcs]
        self.targets = self.targets[indcs]


# def __getitem__(self, index):

#         # generate unaugmented image
#         with patch.object(self, "transform", None):
#             img, target = super().__getitem__(index)

#         if self.test_transform is not None:
#             test_img = self.test_transform(img).clone()
#         else:
#             test_img = img.clone()

#         if self.transform is not None:
#             img = self.transform(img).clone()

#         return img, (target, index, test_img)


class LossylessDataModule(LightningDataModule):
    """Base class for data module for lossy compression but lossless predicitons.

    Notes
    -----
    - very similar to pl_bolts.datamodule.CIFAR10DataModule but more easily modifiable.

    Parameters
    -----------
    data_dir : str, optional
        Directory for saving/loading the dataset.

    val_split : int, optional
        How many of the training images to use for the validation split.
    
    num_workers : int, optional 
        How many workers to use for loading data

    batch_size : list, optional
        Number of example per batch.  
    
    seed : int, optional
        Pseudo random seed.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.

    is_val_on_test : bool, optional
        Whether to validate on test. This is useful for `LossylessDatasetToyImg` when 
        the training set should not be modified (i.e. shouldn't sample valid from it)
        and when the test / validation set is not very important.
    """

    _DATASET = None

    def __init__(
        self,
        *args,
        data_dir=DIR,
        val_split=0,
        num_workers=16,
        batch_size=128,
        seed=123,
        dataset_kwargs={},
        is_val_on_test=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.is_val_on_test = is_val_on_test

        self.dataset_kwargs = dataset_kwargs
        self.Dataset = self._DATASET
        self.dim = self.Dataset.shape

    @property
    def shape(self):
        return self.dim

    @property
    def num_classes(self):
        return self.Dataset.num_classes

    def prepare_data(self):
        """Dowload and save onfile."""
        self.Dataset(self.data_dir, train=True, download=True, **self.dataset_kwargs)
        self.Dataset(self.data_dir, train=False, download=True, **self.dataset_kwargs)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            # train
            trnsf_train = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            dataset = self.Dataset(
                self.data_dir,
                transform=trnsf_train,
                train=True,
                download=False,
                **self.dataset_kwargs,
            )
            self.noaug_length = dataset.noaug_length  # train and valid

            self.dataset_train, _ = random_split(
                dataset,
                [len(dataset) - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # validation
            dataset = self.Dataset(
                self.data_dir,
                transform=self.default_transforms(),
                train=True,
                download=False,
                **self.dataset_kwargs,
            )
            _, self.dataset_val = random_split(
                dataset,
                [len(dataset) - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.dataset_test = self.Dataset(
                self.data_dir,
                train=False,
                transform=self.default_transforms(),
                download=False,
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_test if self.is_val_on_test else self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def default_transforms(self):
        trnsfs = [
            transform_lib.Resize((self.shape[1], self.shape[2])),
            transform_lib.ToTensor(),
        ]

        if self.shape[0] == 3:
            # only normallize colored images
            trnsfs += [get_normalization(self._DATASET)]

        return transform_lib.Compose(trnsfs)


### Toy MNIST ###


class ToyMNISTDATASET(LossylessDatasetToyImg, MNIST):
    """MNIST dataset for toy experiments with group transforms.
    
    Parameters
    ----------
    *args: 
        Arguments to `torchvision.datasets.MNIST`.

    n_per_target: int, optional
        Number of examples to use per target.

    n_luminosity : int, optional
        Order fo the discrete group of luminosity changes (i.e. cyclic group which adds 
        1/n_luminosity modulo 1).  

    **kwargs:
        Arguments to `torchvision.datasets.MNIST` and `LossylessDatasetToyImg`.
    """

    FOLDER = "MNIST"
    shape = (1, 32, 32)
    num_classes = 10

    def __init__(self, *args, n_per_target=5000, n_luminosity=8, **kwargs):
        super().__init__(
            *args, n_per_target=n_per_target, n_luminosity=n_luminosity, **kwargs
        )

    # avoid duplicates by saving once at "MNIST" rather than at multiple  __class__.__name__
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.FOLDER, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.FOLDER, "processed")


class ToyMNISTModule(LossylessDataModule):
    _DATASET = ToyMNISTDATASET


### Toy FASHION MNIST ###
class ToyFashionMNISTDATASET(LossylessDatasetToyImg, FashionMNIST):
    """MNIST dataset for toy experiments with group transforms.
    
    Parameters
    ----------
    *args: 
        Arguments to `torchvision.datasets.FashionMNIST`.

    n_per_target: int, optional
        Number of examples to use per target.

    n_luminosity : int, optional
        Order fo the discrete group of luminosity changes (i.e. cyclic group which adds 
        1/n_luminosity modulo 1).  

    n_rotations : int, optional
        Order of the discrete rotation group (i.e. cyclic group by multiples of 360/n_rotation).

    **kwargs:
        Arguments to `torchvision.datasets.FashionMNIST` and `LossylessDatasetToyImg`.
    """

    FOLDER = "FashionMNIST"
    shape = (1, 32, 32)
    num_classes = 10

    def __init__(
        self, *args, n_per_target=5000, n_luminosity=8, n_rotations=8, **kwargs
    ):
        super().__init__(
            *args,
            n_per_target=n_per_target,
            n_luminosity=n_luminosity,
            n_rotations=n_rotations,
            **kwargs,
        )

    processed_folder = ToyMNISTDATASET.processed_folder
    raw_folder = ToyMNISTDATASET.raw_folder


class ToyFashionMNISTModule(LossylessDataModule):
    _DATASET = ToyFashionMNISTDATASET


### CIFAR ###
# TODO

# class CIFAR10DATASET(LossylessDataset, CIFAR10):
#     @property
#     def test_transform(self):
#         return CIFAR10Module.default_transforms(self)


# class CIFAR10Module(LossylessDataModule, CIFAR10DataModule):
#     _DATASET = CIFAR10DATASET

#     def train_transform(self):
#         transforms = augmentations[0]
#         transforms += self.default_transforms.transforms
#         transforms += augmentations[1]
#         return transform_lib.Compose(transforms)


### IMAGENET ###
# TODO

### KODAK ###
# TODO
