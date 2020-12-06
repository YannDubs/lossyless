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

import torch
import torch.distributions as dist


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

from utils.visualizations.distributions import setup_grid
from utils.helpers import MinMaxScaler
from lossyless.helpers import (
    to_numpy,
    concatenate,
    tmp_seed,
    get_normalization,
    BASE_LOG,
    atleast_ndim,
)

DATASETS_DICT = {
    "cifar10": "CIFAR10Module",
    "toymnist": "ToyMNISTModule",
    "toyfashionmnist": "ToyFashionMNISTModule",
    "distribution": "DistributionDataModule",
}
DATASETS = list(DATASETS_DICT.keys())
DIR = os.path.abspath(os.path.dirname(__file__))


__all__ = ["get_datamodule", "BananaDistribution"]


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


### BASE IMG DATASET ###


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
    additional_target : {"input", "representative", "other_representative", "idx", "max_inv", "target", None}, optional
        Additional target to append to the target. `"input"` is the input image (i.e. augmented),
        `"representative"` is the base image (orbit representative). `"other_representative"` is 
        another random image on the same orbit. `"idx"` is the actual index. `"max_inv"` is the 
        base index (maximal invariant). "target" uses agin the target (i.e. duplicate).

    additional_target : {"input", "representative", "other_representative", "idx", "max_inv", "target", None}, optional
        Like additional_target. I.e. total number of targets will be 3. This will be used as in 
        input to the coder. 

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

    # should assign in children
    is_classification = None
    shape = None
    target_shape = None

    def __init__(
        self,
        *args,
        additional_target=None,
        additional_target2=None,
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
        self.additional_target2 = additional_target2
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

    def toadd_target(self, additional_target, index, img, target):
        notaugmented_idx = index % self.noaug_length

        if additional_target is None:
            to_add = [[]]  # just so that all the code is the same
        elif additional_target == "input":
            to_add = [img]
        elif additional_target == "representative":
            notaug_img, _ = super().__getitem__(notaugmented_idx)
            to_add = [notaug_img]
        elif additional_target == "other_representative":
            k_jump_orbit = random.randint(1, self.aug_factor - 1)
            sampled_idx = notaugmented_idx + self.noaug_length * k_jump_orbit
            new_img, _ = super().__getitem__(sampled_idx)
            to_add = [new_img]
        elif additional_target == "idx":
            to_add = [index]
        elif additional_target == "max_inv":
            to_add = [notaugmented_idx]
        elif additional_target == "target":
            # duplicate but makes code simpler
            to_add = [target]
        else:
            raise ValueError(f"Unkown additional_target={additional_target}")

        return to_add

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        targets = [target]

        targets += self.toadd_target(self.additional_target, index, img, target)
        targets += self.toadd_target(self.additional_target2, index, img, target)

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
            "X": math.log(len(self), base=BASE_LOG),
            "G_rot": math.log(self.n_rotations, base=BASE_LOG),
            "G_lum": math.log(self.n_luminosity, base=BASE_LOG),
            "M": math.log(len(self.noaug_length), base=BASE_LOG),
        }

    @property
    def aux_infos(self):
        return {
            "idx": (True, len(self)),
            "max_inv": (True, self.noaug_length),
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
        super().__init__(*args, dims=self._DATASET.shape, **kwargs)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.is_val_on_test = is_val_on_test

        self.dataset_kwargs = dataset_kwargs
        self.Dataset = self._DATASET

    @property
    def shape(self):
        return self.dims

    def set_info_(self, dataset):
        """Sets some information from the dataset."""
        self.is_classification = dataset.is_classification
        self.target_shape = dataset.target_shape

        aux_infos = {
            "input": (False, self.shape),
            "representative": (False, self.shape),
            "other_representative": (False, self.shape),
            "target": (self.is_classification, self.target_shape),
        }

        if dataset.additional_target in aux_infos:
            # these should be available in all datasets
            info = aux_infos[dataset.additional_target]
        else:
            # dataset specific
            info = dataset.aux_infos[dataset.additional_target]
        self.is_classification_aux, self.target_aux_shape = info

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
            self.set_info_(dataset)

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
    target_shape = 10
    is_classification = True

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
    is_classification = True
    target_shape = 10

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

### Distribution Dataset ###
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
        self.angle = angle * math.pi / 180

        self.rot_mat = self.get_rot_mat(self.angle)
        self.inv_rot_mat = self.get_rot_mat(-self.angle)

    def get_rot_mat(self, angle):
        angle = torch.tensor([angle])
        cos, sin = torch.cos(angle), torch.sin(angle)
        return torch.tensor([[cos, sin], [-sin, cos]])

    def _call(self, x):
        return x @ self.rot_mat

    def _inverse(self, y):
        return y @ self.inv_rot_mat

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
        
    asymmetry : float, optional
        Controls the asymmetry of the banana-shape.
        
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
        factor=4,
        asymmetry=0.0,
        location=torch.tensor([-3.0, -4]),
        angle=-40,
        scale=0.1,
    ):
        base_dist = dist.MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=torch.tensor(
                [[factor * scale, 0.0], [asymmetry * scale, 1.0 * scale]]
            ),
        )
        transforms = dist.ComposeTransform(
            [
                BananaTransform(curvature / scale, factor=factor * scale),
                RotateTransform(angle),
                dist.AffineTransform(location * scale, torch.ones(2)),
            ]
        )
        super().__init__(base_dist, transforms)

        self.curvature = curvature
        self.factor = factor
        self.rotate = rotate


class LossylessDatasetToyDistribution:
    """Base class for toy 2D distribution datasets used for lossy compression but lossless predicitons.

    Parameters
    -----------
    additional_target : {"max_inv", "input", "target", None}, optional
        Additional target to append to the target. "max_inv" is the maximal invariant. "input"
        is the input (sample). "target" uses agin the target (i.e. duplicate).

    additional_target2 : {"max_inv", "input", "target"}, optional
        Like additional_target. I.e. total number of targets will be 3. This will be used as in 
        input to the coder.

    n_inputs : int, optional
        Number of sample (from same orbit) for each example in a batch. Does not work currently.
    
    length : int, optional 
        Size of the dataset.

    distribution : torch.Distribution, optional
        Main distribution to sample from.

    group : {"rotation","y_translation","x_translation"}, optional
        Group with respect to which to be invariant.

    func_target : callable, optional
        Function that creates the target using the maximal invariant for the group (not that any
        target that is conditionally G-invariant and deterministic has to be a function of the max inv),

    range_lim : int, optional
        Range for plotting and estiomating entropies [-range_lim,range_lim]^2.
    """

    shape = (2,)
    is_classification = False
    target_shape = 1
    max_inv_shape = 1

    def __init__(
        self,
        *args,
        additional_target=None,
        additional_target2=None,
        n_inputs=1,
        length=10000,
        distribution=BananaDistribution(),
        group="rotation",
        func_target=lambda m: m,
        range_lim=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert n_inputs == 1

        self.additional_target = additional_target
        self.additional_target2 = additional_target2
        self.n_inputs = n_inputs
        self.length = length
        self.distribution = distribution
        self.group = group
        self.range_lim = range_lim

        self.data = distribution.sample([length])

        # scaling before  maximal invariant is ok because scaling does not change radius if not is_scale_each_dims
        # if scaler is None:
        #     self.scaler = MinMaxScaler(is_scale_each_dims=True).fit(self.data)
        # else:
        #     self.scaler = scaler
        # self.data = self.scaler.transform(self.data)  # make sure in [0,1]

        # make sure y_dim is a dimension to enable use of mse_loss (as y_pred will have y_dim as dimension)
        self.max_invariants = atleast_ndim(self.get_max_invariants(), 2)
        self.targets = func_target(self.max_invariants)

    def __len__(self):
        return self.length

    def get_max_invariants(self):
        if self.group == "rotation":
            return self.data.norm(2, dim=-1)  # L2 norm
        elif self.group == "y_translation":
            return self.data[:, 0]  # max inv is x coord
        elif self.group == "x_translation":
            return self.data[:, 1]  # max inv is y coord
        else:
            raise ValueError(f"Unkown group={self.group}.")

    def toadd_target(self, additional_target, index, sample, target):
        if additional_target is None:
            to_add = [[]]  # just so that all the code is the same
        elif additional_target == "input":
            to_add = [sample]
        elif additional_target == "max_inv":
            to_add = [self.max_invariants[index]]
        elif additional_target == "target":
            to_add = [target]
        else:
            raise ValueError(f"Unkown additional_target={additional_target}")
        return to_add

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]

        targets = [target]
        targets += self.toadd_target(self.additional_target, index, sample, target)
        targets += self.toadd_target(self.additional_target2, index, sample, target)

        return sample, targets

    @property
    def entropies(self):
        if hasattr(self, "_entropies"):
            return self._entropies

        entropies = dict(X=self.distribution.base_dist.entropy() / math.log(BASE_LOG),)
        n_pts = int(1e4)

        if self.group == "rotation":
            pass  # TODO needs to marginalize theta in polar coordinates to get H[rho]
        elif "translation" in self.group:
            is_x = "x_" in self.group
            range_lim = 15
            _, __, zz = setup_grid(range_lim=range_lim, n_pts=n_pts)
            log_probs = self.distribution.log_prob(zz).view(n_pts, n_pts)
            d = (range_lim * 2) / n_pts
            # p(x) = \int p(x,y) dy
            log_marg_probs = (log_probs + math.log(d)).sum(1 if is_x else 0)
            # normalize so that approximate pmf (can use discrete entropy).
            # log (p(x)/(\sum p(x))) = log p(x) - log_sum_exp p(x)
            log_normalized = log_marg_probs - torch.logsumexp(log_marg_probs, 0)
            # uses histogram estimator https://en.wikipedia.org/wiki/Entropy_estimation#Histogram_estimator
            H_M = -(log_normalized.exp() * (log_normalized - math.log(d))).sum()
            entropies["M"] = H_M / math.log(BASE_LOG)
        else:
            raise ValueError(f"Unkown group={self.group}.")

        self._entropies = entropies
        return entropies

    @property
    def aux_infos(self):
        return {
            "max_inv": (False, 1),
        }


class DistributionDataModule(LossylessDataModule):
    _DATASET = LossylessDatasetToyDistribution

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            dataset = self.Dataset(**self.dataset_kwargs)
            self.set_info_(dataset)

            self.dataset_train, self.dataset_val = random_split(
                dataset,
                [len(dataset) - self.val_split, self.val_split],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.dataset_test = self.Dataset(
                **self.dataset_kwargs
            )  # scaler=self.scaler,

    def default_transforms(self):
        pass  # no transformation

    def prepare_data(self):
        pass

    @property
    def y_dim(self):
        return self.Dataset.y_dim
