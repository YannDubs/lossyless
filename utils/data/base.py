import abc
from pathlib import Path

import numpy as np

import torch
from lossyless.helpers import to_numpy
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

DIR = Path(__file__).parents[2].joinpath("data")


__all__ = ["LossylessDataset", "LossylessDataModule"]


### Base Dataset ###
class LossylessDataset(abc.ABC):
    """Base class for lossy compression but lossless predicitons.

    Parameters
    -----------
    additional_target : {"input", "representative", "equiv_x", "target", None}, optional
        Additional target to append to the target. `"input"` is the input example (i.e. augmented),
        `"representative"` is a representative of the equivalence class (always the same).
        `"equiv_x"` is some random equivalent x. "target" uses agin the target (i.e. duplicate).

    equivalence : str or set of str, optional
        Equivalence relationship with respect to which to be invariant. Depends on the dataset.
        `None` means no equivalence.

    is_normalize : bool, optional
        Whether to normalize the data.

    seed : int, optional
        Pseudo random seed.
    """

    def __init__(
        self,
        *args,
        additional_target=None,
        equivalence=None,
        is_normalize=False,
        seed=123,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.additional_target = additional_target
        self.equivalence = equivalence
        self.seed = seed
        self.is_normalize = is_normalize

    @abc.abstractmethod
    def get_x_target_Mx(self, index):
        """Return the correct example, target, and maximal invariant."""
        ...

    @abc.abstractmethod
    def get_representative(self, Mx):
        """Return a representative element for current Mx."""
        ...

    @abc.abstractmethod
    def get_equiv_x(self, x, Mx):
        """Return some other random element from same equivalence class."""
        ...

    @property
    @abc.abstractmethod
    def is_clfs(self):
        """Return a dictionary saying whether `input`, `target`, should be classified."""
        ...

    @property
    @abc.abstractmethod
    def shapes(self):
        """Return dictionary giving the shape `input`, `target`."""
        ...

    def __getitem__(self, index):
        x, target, Mx = self.get_x_target_Mx(index)

        if self.additional_target is None:
            targets = target
        else:
            targets = [target]
            targets += self.toadd_target(self.additional_target, x, target, Mx)

        return x, targets

    def toadd_target(self, additional_target, x, target, Mx):

        if additional_target == "input":
            to_add = [x]
        elif additional_target == "representative":
            # representative element from same equivalence class
            to_add = [self.get_representative(Mx)]
        elif additional_target == "equiv_x":
            # other element from same equivalence class
            to_add = [self.get_equiv_x(x, Mx)]
        elif additional_target == "target":
            # duplicate but makes code simpler
            to_add = [target]
        else:
            raise ValueError(f"Unkown additional_target={additional_target}")

        return to_add

    def get_is_clf(self):
        """Return `is_clf` for the target and aux_target."""
        is_clf = self.is_clfs
        is_clf["representative"] = is_clf["input"]
        is_clf["equiv_x"] = is_clf["input"]
        is_clf[None] = None

        return is_clf["target"], is_clf[self.additional_target]

    def get_shapes(self):
        """Return `shapes` for the target and aux_target."""
        shapes = self.shapes
        shapes["representative"] = shapes["input"]
        shapes["equiv_x"] = shapes["input"]
        shapes[None] = None

        return shapes["target"], shapes[self.additional_target]


### Base Datamodule ###

# cannot use abc because inheriting from lightning :( )
class LossylessDataModule(LightningDataModule):
    """Base class for data module for lossy compression but lossless predicitons.

    Notes
    -----
    - similar to pl_bolts.datamodule.CIFAR10DataModule but more easily modifiable.

    Parameters
    -----------
    data_dir : str, optional
        Directory for saving/loading the dataset.

    val_size : int or float, optional
        How many examples to use for validation. This will generate new examples if possible, or
        split from the training set. If float this is in ratio of training size, eg 0.1 is 10%.

    test_size : int, optional
        How many examples to use for test. `None` means all.

    num_workers : int, optional
        How many workers to use for loading data

    batch_size : int, optional
        Number of example per batch for training.

    val_batch_size : int or None, optional
        Number of example per batch during eval and test. If None uses `batch_size`.

    seed : int, optional
        Pseudo random seed.

    reload_dataloaders_every_epoch : bool, optional
        Whether to reload (all) dataloaders at each epoch.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        data_dir=DIR,
        val_size=0.1,
        test_size=None,
        num_workers=16,
        batch_size=128,
        val_batch_size=None,
        seed=123,
        reload_dataloaders_every_epoch=False,
        dataset_kwargs={},
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.reload_dataloaders_every_epoch = reload_dataloaders_every_epoch

    @property
    def Dataset(self):
        """Return the correct dataset."""
        raise NotImplementedError()

    def get_train_dataset(self, **dataset_kwargs):
        """Return the training dataset."""
        raise NotImplementedError()

    def get_val_dataset(self, **dataset_kwargs):
        """Return the validation dataset."""
        raise NotImplementedError()

    def get_test_dataset(self, **dataset_kwargs):
        """Return the test dataset."""
        raise NotImplementedError()

    def prepare_data(self):
        """Dowload and save data on file if needed."""
        raise NotImplementedError()

    @property
    def mode(self):
        """Says what is the mode/type of data. E.g. images, distributions, ...."""
        raise NotImplementedError()

    @property
    def dataset(self):
        """Return the underlying (train) datset ...."""
        dataset = self.train_dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    def set_info_(self):
        """Sets some information from the dataset."""
        dataset = self.dataset
        self.target_is_clf, self.aux_is_clf = dataset.get_is_clf()
        self.target_shape, self.aux_shape = dataset.get_shapes()
        self.shape = dataset.shapes["input"]
        self.additional_target = dataset.additional_target

    @property
    def balancing_weights(self):
        """Dictionary mapping every target to a weight that examples from this class should carry."""
        return dict()

    def setup(self, stage=None):
        """Prepare the datasets for the current stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_train_dataset(**self.dataset_kwargs)
            self.set_info_()
            self.val_dataset = self.get_val_dataset(**self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = self.get_test_dataset(**self.dataset_kwargs)

    def train_dataloader(self, batch_size=None, train_dataset=None, **kwargs):
        """Return the training dataloader while possibly modifying dataset kwargs."""
        dkwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_epoch or len(dkwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **dkwargs)
            train_dataset = self.get_train_dataset(**curr_kwargs)

        if train_dataset is None:
            train_dataset = self.train_dataset

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def val_dataloader(self, batch_size=None, **kwargs):
        """Return the validation dataloader while possibly modifying dataset kwargs."""
        dkwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_epoch or len(dkwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **dkwargs)
            self.val_dataset = self.get_val_dataset(**curr_kwargs)

        if batch_size is None:
            batch_size = self.val_batch_size

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def test_dataloader(self, batch_size=None, **kwargs):
        """Return the test dataloader while possibly modifying dataset kwargs."""
        dkwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_epoch or len(dkwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **dkwargs)
            self.test_dataset = self.get_test_dataset(**curr_kwargs)

        if batch_size is None:
            batch_size = self.val_batch_size

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def eval_dataloader(self, is_eval_on_test, **kwargs):
        """Return the evaluation dataloader (test or val)."""
        if is_eval_on_test:
            return self.test_dataloader(**kwargs)
        else:
            return self.val_dataloader(**kwargs)
