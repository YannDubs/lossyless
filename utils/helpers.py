import collections
import copy
import glob
import logging
import os
import shutil
import types
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import pl_bolts
import pytorch_lightning as pl
import pytorch_lightning.plugins.training_type as train_plugins
import torch
from lossyless.callbacks import save_img
from omegaconf import OmegaConf
from pytorch_lightning.overrides.data_parallel import LightningParallelModule
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


def format_resolver(x, pattern):
    return f"{x:{pattern}}"


def cfg_save(cfg, filename):
    """Save a config as a yaml file."""
    if isinstance(cfg, NamespaceMap):
        cfg = OmegaConf.create(namespace2dict(cfg))
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif OmegaConf.is_config(cfg):
        pass
    else:
        raise ValueError(f"Unkown type(cfg)={type(cfg)}.")
    return OmegaConf.save(cfg, filename)


def cfg_load(filename):
    """Load a config yaml file."""
    return omegaconf2namespace(OmegaConf.load(filename))


def omegaconf2namespace(cfg, is_allow_missing=False):
    """Converts omegaconf to namesapce so that can use primitive types."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # primitive types
    return dict2namespace(cfg, is_allow_missing=is_allow_missing)


def dict2namespace(d, is_allow_missing=False, all_keys=""):
    """
    Converts recursively dictionary to namespace. Does not work if there is a dict whose
    parent is not a dict.
    """
    namespace = NamespaceMap(d)

    for k, v in d.items():
        if v == "???" and not is_allow_missing:
            raise ValueError(f"Missing value for {all_keys}.{k}.")
        elif isinstance(v, dict):
            namespace[k] = dict2namespace(v, f"{all_keys}.{k}")
    return namespace


def namespace2dict(namespace):
    """
    Converts recursively namespace to dictionary. Does not work if there is a namespace whose
    parent is not a namespace.
    """
    d = dict(**namespace)
    for k, v in d.items():
        if isinstance(v, NamespaceMap):
            d[k] = namespace2dict(v)
    return d


class NamespaceMap(Namespace, collections.abc.MutableMapping):
    """Namespace that can act like a dict."""

    def __init__(self, d):
        # has to take a single argument as input instead of a dictionnary as namespace usually do
        # because from pytorch_lightning.utilities.apply_func import apply_to_collection doesn't work
        # with namespace (even though they think it does)
        super().__init__(**d)

    def select(self, k):
        """Allows selection using `.` in string."""
        to_return = self
        for subk in k.split("."):
            to_return = to_return[subk]
        return to_return

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)


def set_debug(cfg):
    """Enter debug mode."""
    logger.info(OmegaConf.to_yaml(cfg))
    torch.autograd.set_detect_anomaly(True)
    os.environ["HYDRA_FULL_ERROR"] = "1"


def get_latest_match(pattern):
    """
    Return the file that matches the pattern which was modified the latest.
    """
    all_matches = (Path(p) for p in glob.glob(str(pattern), recursive=True))
    latest_match = max(all_matches, key=lambda x: x.stat().st_mtime)
    return latest_match


def update_prepending(to_update, new):
    """Update a dictionary with another. the difference with .update, is that it puts the new keys
    before the old ones (prepending)."""
    # makes sure don't update arguments
    to_update = to_update.copy()
    new = new.copy()

    # updated with the new values appended
    to_update.update(new)

    # remove all the new values => just updated old values
    to_update = {k: v for k, v in to_update.items() if k not in new}

    # keep only values that ought to be prepended
    new = {k: v for k, v in new.items() if k not in to_update}

    # update the new dict with old one => new values are at the begining (prepended)
    new.update(to_update)

    return new


class StrFormatter:
    """String formatter that acts like some default dictionnary `"formatted" == StrFormatter()["toformat"]`.

    Parameters
    ----------
    exact_match : dict, optional
        Dictionary of strings that will be replaced by exact match.

    subtring_replace : dict, optional
        Dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point.

    to_upper : list, optional
        Words that should be upper cased.
    """

    def __init__(self, exact_match={}, subtring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.subtring_replace = subtring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.subtring_replace.items():
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key

    def __call__(self, x):
        return self[x]

    def update(self, new_dict):
        """Update the substring replacer dictionary with a new one (missing keys will be prepended)."""
        self.subtring_replace = update_prepending(self.subtring_replace, new_dict)


def getattr_from_oneof(list_of_obj, name):
    """
    Equivalent to `getattr` but on a list of objects and will return the attribute from the first 
    object that has it.
    """
    if len(list_of_obj) == 0:
        # base case
        raise AttributeError(f"{name} was not found.")

    obj = list_of_obj[0]

    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            return getattr_from_oneof(list_of_obj[1:], name)
        except AttributeError:
            pass

    raise AttributeError(f"{name} was not found in {list_of_obj}.")


def replace_keys(d, old, new):
    """replace keys in a dict."""
    return {k.replace(old, new): v for k, v in d.items()}


# credits : https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def log_dict(trainer, to_log, is_param):
    """Safe logging of param or metrics."""
    try:
        if is_param:
            trainer.logger.log_hyperparams(to_log)
        else:
            trainer.logger.log_metrics(to_log)
    except:
        pass


class SklearnDataModule(pl_bolts.datamodules.SklearnDataModule):
    # so that same as LossylessDataModule
    def eval_dataloader(self, is_eval_on_test, **kwargs):
        """Return the evaluation dataloader (test or val)."""
        if is_eval_on_test:
            return self.test_dataloader(**kwargs)
        else:
            return self.val_dataloader(**kwargs)


def apply_featurizer(datamodule, featurizer, is_eval_on_test=True, **kwargs):
    """Apply a featurizer on every example (precomputed) of a datamodule and return a new datamodule."""
    train_dataset = datamodule.train_dataset
    # ensure that you will not be augmenting
    if isinstance(train_dataset, Subset):
        train_dataset.dataset.curr_split = "validation"
    else:
        train_dataset.curr_split = "validation"

    out_train = featurizer.predict(
        dataloaders=[datamodule.train_dataloader(train_dataset=train_dataset)]
    )
    out_val = featurizer.predict(dataloaders=[datamodule.val_dataloader()])
    out_test = featurizer.predict(
        dataloaders=[datamodule.eval_dataloader(is_eval_on_test)]
    )

    X_train, Y_train = zip(*out_train)
    X_val, Y_val = zip(*out_val)
    X_test, Y_test = zip(*out_test)

    # only select kwargs that can be given to sklearn
    sklearn_kwargs = dict()
    sklearn_kwargs["batch_size"] = kwargs.get("batch_size", 128)
    sklearn_kwargs["num_workers"] = kwargs.get("num_workers", 4)

    # make a datamodule from features that are precomputed
    datamodule = SklearnDataModule(
        np.concatenate(X_train, axis=0),
        np.concatenate(Y_train, axis=0),
        x_val=np.concatenate(X_val, axis=0),
        y_val=np.concatenate(Y_val, axis=0),
        x_test=np.concatenate(X_test, axis=0),
        y_test=np.concatenate(Y_test, axis=0),
        shuffle=True,
        pin_memory=True,
        **sklearn_kwargs,
    )

    return datamodule


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_load_checkpoint(self, checkpointed_state):
        super().on_load_checkpoint(checkpointed_state)

        # trick to keep only one model because pytorch lighnign by default doesn't save
        # best k_models, so when preempting they stask up. Open issue. THis is only correct for k=1
        self.best_k_models = {}
        self.best_k_models[self.best_model_path] = self.best_model_score
        self.kth_best_model_path = self.best_model_path


def remove_rf(path, not_exist_ok=False):
    """Remove a file or a folder"""
    path = Path(path)

    if not path.exists() and not_exist_ok:
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir:
        shutil.rmtree(path)
