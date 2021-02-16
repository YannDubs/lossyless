import collections
import copy
import glob
import logging
import os
import types
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import pytorch_lightning as pl
import torch
from lossyless.callbacks import save_img
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def omegaconf2namespace(cfg, is_allow_missing=False):
    """Converts omegaconf to namesapce so that can use primitive types."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # primitive types
    return dict2namespace(cfg, is_allow_missing=is_allow_missing)


def dict2namespace(d, is_allow_missing=False, all_keys=""):
    """Converts recursively dictionary to namespace."""
    namespace = NamespaceMap(d)
    for k, v in d.items():
        if v == "???" and not is_allow_missing:
            raise ValueError(f"Missing value for {all_keys}.{k}.")
        elif isinstance(v, dict):
            namespace[k] = dict2namespace(v, f"{all_keys}.{k}")
    return namespace


class NamespaceMap(Namespace, collections.abc.MutableMapping):
    """Namespace that can act like a dict."""

    def __init__(self, d):
        # has to take a single argument as input instead of a dictionnary as namespace usually do
        # because from pytorch_lightning.utilities.apply_func import apply_to_collection doesn't work
        # with namespace (even though they think it does)
        super().__init__(**d)

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


def cont_tuple_to_tuple_cont(container):
    """Converts a container (list, tuple, dict) of tuple to a tuple of container."""
    if isinstance(container, dict):
        return tuple(dict(zip(container, val)) for val in zip(*container.values()))
    elif isinstance(container, list) or isinstance(container, tuple):
        return tuple(zip(*container))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(container)))


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


def at_least_ndim(arr, ndim, is_prefix_padded=False):
    """Ensures that a numpy or torch array is at least `ndim`-dimensional."""
    padding = (1,) * (ndim - len(arr.shape))
    if is_prefix_padded:
        padded_shape = padding + arr.shape
    else:
        padded_shape = arr.shape + padding
    return arr.reshape(padded_shape)


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


def learning_rate_finder(
    module, datamodule, trainer, min_max_lr=[1e-7, 10], is_argmin=False
):
    """
    Automatically select the new learning rate and plot the learing rate finder in the `min_max_lr`
    bounds. If `is_argmin` choses the lr that ields the argmin on the plot (usually larger lr), if 
    not selects the one with the most negative derivative (usually smaller lr). 
    """
    # ensure not inplace
    trainer = copy.deepcopy(trainer)

    # ans cannot be pickled => don't deepcopy it (it's not going to change because not trained)
    old_module = module
    featurizer, module.featurizer = module.featurizer, None
    module = copy.deepcopy(old_module)
    module.featurizer = featurizer  # bypass deepcopy
    old_module.featurizer = featurizer  # put back

    min_lr, max_lr = min_max_lr
    module.hparams.optimizer_pred.kwargs.lr = min_lr  #! shouldn't be needed
    lr_finder = trainer.tuner.lr_find(
        module, datamodule=datamodule, min_lr=min_lr, max_lr=max_lr
    )

    if is_argmin:
        lr_finder.suggestion = types.MethodType(suggest_min_lr, lr_finder)

    fig = lr_finder.plot(suggest=True)
    if module.hparams.logger.is_can_plot_img:
        save_img(module, trainer, fig, "Learning Rate Finder", "")

    new_lr = lr_finder.suggestion()
    log_dict(trainer, dict(suggested_lr=new_lr), True)

    return new_lr


def suggest_min_lr(self, skip_begin: int = 10, skip_end: int = 1):
    """ This will propose a suggestion for choice of initial learning rate
    as the point with smallest lr.
    """
    try:
        loss = np.array(self.results["loss"][skip_begin:-skip_end])
        loss = loss[np.isfinite(loss)]
        min_grad = loss.argmin()
        self._optimal_idx = min_grad + skip_begin
        return self.results["lr"][self._optimal_idx]
    except Exception:
        logger.exception(
            "Failed to compute suggesting for `lr`. There might not be enough points."
        )
        self._optimal_idx = None


def apply_featurizer(datamodule, featurizer):
    """Apply a featurizer on every example of a datamodule and return a new datamodule."""
    # TODO
