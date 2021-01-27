import collections
import logging
import os
from argparse import Namespace
from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def create_folders(base_path, names):
    """Safely creates a list of folders at a certain path."""
    for name in names:
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)


def omegaconf2namespace(cfg):
    """Converts omegaconf to namesapce so that can use primitive types."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # primitive types
    return dict2namespace(cfg)


def dict2namespace(d):
    """Converts recursively dictionary to namespace."""
    namespace = NamespaceMap(**d)
    for k, v in d.items():
        if isinstance(v, dict):
            namespace[k] = dict2namespace(v)
    return namespace


class NamespaceMap(Namespace, collections.abc.MutableMapping):
    """Namespace that can act like a dict."""

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
