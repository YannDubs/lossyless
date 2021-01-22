import collections
import logging
import os
from argparse import Namespace

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
