import os
from omegaconf import OmegaConf
from argparse import Namespace
import collections


def create_folders(base_path, names):
    """Safely creates a list of folders ar a certain path."""
    for name in names:
        path = os.path.join(base_path, name)
        if not os.path.exists(path):
            os.makedirs(path)


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


class NamespaceMap(Namespace, collections.abc.Mapping):
    """Namespace that can act like a dict."""

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

