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


class MinMaxScaler:
    """
    Transforms last dim to the range [0, 1].

    Parameters
    ----------
    is_scale_each_dims_sep : bool, optional
        Whether to scale each feature / dimension separately.
    """

    def __init__(self, is_scale_each_dims=True):
        self.is_scale_each_dims = is_scale_each_dims

    def fit(self, X):
        if self.is_scale_each_dims:
            self.min_ = X.min(dim=0)[0]
            self.max_ = X.max(dim=0)[0]
        else:
            self.min_ = X.min()
            self.max_ = X.max()
        return self

    @property
    def dist(self):
        dist = self.max_ - self.min_
        dist[dist == 0.0] = 1.0  # avoid division by 0
        return dist

    def transform(self, X):
        return (X - self.min_) / self.dist

    def inverse_transform(self, X):
        return (X * self.dist) + self.min_

