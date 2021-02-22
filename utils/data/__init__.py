from .analytic_images import *
from .distributions import *
from .images import *


def get_datamodule(datamodule):
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "cifar10":
        return Cifar10DataModule
    elif datamodule == "analytic_mnist":
        return AnalyticMnistDataModule
    elif datamodule == "mnist":
        return MnistDataModule
    elif datamodule == "fashionmnist":
        return FashionMnistDataModule
    elif datamodule == "imagenet":
        return ImagenetDataModule
    elif "galaxy" in datamodule:
        return GalaxyDataModule
    elif datamodule == "banana":
        return BananaDataModule
    else:
        raise ValueError("Unkown datamodule: {}".format(datamodule))
