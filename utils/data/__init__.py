from .distributions import *
from .images import *

def get_datamodule(datamodule):
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "cifar10":
        return Cifar10DataModule
    elif datamodule == "mnist":
        return MnistDataModule
    elif datamodule == "fashionmnist":
        return FashionMnistDataModule
    elif "galaxy" in datamodule:
        return GalaxyDataModule
    elif datamodule == "banana":
        return BananaDataModule
    else:
        raise ValueError("Unkown datamodule: {}".format(datamodule))
