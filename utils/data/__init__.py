from .distributions import *
from .images import *


def get_datamodule(datamodule):
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "cifar10":
        return Cifar10DataModule
    elif datamodule == "mnist":
        return MnistDataModule
    elif datamodule == "imagenet":
        return ImagenetDataModule
    elif datamodule == "stl10":
        return STL10DataModule
    elif datamodule == "stl10unlabeled":
        return STL10UnlabeledDataModule
    elif datamodule == "food101":
        return Food101DataModule
    elif datamodule == "sun397":
        return Sun397DataModule
    elif datamodule == "cars196":
        return Cars196DataModule
    elif datamodule == "pets37":  # might drop
        return Pets37DataModule  # should use mean per class
    # elif datamodule == "caltech101": # might drop because next version
    #    return Caltech101DataModule  # should use mean per class
    elif datamodule == "pcam":
        return PCamDataModule
    elif datamodule == "flowers102":  # might drop
        return Flowers102DataModule  # should use mean per class
    elif "galaxy" in datamodule:
        return GalaxyDataModule
    elif datamodule == "banana":
        return BananaDataModule
    else:
        raise ValueError(f"Unkown datamodule: {datamodule}")
