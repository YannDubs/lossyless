from functools import partial
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torchvision
import math
from typing import Iterable
from .helpers import weights_init, batch_flatten, batch_unflatten, prod

__all__ = ["get_Architecture"]


def get_Architecture(mode, complexity=2, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    mode : {"mlp","flattenmlp","resnet","cnn"}

    complexity : {0,...,4}, optional
        Complexity of the architecture. For `mlp` and `cnn` this is the width, for resnet this is 
        the depth.

    kwargs : 
        Additional arguments to the Module.
    
    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if mode == "mlp":
        # width 8,32,128,512,2048
        return partial(MLP, hid_dim=8 * (4 ** (complexity)), **kwargs)
    elif mode == "flattenmlp":
        return partial(FlattenMLP, hid_dim=8 * (4 ** (complexity)), **kwargs)
    elif mode == "resnet":
        base = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet150"]
        return partial(Resnet, base=base[complexity], **kwargs)
    elif mode == "cnn":
        # width of first layer 2,8,32,128,512
        # width of last layer 16,64,256,1024,4096
        return partial(CNN, hid_dim=2 * (4 ** (complexity)), **kwargs)
    else:
        raise ValueError(f"Unkown mode={mode}.")


### ClASSES ###


class MLP(nn.Module):
    """Multi Layer Perceptron.

    Parameters
    ----------
    in_dim : int

    out_dim : int

    hid_dim : int, optional
        Number of hidden neurones. 

    n_hid_layers : int, optional
        Number of hidden layers.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.
    """

    def __init__(
        self, in_dim, out_dim, n_hid_layers=1, hid_dim=128, norm_layer="identity"
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Norm = get_norm_layer(norm_layer, dim=1)

        layers = [nn.Linear(in_dim, hid_dim), Norm(hid_dim), nn.ReLU()]
        for _ in range(1, n_hid_layers):
            layers += [nn.Linear(hid_dim, hid_dim), Norm(hid_dim), nn.ReLU()]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.module = nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self, X):
        # flatten to make for normalizing layer => only 2 dim
        X, shape = batch_flatten(X)
        X = self.module(X)
        X = batch_unflatten(X, shape)
        return X

    def reset_parameters(self):
        weights_init(self)


class FlattenMLP(MLP):
    """
    MLP that can take a multi dimensional array as input and output (i.e. can be used with same
    input and output shape as CNN but permutation invariant.). E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.
    
    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `MLP`.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_dim=in_dim, out_dim=out_dim, **kwargs)

    def forward(self, X):
        # flattens in_shape
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))
        X = super().forward(X)
        # unflattens out_shape
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X


class Resnet(nn.Module):
    """Base class for renets.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying 
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the 
        first conv, and remove the max pooling layer as done (for cifar10) in 
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469. 
        
    out_dim : int, optional
        Size of the output. 
        
    base : {'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2'}, optional
        Base resnet to use, any model `torchvision.models.resnet` should work (the larger models were 
        not tested).

    is_pretrained : bool, optional
        Whether to load a model pretrained on imagenet. Might not work well with `is_small=True`.
        The last last fully connected layer will only be pretrained if `out_dim=1000`.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.
    """

    def __init__(
        self,
        in_shape,
        out_dim,
        base="resnet18",
        is_pretrained=False,
        norm_layer="batchnorm",
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_dim = out_dim
        self.resnet = torchvision.models.__dict__[base](
            pretrained=is_pretrained,
            num_classes=out_dim,
            norm_layer=get_norm_layer(norm_layer, 2),
        )

        if self.in_shape[1] < 100:
            # resnet for smaller images
            self.resnet.conv1 = nn.Conv2d(
                in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()

        if self.out_dim != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)

        self.reset_parameters()

    def forward(self, X):
        return self.resnet(X)

    def reset_parameters(self):
        # resnet is already correctly initialized
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)

        if self.out_dim != 1000:
            weights_init(self.resnet.fc)


class CNN(nn.Module):
    """CNN in shape of pyramid, which doubles hidden after each layer but decreases image size by 2. 
    
    Notes
    -----
    - Only works for 32*32 images. 
    - If `in_shape` and `out_dim` are reversed (i.e. `in_shape` is int) then will transpose the CNN.

    Parameters
    ----------
    in_shape : tuple of int 
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will 
        transpose ("reverse") the CNN.

    out_dim : int 
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose 
        ("reverse") the CNN.

    hid_dim : int, optional
        Base number of temporary channels (will be multiplied by 2 after each layer).

    norm_layer : callable or {"batchnorm", "identity"}
        Layer to return.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(
        self, in_shape, out_dim, hid_dim=32, norm_layer="batchnorm", **kwargs,
    ):

        super().__init__()

        if isinstance(out_dim, int) and not isinstance(in_shape, int):
            self.is_transpose = False
        else:
            in_shape, out_dim = out_dim, in_shape
            self.is_transpose = True

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.norm_layer = norm_layer

        assert self.in_shape[1] == self.in_shape[2] == 32

        n_layers = 4  # image size will go 32,16,8,4,2 (cannot go to 1 due to stride)
        # channels for hid_dim=32: 3,32,64,128,256
        channels = [self.in_shape[0]]
        channels += [self.hid_dim * (2 ** i) for i in range(0, n_layers)]

        if self.is_transpose:
            channels.reverse()

        layers = []
        in_chan = channels[0]
        for out_chan in channels[1:]:
            layers += self.make_block(in_chan, out_chan, **kwargs)
            in_chan = out_chan

        # mult by 4 because you stopped at 2*2 images
        if self.is_transpose:
            layers = [
                nn.Linear(self.out_dim, channels[0] * 4),
                nn.Unflatten(dim=-1, unflattened_size=(channels[0], 2, 2)),
            ] + layers
        else:
            layers += [
                nn.Flatten(start_dim=1),
                nn.Linear(channels[-1] * 4, self.out_dim),
            ]

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def make_block(self, in_chan, out_chan, **kwargs):
        Normalization = get_norm_layer(self.norm_layer, 2)
        if self.is_transpose:
            return [
                Normalization(in_chan),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_chan,
                    out_chan,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                    output_padding=1,
                    **kwargs,
                ),
            ]
        else:
            return [
                nn.Conv2d(
                    in_chan, out_chan, stride=2, padding=1, kernel_size=3, **kwargs
                ),
                Normalization(out_chan),
                nn.ReLU(),
            ]

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.model(X)


def get_norm_layer(norm_layer, dim=2):
    """Return the correct normalization layer.

    Parameters
    ----------
    norm_layer : callable or {"batchnorm", "identity"}
        Layer to return.

    dim : int, optional
        Number of dimension of the input (e.g. 2 for images).    
    """
    Batchnorms = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    if "batch" in norm_layer:
        Norm = Batchnorms[dim]
    elif norm_layer == "identity":
        Norm = nn.Identity
    elif isinstance(norm_layer, str):
        raise ValueError(f"Uknown normal_layer={norm_layer}")
    else:
        Norm = norm_layer
    return Norm
