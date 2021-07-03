import logging
import math
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transform_lib

from compressai.layers import GDN

from .helpers import batch_flatten, batch_unflatten, prod, weights_init

try:
    import clip
except ImportError:
    pass

logger = logging.getLogger(__name__)
__all__ = ["get_Architecture"]


def get_Architecture(mode, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    mode : {"mlp","linear","resnet","identity", "balle", "clip"}

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if mode == "mlp":
        return partial(FlattenMLP, **kwargs)

    elif mode == "linear":
        return partial(FlattenLinear, **kwargs)

    elif mode == "identity":
        return torch.nn.Identity

    elif mode == "resnet":
        return partial(Resnet, **kwargs)

    elif mode == "balle":
        return partial(BALLE, **kwargs)

    elif mode == "clip":
        return partial(CLIPViT, **kwargs)

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

    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use.

    dropout_p : float, optional
        Dropout rate.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_hid_layers=1,
        hid_dim=128,
        norm_layer="identity",
        activation="ReLU",
        dropout_p=0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = get_Activation(activation)
        Dropout = nn.Dropout if dropout_p > 0 else nn.Identity
        Norm = get_Normalization(norm_layer, dim=1)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity

        layers = [
            nn.Linear(in_dim, hid_dim, bias=is_bias),
            Norm(hid_dim),
            Activation(),
            Dropout(p=dropout_p),
        ]
        for _ in range(1, n_hid_layers):
            layers += [
                nn.Linear(hid_dim, hid_dim, bias=is_bias),
                Norm(hid_dim),
                Activation(),
                Dropout(p=dropout_p),
            ]
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


class FlattenLinear(torch.nn.Linear):
    """
    Linear that can take a multi dimensional array as input and output . E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_features=in_dim, out_features=out_dim, **kwargs)

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

    out_shape : int or tuple, optional
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
        out_shape,
        base="resnet18",
        is_pretrained=False,
        norm_layer="batchnorm",
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)

        self.resnet = torchvision.models.__dict__[base](
            pretrained=is_pretrained,
            num_classes=self.out_dim,
            norm_layer=get_Normalization(norm_layer, 2),
        )

        if self.in_shape[1] < 100:
            # resnet for smaller images
            self.resnet.conv1 = nn.Conv2d(
                in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()

        # TODO should deal with the case of pretrained but out dim != 1000

        self.reset_parameters()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)


class CLIPViT(nn.Module):
    """Pretrained visual transformer using multimodal self supervised learning (CLIP).

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). Needs to be 3,224,224.

    out_shape : int or tuple, optional
        Size of the output. Flattened needs to be 512.

    kwargs :
        Additional argument to clip.load model.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.kwargs = kwargs

        self.load_weights_()

        assert self.out_dim == 512
        assert self.in_shape[0] == 3
        assert self.in_shape[1] == self.in_shape[2] == 224

        self.reset_parameters()

    def forward(self, X):
        z = self.vit(X)
        z = z.unflatten(dim=-1, sizes=self.out_shape)
        return z

    def load_weights_(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device, jit=False, **self.kwargs)
        model.float()
        self.vit = model.visual  # only keep the image model

    def reset_parameters(self):
        self.load_weights_()


class MLPCLIPViT(CLIPViT):
    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__(in_shape, out_shape, **kwargs)
        self.mlp = FlattenMLP(
            self.out_shape, self.out_shape, n_hid_layers=2, hid_dim=1024
        )

    def forward(self, X):
        z = super().forward(X)
        z = self.mlp(z)
        return z

    def reset_parameters(self):
        weights_init(self)
        super().reset_parameters()


class BALLE(nn.Module):
    """CNN from Balle's factorized prior. The key difference with the other encoders, is that it
    keeps some spatial structure in Z. I.e. representation can be seen as a flattened latent image.

    Notes
    -----
    - replicates https://github.com/InterDigitalInc/CompressAI/blob/a73c3378e37a52a910afaf9477d985f86a06634d/compressai/models/priors.py#L104

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will
        transpose ("reverse") the CNN.

    out_dim : int
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose
        ("reverse") the CNN.

    hid_dim : int, optional
        Number of channels for every layer.

    n_layers : int, optional
        Number of layers, after every layer divides image by 2 on each side.

    norm_layer : callable or {"batchnorm", "identity"}
        Normalization layer.

    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use. Typically that would be GDN for lossy image compression, but did not
        work for Galaxy (maybe because all black pixels).
    """

    validate_sizes = CNN.validate_sizes

    def __init__(
        self,
        in_shape,
        out_dim,
        hid_dim=256,
        n_layers=4,
        norm_layer="batchnorm",
        activation="ReLU",
    ):
        super().__init__()

        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.activation = activation
        self.norm_layer = norm_layer

        # divide length by 2 at every step until smallest is 2
        end_h = self.in_shape[1] // (2 ** self.n_layers)
        end_w = self.in_shape[2] // (2 ** self.n_layers)

        # channels of the output latent image
        self.channel_out_dim = self.out_dim // (end_w * end_h)

        layers = [
            self.make_block(self.hid_dim, self.hid_dim)
            for _ in range(self.n_layers - 2)
        ]

        if self.is_transpose:
            pre_layers = [
                nn.Unflatten(
                    dim=-1, unflattened_size=(self.channel_out_dim, end_h, end_w)
                ),
                self.make_block(self.channel_out_dim, self.hid_dim),
            ]
            post_layers = [
                self.make_block(self.hid_dim, self.in_shape[0], is_last=True),
                resizer,
            ]

        else:
            pre_layers = [resizer, self.make_block(self.in_shape[0], self.hid_dim)]
            post_layers = [
                self.make_block(self.hid_dim, self.channel_out_dim, is_last=True),
                nn.Flatten(start_dim=1),
            ]

        self.model = nn.Sequential(*(pre_layers + layers + post_layers))

        self.reset_parameters()

    def make_block(
        self, in_chan, out_chan, is_last=False, kernel_size=5, stride=2,
    ):
        if is_last:
            Norm = nn.Identity
        else:
            Norm = get_Normalization(self.norm_layer, 2)

        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity

        if self.is_transpose:
            conv = nn.ConvTranspose2d(
                in_chan,
                out_chan,
                kernel_size=kernel_size,
                stride=stride,
                output_padding=stride - 1,
                padding=kernel_size // 2,
                bias=is_bias,
            )
        else:
            conv = nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=is_bias,
            )

        if not is_last:
            Activation = get_Activation(self.activation, inverse=self.is_transpose)
            conv = nn.Sequential(conv, Norm(out_chan), Activation(out_chan))

        return conv

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.model(X)


def get_Normalization(norm_layer, dim=2):
    """Return the correct normalization layer.

    Parameters
    ----------
    norm_layer : callable or {"batchnorm", "identity"}U{any torch.nn layer}
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
        Norm = getattr(torch.nn, norm_layer)
    else:
        Norm = norm_layer
    return Norm


def get_Activation(activation, inverse=False):
    """Return an uninistantiated activation that takes the number of channels as inputs.

    Parameters
    ----------
    activation : {"gdn"}U{any torch.nn activation}
        Activation to use.

    inverse : bool, optional
        Whether using the activation in a transposed model.
    """
    if activation == "gdn":
        return partial(GDN, inverse=inverse)
    return getattr(torch.nn, activation)
