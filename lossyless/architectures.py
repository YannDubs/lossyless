import math
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
from compressai.layers import GDN
from pl_bolts.models.self_supervised import SimCLR

from .helpers import batch_flatten, batch_unflatten, is_pow2, prod, weights_init

try:
    import clip
except ImportError:
    pass


__all__ = ["get_Architecture"]


def get_Architecture(mode, complexity=None, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    mode : {"mlp","flattenmlp","resnet","cnn"}

    complexity : {0,...,4,None}, optional
        Complexity of the architecture. For `mlp` and `cnn` this is the width, for resnet this is
        the depth. None lets `kwargs` have the desired argument.

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if mode == "mlp":
        if complexity is not None:
            # width 8,32,128,512,2048
            kwargs["hid_dim"] = 8 * (4 ** (complexity))
        return partial(FlattenMLP, **kwargs)

    elif mode == "resnet":
        if complexity is not None:
            base = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet150"]
            kwargs["base"] = base[complexity]
        return partial(Resnet, **kwargs)

    elif mode == "cnn":
        if complexity is not None:
            # width of first layer 2,8,32,128,512
            # width of last layer 16,64,256,1024,4096
            kwargs["hid_dim"] = 2 * (4 ** (complexity))

        return partial(CNN, **kwargs)

    elif mode == "simclr":
        return partial(SimCLRResnet50, **kwargs)
    elif mode == "simclr_projector":
        return partial(SimCLRProjector, **kwargs)
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

        if self.out_dim != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.out_dim)

        self.reset_parameters()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)

        if self.out_dim != 1000:
            weights_init(self.resnet.fc)


class CLIPResnet50(nn.Module):
    """Pretrained Resnet50 using multimodal self supervised learning (CLIP).

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the
        first conv, and remove the max pooling layer as done (for cifar10) in
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469.

    out_shape : int or tuple, optional
        Size of the output.

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

        if self.out_dim != 1024:
            self.resizer = nn.Linear(1024, self.out_dim)
        else:
            self.resizer = nn.Identity()

        self.reset_parameters()

    def forward(self, X):
        z = self.resnet(X)
        z = self.resizer(z)
        z = z.unflatten(dim=-1, sizes=self.out_shape)
        return z

    def load_weights_(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("RN50", device, jit=False, **self.kwargs)
        self.resnet = model.visual  # only keep the image model

    def reset_parameters(self):
        self.load_weights_()


class SimCLRResnet50(nn.Module):
    """Pretrained Resnet50 using self supervised learning (simclr).

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the
        first conv, and remove the max pooling layer as done (for cifar10) in
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469.

    out_shape : int or tuple, optional
        Size of the output.

    is_post_project : bool, optional
        Whether to return the output after projection head instead of before.

    kwargs : 
        Additional argument to the pl_bolts model.
    """

    def __init__(self, in_shape, out_shape, is_post_project=False, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_post_project = is_post_project
        self.kwargs = kwargs

        self.load_weights_()

        if self.is_post_project and self.out_dim != 128:
            self.resizer = nn.Linear(128, self.out_dim)
        elif (not self.is_post_project) and self.out_dim != 2048:
            self.resizer = nn.Linear(2048, self.out_dim)
        else:
            self.resizer = nn.Identity()

        self.reset_parameters()

    def forward(self, X):
        z = self.resnet(X)

        if isinstance(z, (list, tuple)):
            z = z[-1]  # bolts resnet currently returns list

        z = self.projector(z)
        z = self.resizer(z)
        z = z.unflatten(dim=-1, sizes=self.out_shape)
        return z

    def load_weights_(self):
        if self.in_shape[1] < 100:
            maxpool1 = False
            first_conv = False
        else:
            maxpool1 = True
            first_conv = True

        module = SimCLR(
            gpus=0,
            nodes=1,
            num_samples=1,
            batch_size=64,
            maxpool1=maxpool1,
            first_conv=first_conv,
            dataset="",  # not importnant,
            **self.kwargs,
        )

        ckpt_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
        module = module.load_from_checkpoint(ckpt_path, strict=False)

        self.resnet = module.encoder

        if self.is_post_project:
            self.projector = module.projection
        else:
            self.projector = nn.Identity()

    def reset_parameters(self):
        weights_init(self.resizer)
        self.load_weights_()


class SimCLRProjector(nn.Module):
    """Pretrained simclr projector head."""

    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        if isinstance(in_shape, int):
            in_shape = [in_shape]
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        assert in_shape[0] == prod(in_shape) == 2048
        assert out_shape[0] == prod(out_shape) == 128

        self.load_weights_()
        self.reset_parameters()

    def forward(self, X):
        #  make sure only 2 dims
        X, shape = batch_flatten(X)
        X = self.projection(X)
        X = batch_unflatten(X, shape)
        return X

    def load_weights_(self):
        # give random non important args
        module = SimCLR(
            gpus=0, nodes=1, num_samples=1, batch_size=64, dataset="", **self.kwargs
        )
        ckpt_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
        module.load_from_checkpoint(ckpt_path, strict=False)
        self.projection = module.projection

    def reset_parameters(self):
        self.load_weights_()


class CNN(nn.Module):
    """CNN in shape of pyramid, which doubles hidden after each layer but decreases image size by 2.

    Notes
    -----
    - Only works for images whose side is a power of 2 (not necessarily  squared).
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

    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use.

    n_layers : int, optional, optional
        Number of layers. If `None` uses the required number of layers so that the smallest side 
        is 2 after encoding (i.e. one less than the maximum).

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(
        self,
        in_shape,
        out_dim,
        hid_dim=32,
        norm_layer="batchnorm",
        activation="ReLU",
        n_layers=None,
        **kwargs,
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
        self.activation = activation
        self.n_layers = n_layers

        # if want non pow2 you can pass it through a linear layer to get the closest power of 2
        # and if transposed a linear layer to get in_shape. But better if it's done when image is
        # smaller (or best is simply to resize it to power of 2)
        assert is_pow2(self.in_shape[1]) and is_pow2(self.in_shape[2])

        if self.n_layers is None:
            # divide length by 2 at every step until smallest is 2
            min_side = min(self.in_shape[1], self.in_shape[2])
            self.n_layers = int(math.log2(min_side) - 1)

        Norm = get_Normalization(self.norm_layer, 2)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity

        # for size 32 will go 32,16,8,4,2
        # channels for hid_dim=32: 3,32,64,128,256
        channels = [self.in_shape[0]]
        channels += [self.hid_dim * (2 ** i) for i in range(0, self.n_layers)]
        end_h = self.in_shape[1] // (2 ** self.n_layers)
        end_w = self.in_shape[2] // (2 ** self.n_layers)

        if self.is_transpose:
            channels.reverse()

        layers = []
        in_chan = channels[0]
        for i, out_chan in enumerate(channels[1:]):
            is_last = i == len(channels[1:]) - 1
            layers += self.make_block(
                in_chan, out_chan, Norm, is_bias, is_last, **kwargs
            )
            in_chan = out_chan

        if self.is_transpose:
            layers = [
                nn.Linear(self.out_dim, channels[0] * end_w * end_h, bias=is_bias),
                nn.Unflatten(dim=-1, unflattened_size=(channels[0], end_h, end_w)),
            ] + layers
        else:
            layers += [
                nn.Flatten(start_dim=1),
                nn.Linear(channels[-1] * end_w * end_h, self.out_dim),
                # last layer should always have bias
            ]

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def make_block(self, in_chan, out_chan, Norm, is_bias, is_last, **kwargs):

        if self.is_transpose:
            Activation = get_Activation(self.activation, inverse=True)
            return [
                Norm(in_chan),
                Activation(in_chan),
                nn.ConvTranspose2d(
                    in_chan,
                    out_chan,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                    output_padding=1,
                    bias=is_bias or is_last,
                    **kwargs,
                ),
            ]
        else:
            Activation = get_Activation(self.activation, inverse=True)
            return [
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                    bias=is_bias,
                    **kwargs,
                ),
                Norm(out_chan),
                Activation(out_chan),
            ]

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
