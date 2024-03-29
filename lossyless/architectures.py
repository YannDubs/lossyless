import logging
import math
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transform_lib

from compressai.layers import GDN

from .helpers import (
    batch_flatten,
    batch_unflatten,
    closest_pow,
    is_pow2,
    prod,
    weights_init,
)

try:
    import clip
except ImportError:
    pass

try:
    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.models.self_supervised.simclr.transforms import (
        SimCLRFinetuneTransform,
    )
    from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
    from pl_bolts.models.self_supervised import SwAV
    from pl_bolts.models.self_supervised.swav.transforms import SwAVFinetuneTransform
except ImportError:
    pass

logger = logging.getLogger(__name__)
__all__ = ["get_Architecture"]


def get_Architecture(mode, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    mode : {"mlp","linear","resnet","identity", "balle", "clip", "clip_rn50", "simclr", "swav"}

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

    elif mode == "cnn":
        return partial(CNN, **kwargs)

    elif mode == "balle":
        return partial(BALLE, **kwargs)

    elif mode == "clip":
        return partial(PretrainedSSL, model="clip_vit", **kwargs)

    elif mode == "clip_rn50":
        return partial(PretrainedSSL, model="clip_rn50", **kwargs)

    elif mode == "simclr":
        return partial(PretrainedSSL, model="simclr", **kwargs)

    elif mode == "swav":
        return partial(PretrainedSSL, model="swav", **kwargs)

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

    out_shape : int or tuple
        Size of the output.

    base : {'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2'}, optional
        Base resnet to use, any model `torchvision.models.resnet` should work (the larger models were
        not tested).

    is_pretrained : bool, optional
        Whether to load a model pretrained on imagenet. Might not work well with `is_small=True`.

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
        kwargs = {}
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_pretrained = is_pretrained

        if not self.is_pretrained:
            # cannot load pretrained if wrong out dim
            kwargs["num_classes"] = self.out_dim

        self.resnet = torchvision.models.__dict__[base](
            pretrained=self.is_pretrained,
            norm_layer=get_Normalization(norm_layer, 2),
            **kwargs,
        )

        if self.is_pretrained:
            assert self.out_dim == self.resnet.fc.in_features
            # when pretrained has to remove last layer
            self.resnet.fc = torch.nn.Identity()

        if self.in_shape[1] < 100:
            # resnet for smaller images
            self.resnet.conv1 = nn.Conv2d(
                in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()

        self.reset_parameters()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)


class PretrainedSSL(nn.Module):
    """Pretrained self supervised models.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). Needs to be 3,224,224.

    out_shape : int or tuple
        Size of the output. Flattened needs to be 512 for clip_vit, 1024 for clip_rn50, and
        2048 for swav and simclr.

    model : {"swav", "simclr", "clip_vit", "clip_rn50"}
        Which SSL model to use.
    """

    def __init__(self, in_shape, out_shape, model):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.model = model
        self.out_dim = prod(self.out_shape)

        self.load_weights_()

        if self.model == "clip_vit":
            assert self.out_dim == 512
        elif self.model == "clip_rn50":
            assert self.out_dim == 1024
        elif self.model in ["swav", "simclr"]:
            assert self.out_dim == 2048
        else:
            raise ValueError(f"Unkown model={self.model}.")

        assert self.in_shape[0] == 3
        assert self.in_shape[1] == self.in_shape[2] == 224

        self.reset_parameters()

    def forward(self, X):
        z = self.encoder(X)
        z = z.unflatten(dim=-1, sizes=self.out_shape)
        return z

    def load_weights_(self):
        if self.model == "simclr":
            # load resnet50 pretrained using SimCLR on imagenet
            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            self.encoder = SimCLR.load_from_checkpoint(weight_path, strict=False)

        elif self.model == "swav":
            # load resnet50 pretrained using SwAV on imagenet
            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar"
            self.encoder = SwAV.load_from_checkpoint(weight_path, strict=False)

        elif "clip" in self.model:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            arch = "ViT-B/32" if "vit" in self.model else "RN50"
            model, _ = clip.load(arch, device, jit=False)
            self.encoder = model.visual  # only keep the image model

        else:
            raise ValueError(f"Unkown model={self.model}.")

        self.encoder.float()

    def reset_parameters(self):
        self.load_weights_()


class CNN(nn.Module):
    """CNN in shape of pyramid, which doubles hidden after each layer but decreases image size by 2.

    Notes
    -----
    - if some of the sides of the inputs are not power of 2 they will be resized to the closest power
    of 2 for prediction.
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
    n_layers : int, optional
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

        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.n_layers = n_layers

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
            pre_layers = [
                nn.Linear(self.out_dim, channels[0] * end_w * end_h, bias=is_bias),
                nn.Unflatten(dim=-1, unflattened_size=(channels[0], end_h, end_w)),
            ]
            post_layers = [resizer]

        else:
            pre_layers = [resizer]
            post_layers = [
                nn.Flatten(start_dim=1),
                nn.Linear(channels[-1] * end_w * end_h, self.out_dim),
                # last layer should always have bias
            ]

        self.model = nn.Sequential(*(pre_layers + layers + post_layers))

        self.reset_parameters()

    def validate_sizes(self, out_dim, in_shape):
        if isinstance(out_dim, int) and not isinstance(in_shape, int):
            self.is_transpose = False
        else:
            in_shape, out_dim = out_dim, in_shape
            self.is_transpose = True

        resizer = torch.nn.Identity()
        is_input_pow2 = is_pow2(in_shape[1]) and is_pow2(in_shape[2])
        if not is_input_pow2:
            # shape that you will work with which are power of 2
            in_shape_pow2 = list(in_shape)
            in_shape_pow2[1] = closest_pow(in_shape[1], base=2)
            in_shape_pow2[2] = closest_pow(in_shape[2], base=2)

            if self.is_transpose:
                # the model will output image of `in_shape_pow2` then will reshape to actual
                resizer = transform_lib.Resize((in_shape[1], in_shape[2]))
            else:
                # the model will first resize to power of 2
                resizer = transform_lib.Resize((in_shape_pow2[1], in_shape_pow2[2]))

            logger.warning(
                f"The input shape={in_shape} is not powers of 2 so we will rescale it and work with shape {in_shape_pow2}."
            )
            # for the rest treat the image as if pow 2
            in_shape = in_shape_pow2

        return in_shape, out_dim, resizer

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
