dependencies = [
    "torch",
    "compressai",
    "clip",
    "tqdm",
    "numpy",
]  # dependencies required for loading a model

import os

import torch

from hub import ClipCompressor as _ClipCompressor

PATH = "https://github.com/YannDubs/lossyless/releases/download/v0.1-alpha/beta{beta:0.0e}_factorized_rate.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clip_compressor_b005(is_jit=False, device=DEVICE):
    ckpt_path = PATH.format(beta=0.05)
    pretrained_state_dict = torch.hub.load_state_dict_from_url(
        ckpt_path, progress=False
    )
    compressor = _ClipCompressor(
        pretrained_state_dict=pretrained_state_dict, is_jit=is_jit, device=device
    )
    return compressor, compressor.preprocess


def clip_compressor_b001(is_jit=False, device=DEVICE):
    ckpt_path = PATH.format(beta=0.01)
    pretrained_state_dict = torch.hub.load_state_dict_from_url(
        ckpt_path, progress=False
    )
    compressor = _ClipCompressor(
        pretrained_state_dict=pretrained_state_dict, is_jit=is_jit, device=device
    )
    return compressor, compressor.preprocess


def clip_compressor_b01(is_jit=False, device=DEVICE):
    ckpt_path = PATH.format(beta=0.1)
    pretrained_state_dict = torch.hub.load_state_dict_from_url(
        ckpt_path, progress=False
    )
    compressor = _ClipCompressor(
        pretrained_state_dict=pretrained_state_dict, is_jit=is_jit, device=device
    )
    return compressor, compressor.preprocess


DOCSTRING = """
    Load invariant CLIP compressor with beta={beta:.0e} (beta proportional to compression not like in paper).

    Parameters
    ----------
    is_jit : bool
        Whether to use just in time compilation => production ready.

    device : str
        Device on which to load the model.

    Return
    ------
    compressor : nn.Module
        Pytorch module that when called as `compressor(X)` on a batch of image, will return
        decompressed representations. Use `compressor.compress(X)` to get a batch of compressed 
        representations (in bytes). To save a compressed torch dataset to file use 
        `compressor.compress_dataset(dataset,file)` and `dataset = compressor.decompress_dataset(file)`.
        For more information check the docstrings of all functions of the module and the examples below.

    transform : callable
        Transforms that can be used directly in place of torchvision transform. It will resize the
        image to (3,224,224), apply clip normalization and convert it to tensor.
    """

clip_compressor_b005.__doc__ = DOCSTRING.format(beta=0.05)
clip_compressor_b001.__doc__ = DOCSTRING.format(beta=0.01)
clip_compressor_b01.__doc__ = DOCSTRING.format(beta=0.1)
