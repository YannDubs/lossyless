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


def clip_compressor(beta=5e-02, pretrained=True, **kwargs):
    """Load an invariant CLIP compressor.

    Parameters
    ----------
    beta : {1e-02, 5e-02, 1e-01}, optional
        What beta value to use. Larger means stronger compressor. This
        correspond to 1/beta in the paper (OOPS).

    pretrained : bool, optional
        Whether to load pretrained model, currently must be true.

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
    base = "https://github.com/YannDubs/lossyless/releases/download"
    ckpt = f"{base}/v0.1-alpha/beta{beta:0.0e}_factorized_rate.pt"
    pretrained_state_dict = torch.hub.load_state_dict_from_url(ckpt, progress=False)

    assert pretrained

    compressor = _ClipCompressor(pretrained_state_dict=pretrained_state_dict, **kwargs)

    return compressor, compressor.preprocess
