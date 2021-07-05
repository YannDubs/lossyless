import struct
import time
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers


class ClipCompressor(nn.Module):
    """Our CLIP compressor.

    Parameters
    ----------
    pretrained_state_dict : dict or str or Path
        State dict of pretrained paths of the compressor. Can also be a path to the weights
        to load.

    is_jit : bool
        Whether to use just in time compilation => production ready.

    device : str
        Device on which to load the model.
    """

    def __init__(
        self,
        pretrained_state_dict,
        is_jit=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        model, self.preprocess = clip.load("ViT-B/32", jit=is_jit, device=device)
        self.clip = model.visual

        self.z_dim = 512
        self.side_z_dim = 512 // 5

        # => as if you use entropy coding that uses different scales in each dim
        self.scaling = torch.nn.Parameter(torch.ones(self.z_dim))
        self.biasing = torch.nn.Parameter(torch.zeros(self.z_dim))

        self.entropy_bottleneck = EntropyBottleneck(
            self.z_dim, init_scale=10, filters=[3, 3, 3, 3]
        )

        if not isinstance(pretrained_state_dict, dict):
            pretrained_state_dict = torch.load(pretrained_path)

        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            pretrained_state_dict,
        )  # compressai needs special because dynamic sized weights
        self.load_state_dict(pretrained_state_dict, strict=False)
        self.entropy_bottleneck.update()

        self.device = device
        self.to(self.device)
        self.eval()

    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.cuda.amp.autocast(False)  # precision here is important
    def forward(self, X, is_compress=False):
        """Takes a batch of images as input and featurizes it with or without compression.

        Parameters
        ----------
        X : torch.Tensor shape=(batch_size,3,224,224)
            Batch of images, should be normalized (with CLIP normalization).

        is_compress : bool, optional
            Whether to return the compressed features instead of decompressed.

        Return
        ------
        if is_compress:
            byte_str : bytes
        else:
            z_hat : torch.Tensor shape=(batch_size,512)
        """
        z = self.clip(X)

        z_in = self.process_z_in(z)

        if is_compress:
            out = self.entropy_bottleneck.compress(z_in)
        else:
            z_hat, _ = self.entropy_bottleneck(z_in)
            out = self.process_z_out(z_hat)

        return out

    def process_z_in(self, z):
        """Preprocessing of representation before entropy bottleneck."""
        z_in = (z.float() + self.biasing) * self.scaling.exp()
        # compressai needs 4 dimension (images) as input
        return z_in.unsqueeze(-1).unsqueeze(-1)

    def process_z_out(self, z_hat):
        """Postprocessing of representation after entropy bottleneck."""
        # back to vector
        z_hat = z_hat.squeeze(-1).squeeze(-1)
        return (z_hat / self.scaling.exp()) - self.biasing

    def compress(self, X):
        """Return comrpessed features (byte string)."""
        with torch.no_grad():
            return self(X, is_compress=True)

    def decompress(self, byte_str):
        """Decompressed the byte strings."""
        with torch.no_grad():
            z_hat = self.entropy_bottleneck.decompress(byte_str, [1, 1])
            return self.process_z_out(z_hat)

    def get_rate(self, X):
        """Compute actual (after entropy coding) mean (over a batch) rate per image. In bits."""
        byte_str = self.compress(X)

        # sum over all latents (for hierachical). mean over batch.
        n_bytes = sum([len(s) for s in byte_str]) / len(byte_str)
        n_bits = n_bytes * 8

        return n_bits

    def make_pickable_(self):
        """Ensure that the estimator is pickable, which is necessary for distributed training."""
        for m in self.modules():  # recursive iteration over all
            if isinstance(m, EntropyModel):
                m.entropy_coder = None

    def undo_pickable_(self):
        """Undo `make_pickable_`, e.g. ensures that the coder is available."""
        for m in self.modules():
            if isinstance(m, EntropyModel):
                # TODO : allow resetting non default coder
                m.entropy_coder = _EntropyCoder(default_entropy_coder())

    def compress_dataset(
        self,
        dataset,
        file,
        label_file=None,
        kwargs_dataloader=dict(batch_size=128, num_workers=16),
        is_info=True,
    ):
        """Compress a torchvision dataset and save it to a file.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset from which to load the data. Note: the dataset whould be (clip)
            normalized and of shape (3,224,224). The easiest is to use self.preprocess
            as transforms.
            
        file : str or Path
            Path to which to save the compressed images.
            
        label_file : str or Path, optional
            File to which to save the labels (if given).
            
        is_info : bool, optional
            Whether to print compression time and mean rate per image.
        """
        if self.device == "cpu":
            raise ValueError("Compression only implemented on GPU (as uses fp16).")

        start = time.time()

        Z_bytes, Y = [], []
        for x, *y in tqdm.tqdm(DataLoader(dataset, **kwargs_dataloader)):
            Z_bytes += self.compress(x.to(self.device).half())
            if label_file is not None:
                Y += [y[0].cpu().numpy().astype(np.uint16)]

        # save representations
        with Path(file).open("wb") as f:
            write_uints(f, (len(Z_bytes),))
            for b in Z_bytes:
                write_uints(f, (len(b),))
                write_bytes(f, b)

        enc_time = (time.time() - start) / len(Z_bytes)
        rate = 8 * Path(file).stat().st_size / len(Z_bytes)

        # save labels
        if label_file is not None:
            # no pickle for portability
            np.save(label_file, np.concatenate(Y), allow_pickle=False)

        if is_info:
            print(f"Rate: {rate:.2f} bits/img | Encoding: {1/enc_time:.2f} img/sec ")

    def decompress_dataset(self, file, label_file=None, is_info=True, is_cpu=True):
        """Decompresses a dataset saved on file and return numpy array.
        
        Parameters
        ----------
        file : str or Path
            Path from which to load the compressed images.
            
        label_file : str or Path, optional
            File from which to load the labels (if given).
            
        is_info : bool, optional
            Whether to print decompression time.
            
        is_cpu : bool, optional
            Whether to decompress on CPU even if GPU available. This can be a little quicker
            as we do not perform batch decompression.
        """
        if is_cpu:
            old_device = self.device
            self.to("cpu")

        start = time.time()

        with Path(file).open("rb") as f:
            n_Z = read_uints(f, 1)[0]
            Z_hat = [None] * n_Z
            for i in tqdm.tqdm(range(n_Z)):
                s = read_bytes(f, read_uints(f, 1)[0])
                Z_hat[i] = self.decompress([s]).cpu().numpy()

        Z_hat = np.concatenate(Z_hat)

        dec_time = (time.time() - start) / len(Z_hat)

        if is_info:
            print(f"Decoding: {1/dec_time:.2f} img/sec ")

        if is_cpu:
            self.to(old_device)

        if label_file is not None:
            Y = np.load(label_file, allow_pickle=False).astype(np.int64)
            return Z_hat, Y

        return Z_hat


# taken from https://github.com/InterDigitalInc/CompressAI/blob/976e05234b249678576bdc7909431598f518ee05/examples/codec.py
def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]
