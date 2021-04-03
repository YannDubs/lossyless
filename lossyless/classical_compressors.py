import logging

import PIL

import pytorch_lightning as pl
import torch
import torch.nn as nn
from compressai.utils.bench import codecs
from pytorch_lightning.core.decorators import auto_move_data
from torchvision.transforms import ToPILImage, ToTensor

from .helpers import UnNormalizer, dict_mean, rename_keys_

__all__ = ["ClassicalCompressor"]

logger = logging.getLogger(__name__)


class PillowCodec(codecs.PillowCodec):
    def __init__(self, quality, *args):
        super().__init__(args)
        self.to_PIL = ToPILImage()
        self.to_tensor = ToTensor()
        self.quality = quality

    def batch_run(self, tensors, return_rec=True, return_metrics=True):
        """Perform compression on a batch of tensors."""
        batch, channel, height, width = tensors.shape
        device = tensors.device

        outs = []
        recs = []
        if height < 160 or width < 160:
            return_metrics = False  # cannot use mssim if smaller than 160

        for tensor in tensors:
            img = self.to_PIL(tensor.cpu().detach())
            out = self._run(
                img, self.quality, return_rec=return_rec, return_metrics=return_metrics
            )

            if return_rec:
                out, rec = out
                recs.append(self.to_tensor(rec))

            outs.append(out)

        batch_out = dict_mean(outs)
        batch_out["n_bits"] = batch_out["bpp"] * height * width
        rename_keys_(
            batch_out,
            {"encoding_time": "sender_time", "decoding_time": "receiver_time"},
        )

        if return_rec:
            batch_rec = torch.stack(recs).to(device)
            return batch_out, batch_rec

        return batch_out


class JPEG(PillowCodec):
    """Use libjpeg linked in Pillow"""

    fmt = "jpeg"
    _description = f"JPEG. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "JPEG"


#! webp shouldn't be used with black and white images because converts to colors!!
class WebP(PillowCodec):
    """Use libwebp linked in Pillow"""

    fmt = "webp"
    _description = f"WebP. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "WebP"


class PNG(PillowCodec):
    """
    Use ZLIB linked in Pillow for PNG (lossless) compression. 
    """

    fmt = "png"
    _description = f"PNG. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "PNG"


class Identity(codecs.Codec):
    """Placeholding compressor. Only has `run_from_tensor`. """

    _description = f"Identity."

    def __init__(self, *args):
        super().__init__(args)

    @property
    def name(self):
        return "identity"

    def batch_run(self, tensors, return_rec=True, return_metrics=True):
        """Perform no compression on a batch of tensors."""
        batch, channel, height, width = tensors.shape

        bpp = torch.finfo(tensors[0].dtype).bits * channel
        n_bits = bpp * height * width
        out = {
            "bpp": bpp,
            "n_bits": n_bits,
            "sender_time": 0,
            "receiver_time": 0,
        }

        if return_metrics:
            out["psnr"] = float("inf")  # division by 0
            out["ms-ssim"] = 1

        if return_rec:
            return out, tensors

        return out


class ClassicalCompressor(pl.LightningModule):
    is_features = False  # classical compressors never return Z

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.out_shape = self.hparams.data.shape  # always return reconstruction

        if self.hparams.data.kwargs.dataset_kwargs.is_normalize:
            dataset = self.hparams.data.dataset
            self.unormalizer = UnNormalizer(dataset)
        else:
            self.unormalizer = nn.Identity()

        if self.hparams.featurizer.mode is None:
            self.compressor = Identity()
        else:
            quality = self.hparams.featurizer.quality
            if self.hparams.featurizer.mode.lower() == "png":
                self.compressor = PNG(quality)
            elif self.hparams.featurizer.mode.lower() == "jpeg":
                self.compressor = JPEG(quality)
            elif self.hparams.featurizer.mode.lower() == "webp":
                self.compressor = WebP(quality)
            else:
                raise ValueError(f"Unkown featurizer={self.hparams.featurizer.mode}")

    # @auto_move_data  # move data on correct device for inference
    def forward(self, x, is_return_out=False, **kwargs):
        """Represents the data `x`.

        Parameters
        ----------
        X : torch.Tensor of shape=[batch_size, *data.shape]
            Data to compress.

        kwargs : 
            Placeholder.

        Returns
        -------
        X_hat : torch.Tensor of shape=[batch_size,  *data.shape]
            Reconstructed data. If image it's the unormalized version.

        out : dict
            Only if `is_return_out`. Dictionnary containing information shuch as reconstruction quality
            bits per pixel, compression time...
        """
        x = self.unormalizer(x)  # temporary unormalize for compression

        out, x_hat = self.compressor.batch_run(
            x, return_rec=True, return_metrics=is_return_out
        )

        if is_return_out:
            return x_hat, out

        return x_hat

    def step(self, batch):
        other = {}
        loss = 0
        x, _ = batch
        _, logs = self(x, is_return_out=True)
        return loss, logs, other

    def test_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch)
        self.log_dict(
            {f"test/{k}": v for k, v in logs.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    # uses placeholders (necessary for `setup`)
    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        pass

    def set_featurize_mode_(self):
        pass
