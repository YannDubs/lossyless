import torch
import torchvision
import math

import pl_bolts
from pytorch_lightning.callbacks import Callback
from pl_bolts.callbacks import ssl_online, LatentDimInterpolator
import einops

from .helpers import undo_normalization

try:
    import wandb
except ImportError:
    pass

from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F


def save_img_wandb(pl_module, trainer, img, name, caption):
    """Save an image on wandb logger."""
    wandb_idx = pl_module.hparams.logger.loggers.index("wandb")
    wandb_img = wandb.Image(img, caption=caption)
    trainer.logger[wandb_idx].experiment.log({name: [wandb_img]})


class SSLOnlineEvaluator(ssl_online.SSLOnlineEvaluator):
    #! check why not logging training waiting for lighning #4857

    def to_device(self, batch, device):
        x, (y, _) = batch  # only return the real label
        batch = [x], y  # x assumes to be a list
        return super().to_device(batch, device)


class WandbReconstructImages(Callback):
    """Logs some reconstructed images on Wandb.

    Notes
    -----
    - the model should return a dictionary after each training step, containing
    a tensor "y_hat" and a tensor "target" both of image shape.
    - wandb needs to be in the loggers.
    - this will log one reconstructed image (+real) after each training epoch.
    """

    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        x_hat = trainer.hiddens["y_hat"]
        x = trainer.hiddens["target"]
        # undo normalization for plotting
        x_hat, x = undo_normalization(x_hat, x, pl_module.hparams.data.dataset)
        caption = f"ep: {trainer.current_epoch}"
        save_img_wandb(pl_module, trainer, x_hat, "rec_img", caption)
        save_img_wandb(pl_module, trainer, x, "real_img", caption)


class WandbLatentDimInterpolator(Callback):
    """Logs interpolated images (steps through the first 2 dimensions) on Wandb.

    Parameters
    ----------
    z_dim : int
        Number of dimensions for latents.

    plot_interval : int, optional
        Every how many epochs to plot.

    range_start : float, optional
        Start of the interpolating range.

    range_end : float, optional
        End of the interpolating range.

    n_per_lat : int, optional
        Number of traversal to do for each latent.

    n_lat_traverse : int, optional
        Number of latent to traverse for traversal 1_d. Max is `z_dim`.
    """

    def __init__(
        self,
        z_dim,
        plot_interval=20,
        range_start=-5,
        range_end=5,
        n_per_lat=10,
        n_lat_traverse=10,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.plot_interval = plot_interval
        self.range_start = range_start
        self.range_end = range_end
        self.n_per_lat = n_per_lat
        self.n_lat_traverse = n_lat_traverse

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_interval == 0:
            with torch.no_grad():
                pl_module.eval()
                traversals_2d = self.latent_traverse_2d(pl_module)
                traversals_1d = self.latent_traverse_1d(pl_module)
            pl_module.train()

            caption = f"ep: {trainer.current_epoch}"
            save_img_wandb(pl_module, trainer, traversals_2d, "traversals_2d", caption)
            save_img_wandb(pl_module, trainer, traversals_1d, "traversals_1d", caption)

    def _traverse_line(self, idx, pl_module, z=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx."""

        if z is None:
            z = torch.zeros(1, self.n_per_lat, self.z_dim, device=pl_module.device)

        traversals = torch.linspace(
            self.range_start,
            self.range_end,
            steps=self.n_per_lat,
            device=pl_module.device,
        )
        for i in range(self.n_per_lat):
            z[:, i, idx] = traversals[i]

        z = einops.rearrange(z, "r c ... -> (r c) ...")
        img = pl_module.q_YlZ(z)

        # undo normalization for plotting
        img, _ = undo_normalization(img, img, pl_module.hparams.data.dataset)
        return img

    def latent_traverse_2d(self, pl_module):
        """Traverses the first 2 latents TOGETHER."""
        traversals = torch.linspace(
            self.range_start,
            self.range_end,
            steps=self.n_per_lat,
            device=pl_module.device,
        )
        z_2d = torch.zeros(
            self.n_per_lat, self.n_per_lat, self.z_dim, device=pl_module.device
        )
        for i in range(self.n_per_lat):
            z_2d[i, :, 0] = traversals[i]  # fill first latent

        imgs = self._traverse_line(1, pl_module, z=z_2d)  # fill 2nd latent and rec.
        grid = torchvision.utils.make_grid(imgs, nrow=self.n_per_lat)

        return grid

    def latent_traverse_1d(self, pl_module):
        """Traverses the first `self.n_lat` latents separately."""
        n_lat_traverse = min(self.n_lat_traverse, self.z_dim)
        imgs = [self._traverse_line(i, pl_module) for i in range(n_lat_traverse)]
        imgs = torch.cat(imgs, dim=0)
        grid = torchvision.utils.make_grid(imgs, nrow=self.n_per_lat)
        return grid

