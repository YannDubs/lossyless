import torch
import torchvision
import math


from pytorch_lightning.callbacks import Callback
from pl_bolts.callbacks import ssl_online
import einops

from .helpers import undo_normalization

try:
    import wandb
except ImportError:
    pass


from torch.nn import functional as F


def save_img_wandb(pl_module, trainer, img, name, caption):
    """Save an image on wandb logger."""
    wandb_idx = pl_module.hparams.logger.loggers.index("wandb")
    wandb_img = wandb.Image(img, caption=caption)
    trainer.logger[wandb_idx].experiment.log({name: [wandb_img]})


class ClfOnlineEvaluator(ssl_online.SSLOnlineEvaluator):
    def __init__(self, *args, y_dim_pred=None, **kwargs):
        super().__init__(*args, num_classes=y_dim_pred, **kwargs)

    def to_device(self, batch, device):
        x, (y, _, _) = batch  # only return the real label
        batch = [x], y  # x assumes to be a list
        return super().to_device(batch, device)

    #! waiting for #4955
    def toggle_optimizer(self, trainer, pl_module):
        """activates current optimizer."""
        if len(trainer.optimizers) > 1:
            for param in pl_module.non_linear_evaluator.parameters():
                param.requires_grad = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.toggle_optimizer(trainer, pl_module)
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class RgrsOnlineEvaluator(ClfOnlineEvaluator):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Only change with ClfOnlineEvaluator is lmse_oss and not computing accuracy."""
        self.toggle_optimizer(trainer, pl_module)

        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.mse_loss(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Only change with ClfOnlineEvaluator is lmse_oss and not computing accuracy."""
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.mse_loss(mlp_preds, y)

        # log metrics
        pl_module.log(
            "online_val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True
        )


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

        #! waiting for torch lighning #1243
        x_hat = pl_module._save["rec_img"].float()
        x = pl_module._save["real_img"].float()
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


class AlphaScheduler(Callback):
    """
    Set the parameter `alpha` from a model. To replicate 
    `https://github.com/tensorflow/compression/blob/master/models/toy_sources/toy_sources.ipynb`. 
    """

    def on_epoch_start(self, trainer, logs=None):
        model = trainer.get_model()
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        if epoch < max_epochs / 4:
            model.force_alpha = 3 * (epoch + 1) / (max_epochs / 4 + 1)

