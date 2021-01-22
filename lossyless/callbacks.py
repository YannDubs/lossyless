import math

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from matplotlib.lines import Line2D
from pytorch_lightning.callbacks import Callback

from .helpers import BASE_LOG, plot_density, setup_grid, to_numpy, undo_normalization

try:
    import wandb
except ImportError:
    pass


def save_img_wandb(pl_module, trainer, img, name, caption):
    """Save an image on wandb logger."""
    wandb_img = wandb.Image(img, caption=caption)
    trainer.logger.experiment.log({name: [wandb_img]}, commit=False)


class WandbReconstructImages(Callback):
    """Logs some reconstructed images on Wandb.

    Notes
    -----
    - the model should return a dictionary after each training step, containing
    a tensor "Y_hat" and a tensor "Y" both of image shape.
    - wandb needs to be in the loggers.
    - this will log one reconstructed image (+real) after each training epoch.
    """

    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        #! waiting for torch lighning #1243
        x_hat = pl_module._save["Y_hat"].float()
        x = pl_module._save["Y"].float()
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
        img = pl_module.distortion_estimator.q_YlZ(z)

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


class WandbCodebookPlot(Callback):
    """Plot the source distribution and codebook for a distribution.

    Notes
    -----
    - datamodule has to be `DistributionDataModule`.
    - distortion should be ivae.

    Parameters
    ----------
    plot_interval : int, optional
        How many epochs to wait before plotting.

    range_lim : int, optional
        Will plot x axis and y axis in (-range_lim, range_lim).

    n_pts : int, optional
        Number of points to use for the mesgrid will be n_pts**2. Can be memory heavy as all given
        in a single batch.

    figsize : tuple of int, optional
        Size fo figure.
    """

    def __init__(self, plot_interval=10, range_lim=5, n_pts=500, figsize=(9, 9)):
        super().__init__()
        self.plot_interval = plot_interval
        self.range_lim = range_lim
        self.n_pts = n_pts
        self.figsize = figsize

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_interval == 0:
            source = trainer.datamodule.distribution
            device = pl_module.device

            with torch.no_grad():
                pl_module.eval()

                # ensure cpu because memory ++
                pl_module_cpu = pl_module.to(torch.device("cpu"))
                fig = self.plot_quantization(pl_module_cpu, source)

            pl_module = pl_module.to(device)  # ensure back on correct
            pl_module.train()

            caption = f"ep: {trainer.current_epoch}"
            save_img_wandb(pl_module, trainer, fig, "quantization", caption)
            plt.close(fig)

    def quantize(self, pl_module, x):
        """Maps (through `idcs`) all elements in batch to the codebook and corresponding rates."""

        # batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlx = pl_module.p_ZlX(x)

        # shape: [batch_size, z_dim]
        z = p_Zlx.mean  # deterministic so should be same as sampling

        # shape: [batch_size, z_dim]
        # entropy bottleneck assumes n_z in first dim
        z_hat, q_z = pl_module.rate_estimator.entropy_bottleneck(z.unsqueeze(0))
        z_hat, q_z = z_hat.squeeze(), q_z.squeeze()

        # - log q(z). shape: [batch_shape]
        rates = -torch.log(q_z).sum(-1) / math.log(BASE_LOG)

        # shape: [batch_size, *y_shape]
        Y_hat = pl_module.distortion_estimator.q_YlZ(z_hat)

        # Find the unique set of latents for these inputs. Converts integer indexes
        # on the infinite lattice to scalar indexes into a codebook (which is only
        # valid for this set of inputs).
        z_hat = to_numpy(z_hat)
        _, i, idcs = np.unique(z_hat, return_index=True, return_inverse=True, axis=0)

        # shape: [n_codebook, *y_shape]
        codebook = to_numpy(Y_hat[i])

        # shape: [n_codebook]
        ratebook = to_numpy(rates[i])  # rate for each codebook

        return codebook, ratebook, idcs

    def plot_quantization(self, pl_module, source):
        """Return a figure of the source and codebooks."""

        # shape: [n_pts, n_pts, 2]
        xy = setup_grid(
            range_lim=self.range_lim, n_pts=self.n_pts, device=pl_module.device
        )

        # shape: [n_pts, n_pts, 2]
        flat_xy = einops.rearrange(xy, "x y d -> (x y) d", d=2)

        # codebook. shape: [n_codebook, 2]
        # ratebook, counts, p_codebook. shape: [n_codebook]
        # idcs. shape: [n_pts * n_pts]
        codebook, ratebook, idcs = self.quantize(pl_module, flat_xy)
        counts = np.bincount(idcs, minlength=len(codebook))
        p_codebook = BASE_LOG ** -(ratebook)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        plot_density(source, n_pts=self.n_pts, range_lim=self.range_lim, ax=ax)
        xy = to_numpy(xy)

        google_pink = (0xF4 / 255, 0x39 / 255, 0xA0 / 255)

        # contour lines
        ax.contour(
            xy[:, :, 0],
            xy[:, :, 1],
            idcs.reshape(self.n_pts, self.n_pts),
            np.arange(len(codebook)) + 0.5,
            colors=[google_pink],
            linewidths=0.5,
        )

        # codebook
        ax.scatter(
            codebook[counts > 0, 0],
            codebook[counts > 0, 1],
            color=google_pink,
            s=500 * p_codebook[counts > 0],  # size prop. to proba q(z)
        )

        return fig


class WandbMaxinvDistributionPlot(Callback):
    """Plot the distribtion of a maximal invariant p(M(X)) as well as the learned marginal
    q(M(X)) = E_{p(Z)}[q(M(X)|Z)].

    Notes
    -----
    - datamodule has to be `DistributionDataModule`.
    - distortion should be ivae.

    Parameters
    ----------
    plot_interval : int, optional
        How many epochs to wait before plotting.

    quantile_lim : int, optional
        Will plot M(X) in (quantile_lim,1-quantile_lim).

    n_pts : int, optional
        Number of points to sample to estimate the distribution.

    figsize : tuple of int, optional
        Size fo figure.

    equivalences : list of str, optional
        List of equivalences to use in case you are invariant to nothing.
    """

    def __init__(
        self,
        plot_interval=10,
        quantile_lim=5,
        n_pts=500 ** 2,
        figsize=(9, 9),
        equivalences=["rotation", "y_translation", "x_translation"],
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.quantile_lim = quantile_lim
        self.n_pts = n_pts
        self.figsize = figsize
        self.equivalences = equivalences

    def prepare(data, source):
        if source.decimals is not None:
            data = np.around(data, decimals=source.decimals)
        return data

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_interval == 0:
            dataset = trainer.datamodule.train_dataset
            # ensure don't change
            seed, device, equiv = dataset.seed, pl_module.device, dataset.equivalence
            pl_module.seed = None  # generate new data

            # generate new data
            x, _ = dataset.get_n_data_Mxs(self.n_pts)

            with torch.no_grad():
                pl_module.eval()

                # ensure cpu because memory ++
                pl_module_cpu = pl_module.to(torch.device("cpu"))
                # shape: [batch_size, z_dim]
                z_hat = pl_module_cpu(x)

                # shape: [batch_size, *x_shape]
                x_hat = pl_module_cpu.distortion_estimator.q_YlZ(z_hat)

                # allow computing plots for multiple equivalences (useful for banana without equivalence)
                equivalences = [equiv] if equiv is not None else self.equivalences
                for eq in equivalences:
                    dataset.equivalence = eq

                    # shape: [batch_size, *mx_shape]
                    mx_hat = dataset.max_invariant(x_hat)
                    mx = dataset.max_invariant(x)

                    fig = self.plot_maxinv(mx, mx_hat)

                    caption = f"ep: {trainer.current_epoch}"
                    save_img_wandb(pl_module, trainer, fig, f"max. inv. {eq}", caption)
                    plt.close(fig)

            # restore
            pl_module = pl_module.to(device)
            dataset.seed = seed
            dataset.equivalence = equiv
            pl_module.train()

    def plot_maxinv(self, mx, mx_hat):
        """Return a figure of the maximal invariant computed from the source and the reconstructions."""

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        data = pd.DataFrame(
            {"p(M(X))": mx.flatten().numpy(), "q(M(X))": mx_hat.flatten().numpy()}
        )

        h = sns.histplot(
            data=data[["q(M(X))"]],
            fill=True,
            element="bars",
            stat="density",
            discrete=True,
            color="tab:red",
            linestyle="-",
            lw=0.1,
            ax=ax,
            legend=False,
            alpha=0.3,
        )
        # color not working because not using x=
        for p in h.patches:
            p.set_edgecolor("tab:red")
            p.set_facecolor("tab:red")
            p.set_alpha(0.3)

        k = sns.kdeplot(
            data=data[["p(M(X))"]],
            ax=ax,
            fill=True,
            color="tab:blue",
            legend=False,
            alpha=0.1,
        )

        # manual legend because changes colors
        custom_lines = [
            Line2D([0], [0], color="tab:red", lw=1, linestyle="-"),
            Line2D([0], [0], color="tab:blue", lw=1),
        ]
        ax.legend(custom_lines, [r"$q(M(X))$", r"$p(M(X))$"])
        ax.set_xlim(
            data[["p(M(X))"]].quantile(0.001)[0],
            data[["p(M(X))"]].quantile(1 - 0.001)[0],
        )
        ax.set_xlabel(r"$M(X)$")
        sns.despine()

        return fig
