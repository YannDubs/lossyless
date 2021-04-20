import logging
import math
from typing import Hashable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import einops
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from .helpers import (
    BASE_LOG,
    UnNormalizer,
    is_colored_img,
    plot_config,
    plot_density,
    setup_grid,
    tensors_to_fig,
    to_numpy,
)

try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)


def save_img(pl_module, trainer, img, name, caption):
    """Save an image on logger. Currently only Tensorboard and wandb."""
    experiment = trainer.logger.experiment
    if isinstance(trainer.logger, WandbLogger):
        wandb_img = wandb.Image(img, caption=caption)
        experiment.log({name: [wandb_img]}, commit=False)

    elif isinstance(trainer.logger, TensorBoardLogger):
        # TODO @karen the following is not tested
        if isinstance(img, matplotlib.figure.Figure):
            experiment.add_figure(name, img, global_step=trainer.global_step)
        else:
            experiment.add_image(name, img, global_step=trainer.global_step)

    else:
        err = f"Plotting images is only available on tensorboard and Wandb but you are using {type(trainer.logger)}."
        raise ValueError(err)


def is_plot(trainer, plot_interval):
    is_plot_interval = (trainer.current_epoch + 1) % plot_interval == 0
    is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
    return is_plot_interval or is_last_epoch


class PlottingCallback(Callback):
    """Base classes for calbacks that plot.

    Parameters
    ----------
    plot_interval : int, optional
        Every how many epochs to plot.

    plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...
    """

    def __init__(self, plot_interval=10, plot_config_kwargs={}):
        super().__init__()
        self.plot_interval = plot_interval
        self.plot_config_kwargs = plot_config_kwargs

    @rank_zero_only  # only plot on one machine
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if is_plot(trainer, self.plot_interval):
            try:
                for fig, kwargs in self.yield_figs_kwargs(trainer, pl_module):
                    if "caption" not in kwargs:
                        kwargs["caption"] = f"ep: {trainer.current_epoch}"

                    save_img(pl_module, trainer, fig, **kwargs)
                    plt.close(fig)
            except:
                logger.exception(f"Couldn't plot for {type(PlottingCallback)}, error:")

    def yield_figs_kwargs(self, trainer, pl_module):
        raise NotImplementedError()


class ReconstructImages(PlottingCallback):
    """Logs some reconstructed images.

    Notes
    -----
    - the model should return a dictionary after each training step, containing
    a tensor "Y_hat" and a tensor "Y" both of image shape.
    - this will log one reconstructed image (+real) after each training epoch.
    """

    def yield_figs_kwargs(self, trainer, pl_module):
        cfg = pl_module.hparams
        #! waiting for torch lighning #1243
        x_hat = pl_module._save["Y_hat"].float()
        x = pl_module._save["X"].float()

        if is_colored_img(x):
            if cfg.data.kwargs.dataset_kwargs.is_normalize:
                # undo normalization for plotting
                unnormalizer = UnNormalizer(cfg.data.dataset)
                x = unnormalizer(x)

        yield x_hat, dict(name="rec_img")

        yield x, dict(name="real_img")


class LatentDimInterpolator(PlottingCallback):
    """Logs interpolated images.

    Parameters
    ----------
    z_dim : int
        Number of dimensions for latents.

    range_start : float, optional
        Start of the interpolating range.

    range_end : float, optional
        End of the interpolating range.

    n_per_lat : int, optional
        Number of traversal to do for each latent.

    n_lat_traverse : int, optional
        Number of latent to traverse for traversal 1_d. Max is `z_dim`.

    kwargs :
        Additional arguments to PlottingCallback.
    """

    def __init__(
        self,
        z_dim,
        range_start=-5,
        range_end=5,
        n_per_lat=7,
        n_lat_traverse=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.range_start = range_start
        self.range_end = range_end
        self.n_per_lat = n_per_lat
        self.n_lat_traverse = n_lat_traverse

    def yield_figs_kwargs(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            with plot_config(**self.plot_config_kwargs, font_scale=2):
                traversals_2d = self.latent_traverse_2d(pl_module)

            with plot_config(**self.plot_config_kwargs, font_scale=1.5):
                traversals_1d = self.latent_traverse_1d(pl_module)

        pl_module.train()

        yield traversals_2d, dict(name="traversals_2d")
        yield traversals_1d, dict(name="traversals_1d")

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

        # put back to [0,1]
        img = torch.sigmoid(img)
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
        fig = tensors_to_fig(
            imgs,
            n_cols=self.n_per_lat,
            x_labels=["1st Latent"],
            y_labels=["2nd Latent"],
        )

        return fig

    def latent_traverse_1d(self, pl_module):
        """Traverses the first `self.n_lat` latents separately."""
        n_lat_traverse = min(self.n_lat_traverse, self.z_dim)
        imgs = [self._traverse_line(i, pl_module) for i in range(n_lat_traverse)]
        imgs = torch.cat(imgs, dim=0)
        fig = tensors_to_fig(
            imgs,
            n_cols=self.n_per_lat,
            x_labels=["Sweeps"],
            y_labels=[f"Lat. {i}" for i in range(n_lat_traverse)],
        )
        return fig


class CodebookPlot(PlottingCallback):
    """Plot the source distribution and codebook for a distribution.

    Notes
    -----
    - datamodule has to be `DistributionDataModule`.
    - Z should be deterministic

    Parameters
    ----------
    range_lim : int, optional
        Will plot x axis and y axis in (-range_lim, range_lim).

    n_pts : int, optional
        Number of points to use for the mesgrid will be n_pts**2. Can be memory heavy as all given
        in a single batch.

    figsize : tuple of int, optional
        Size fo figure.

    is_plot_codebook : bool, optional
        Whether to plot the codebook or only the quantization space. This can only be true for VAE
        and iVAE, not iNCE because it doesn't reconstruct an element in X space.

    kwargs :
        Additional arguments to PlottingCallback.
    """

    def __init__(
        self, range_lim=5, n_pts=500, figsize=(9, 9), is_plot_codebook=True, **kwargs,
    ):
        super().__init__(**kwargs)
        self.range_lim = range_lim
        self.n_pts = n_pts
        self.figsize = figsize
        self.is_plot_codebook = is_plot_codebook

    def yield_figs_kwargs(self, trainer, pl_module):
        source = trainer.datamodule.distribution

        with torch.no_grad():
            pl_module.eval()

            with plot_config(**self.plot_config_kwargs):
                fig = self.plot_quantization(pl_module, source)

        pl_module.train()

        yield fig, dict(name="quantization")

    def quantize(self, pl_module, x):
        """Maps (through `idcs`) all elements in batch to the codebook and corresponding rates."""

        # batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlx = pl_module.p_ZlX(x)

        # shape: [1, batch_size, z_dim]
        z = p_Zlx.mean.unsqueeze(0)

        # shape: [batch_size, z_dim]
        z_hat, rates, _, __ = pl_module.rate_estimator(z, p_Zlx, pl_module)
        z_hat, rates = z_hat.squeeze(0), rates.squeeze(0)

        # Find the unique set of latents for these inputs. Converts integer indexes
        # on the infinite lattice to scalar indexes into a codebook (which is only
        # valid for this set of inputs).
        _, i, idcs = np.unique(
            to_numpy(z_hat), return_index=True, return_inverse=True, axis=0
        )

        # shape: [n_codebook]
        ratebook = to_numpy(rates[i])  # rate for each codebook

        if self.is_plot_codebook:
            # shape: [batch_size, *y_shape]
            Y_hat = pl_module.distortion_estimator.q_YlZ(z_hat)

            # shape: [n_codebook, *y_shape]
            codebook = to_numpy(Y_hat[i])
        else:
            codebook = None

        return codebook, ratebook, idcs

    def plot_quantization(self, pl_module, source):
        """Return a figure of the source and codebooks."""

        # shape: [n_pts, n_pts, 2]
        xy = setup_grid(
            range_lim=self.range_lim, n_pts=self.n_pts, device=pl_module.device
        )

        # shape: [n_pts * n_pts, 2]
        flat_xy = einops.rearrange(xy, "x y d -> (x y) d", d=2)

        # codebook. shape: [n_codebook, 2]
        # ratebook, counts, p_codebook. shape: [n_codebook]
        # idcs. shape: [n_pts * n_pts]
        codebook, ratebook, idcs = self.quantize(pl_module, flat_xy)
        n_codebook = len(ratebook)  # uses ratebook because codebook can be None
        counts = np.bincount(idcs, minlength=n_codebook)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        plot_density(source, n_pts=self.n_pts, range_lim=self.range_lim, ax=ax)
        xy = to_numpy(xy)

        google_pink = (0xF4 / 255, 0x39 / 255, 0xA0 / 255)

        # contour lines
        ax.contour(
            xy[:, :, 0],
            xy[:, :, 1],
            idcs.reshape(self.n_pts, self.n_pts),
            np.arange(n_codebook) + 0.5,
            colors=[google_pink],
            linewidths=0.5,
        )

        if self.is_plot_codebook:
            p_codebook = BASE_LOG ** -(ratebook)

            # codebook
            ax.scatter(
                codebook[counts > 0, 0],
                codebook[counts > 0, 1],
                color=google_pink,
                s=500 * p_codebook[counts > 0],  # size prop. to proba q(z)
            )

        return fig


class MaxinvDistributionPlot(PlottingCallback):
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

    kwargs :
        Additional arguments to PlottingCallback.
    """

    def __init__(
        self,
        quantile_lim=5,
        n_pts=500 ** 2,
        figsize=(9, 9),
        equivalences=["rotation", "y_translation", "x_translation"],
        plot_interval=50,
        **kwargs,
    ):
        super().__init__(plot_interval=plot_interval, **kwargs)
        self.quantile_lim = quantile_lim
        self.n_pts = n_pts
        self.figsize = figsize
        self.equivalences = equivalences

    def yield_figs_kwargs(self, trainer, pl_module):
        dataset = trainer.datamodule.train_dataset
        # ensure don't change
        seed, device, equiv = dataset.seed, pl_module.device, dataset.equivalence
        pl_module.seed = None  # generate new data

        # generate new data
        x, _ = dataset.get_n_data_Mxs(self.n_pts)

        with torch.no_grad():
            pl_module.eval()

            # shape: [batch_size, *x_shape]
            x_hat = pl_module(x, is_features=False)

            # allow computing plots for multiple equivalences (useful for banana without equivalence)
            equivalences = [equiv] if equiv is not None else self.equivalences
            for eq in equivalences:
                dataset.equivalence = eq

                # shape: [batch_size, *mx_shape]
                mx_hat = dataset.max_invariant(x_hat) if x_hat.shape[-1] == 2 else x_hat
                mx = dataset.max_invariant(x)

                with plot_config(**self.plot_config_kwargs):
                    fig = self.plot_maxinv(mx, mx_hat)

                yield fig, dict(name=f"max. inv. {eq}")

        # restore
        dataset.seed = seed
        dataset.equivalence = equiv
        pl_module.train()

    def prepare(data, source):
        return data

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


class Freezer(BaseFinetuning):
    """Freeze entire model.

    Parameters
    ----------
    model_name : string
        Name of the module to freeze from pl module. Can use dots.
    """

    def __init__(
        self, model_name,
    ):
        super().__init__()
        self.model_name = model_name.split(".")

    def get_model(self, pl_module):
        model = pl_module

        for model_name in self.model_name:
            model = getattr(model, model_name)

        return model

    def freeze_before_training(self, pl_module):
        model = self.get_model(pl_module)
        self.freeze(modules=model, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass


class ResnetFinetuning(BaseFinetuning):
    """Finetuner for a resnet.

    Parameters
    ----------
    model_name : string
        Name of the model from pl module. Can use dots.

    unfreeze_last_layer : int, optional
        Epoch at which to unfreeze last layer.

    unfreeze_last_block : int, optional
        Epoch at which to unfreeze last block.

    unfreeze_penult_block : int, optional
        Epoch at which to unfreeze penultimate block.

    lr_factor_last_layer : int, optional
        Learning rate factor (how much to decrease compared to normal lr) to use for last layer.

    lr_factor_last_block : int, optional
        Learning rate factor (how much to decrease compared to normal lr) to use for last block.

    lr_factor_penult_block : int, optional
        Learning rate factor (how much to decrease compared to normal lr) to use for penultimate layer.

    train_bn : bool, optional
        Whether to unfreeze batchnorm.
    """

    def __init__(
        self,
        model_name,
        unfreeze_last_layer=0,
        unfreeze_last_block=3,
        unfreeze_penult_block=5,
        lr_factor_last_layer=10,
        lr_factor_last_block=100,
        lr_factor_penult_block=1000,
        train_bn=False,
    ):
        super().__init__()
        self.model_name = model_name.split(".")
        self.unfreeze_last_layer = unfreeze_last_layer
        self.unfreeze_last_block = unfreeze_last_block
        self.unfreeze_penult_block = unfreeze_penult_block
        self.lr_factor_last_layer = lr_factor_last_layer
        self.lr_factor_last_block = lr_factor_last_block
        self.lr_factor_penult_block = lr_factor_penult_block
        self.train_bn = train_bn
        self.loaded_epoch = -1

    def get_model(self, pl_module):
        model = pl_module

        for model_name in self.model_name:
            model = getattr(model, model_name)

        return model

    def freeze_before_training(self, pl_module):
        model = self.get_model(pl_module)
        self.freeze(modules=model, train_bn=self.train_bn)

    def is_unfreeze(self, curr_epoch, target_epoch):
        #! waiting for https://github.com/PyTorchLightning/pytorch-lightning/issues/6891
        return (curr_epoch == target_epoch) or (self.loaded_epoch >= target_epoch)

    def finetune_function(self, pl_module, curr_epoch, optimizer, opt_idx):

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_last_layer):
            resnet = self.get_model(pl_module)

            if hasattr(resnet, "attnpool"):
                last_layer = resnet.attnpool
            elif hasattr(resnet, "fc"):
                last_layer = resnet.fc
            else:
                raise ValueError(
                    f"Unkown resnet architecture {resnet} which has no attnpool nor fc."
                )

            self.unfreeze_and_add_param_group(
                modules=last_layer,
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=self.lr_factor_last_layer,
            )

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_last_block):
            resnet = self.get_model(pl_module)
            last_block = resnet.layer4
            self.unfreeze_and_add_param_group(
                modules=last_block,
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=self.lr_factor_last_block,
            )

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_penult_block):
            resnet = self.get_model(pl_module)
            penult_block = resnet.layer3
            self.unfreeze_and_add_param_group(
                modules=penult_block,
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=self.lr_factor_penult_block,
            )


class ViTFinetuning(ResnetFinetuning):
    # finetuning of CLIP visual tranformer
    def finetune_function(self, pl_module, curr_epoch, optimizer, opt_idx):

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_last_layer):

            vit = self.get_model(pl_module)
            last_layer = vit.proj

            # last layer is just a matrix of parameters, not a layer
            last_layer.requires_grad = True
            lr = optimizer.param_groups[0]["lr"] / self.lr_factor_last_layer

            if any(
                torch.equal(p, last_layer)
                for group in optimizer.param_groups
                for p in group["params"]
            ):
                logger.warning("Skipping last_layer freezing as already in optimizer.")
            else:
                optimizer.add_param_group({"params": [last_layer], "lr": lr})

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_last_block):

            vit = self.get_model(pl_module)
            last_block = vit.transformer.resblocks[-1]
            self.unfreeze_and_add_param_group(
                modules=[last_block.attn, last_block.mlp],  # don't add layer norms
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=self.lr_factor_last_block,
            )

        if opt_idx == 0 and self.is_unfreeze(curr_epoch, self.unfreeze_penult_block):

            vit = self.get_model(pl_module)
            penult_block = vit.transformer.resblocks[-2]
            self.unfreeze_and_add_param_group(
                modules=[penult_block.attn, penult_block.mlp],
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=self.lr_factor_penult_block,
            )
