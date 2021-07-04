"""Entry point to load a pretrained model for inference / plotting.

This should be called by `python utils/load_pretrained.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/load_pretrained.yaml` for details about the configs. or use `python utils/load_pretrained.py -h`.
"""
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

import einops
import hydra
from omegaconf import OmegaConf

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from lossyless.helpers import (  # isort:skip
    UnNormalizer,
    is_colored_img,
    plot_config,
    tensors_to_fig,
    tmp_seed,
)
from main import main as main_training  # isort:skip
from utils.helpers import all_logging_disabled, format_resolver  # isort:skip
from utils.postplotting import PRETTY_RENAMER, PostPlotter  # isort:skip
from utils.postplotting.helpers import save_fig  # isort:skip
from lossyless.callbacks import (  # isort:skip
    CodebookPlot,
    LatentDimInterpolator,
    MaxinvDistributionPlot,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{MAIN_DIR}/config", config_name="load_pretrained")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    try:
        return main(cfg)
    except:
        logger.exception("Failed to load pretrained with this error:")
        # don't raise error because if multirun want the rest to work
        pass


def main(cfg):

    begin(cfg)

    analyser = PretrainedAnalyser(**cfg.load_pretrained.kwargs)

    logger.info(f"Collecting the data ..")
    analyser.collect_data(cfg, **cfg.load_pretrained.collect_data)

    for f in cfg.load_pretrained.mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in cfg.load_pretrained:
            kwargs = cfg.load_pretrained[f]
        else:
            kwargs = {}

        getattr(analyser, f)(**kwargs)

    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    OmegaConf.set_struct(cfg, False)  # allow pop
    PRETTY_RENAMER.update(cfg.load_pretrained.kwargs.pop("pretty_renamer"))
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Analysing pretrained models for {cfg.experiment} ...")


class PretrainedAnalyser(PostPlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modules = dict()
        self.trainers = dict()
        self.datamodules = dict()
        self.results = dict()

    def collect_data(self, cfg, is_only_feat=True, is_force_cpu=False):
        """Collects all the data.

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            All configs.
        """
        cfg = deepcopy(cfg)

        cfg.is_return = True
        cfg.is_no_save = True

        # remove unessecary work
        cfg.evaluation.featurizer.is_evaluate = False
        cfg.evaluation.predictor.is_evaluate = False
        cfg.predictor.is_train = False
        cfg.featurizer.is_train = False
        cfg.logger.name = None

        if is_force_cpu:
            # make sure can run on cpu
            cfg.trainer.gpus = 0
            cfg.trainer.precision = 32
            cfg.callbacks.is_force_no_additional_callback = True

        if is_only_feat:
            cfg.is_only_feat = True

        with all_logging_disabled(highest_level=logging.INFO):
            try:
                (
                    self.modules,
                    self.trainers,
                    self.datamodules,
                    self.cfgs,
                    self.results,
                ) = main_training(cfg)
            except Exception as e:
                raise e

    def plot_using_callback(
        self,
        Callback,
        is_featurizer=True,
        plot_config_kwargs={},
        kwargs_from_cfg={},
        **kwargs,
    ):
        """Plot using a callback plotter."""

        mode = "featurizer" if is_featurizer else "predictor"

        module = self.modules[mode]
        trainer = self.trainers[mode]
        trainer.datamodule = self.datamodules[mode]
        cfg = self.cfgs[mode]

        used_plot_config = dict(self.plot_config_kwargs, **plot_config_kwargs)

        for k, v in kwargs_from_cfg.items():
            kwargs[k] = cfg.select(v)

        plotter = Callback(plot_config_kwargs=used_plot_config, **kwargs)

        for fig, kwargs in plotter.yield_figs_kwargs(trainer, module):
            # has to compute save dir on the fly because you want it to depend on result path

            save_dir = Path(cfg.load_pretrained.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            file_path = save_dir / Path(f"{self.prfx}{kwargs['name']}.png")
            save_fig(fig, file_path, dpi=self.dpi)

    def maxinv_distribution_plot(
        self, plot_config_kwargs={"is_rm_xticks": True, "is_rm_yticks": True}, **kwargs
    ):
        """
        Plot the distribtion of a maximal invariant p(M(X)) as well as the learned marginal
        q(M(X)) = E_{p(Z)}[q(M(X)|Z)].

        Parameters
        ----------
        kwargs :
            Additional arguments to lossyless.callbacks.MaxinvDistributionPlot.
        """
        self.plot_using_callback(
            MaxinvDistributionPlot,
            is_featurizer=True,
            plot_config_kwargs=plot_config_kwargs,
            **kwargs,
        )

    def codebook_plot(self, plot_config_kwargs={"is_ax_off": True}, **kwargs):
        """Plot the source distribution and codebook for a distribution.

        Parameters
        ----------
        kwargs :
            Additional arguments to lossyless.callbacks.CodebookPlot.
        """
        additional_target = self.cfgs[
            "featurizer"
        ].data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]
        self.plot_using_callback(
            CodebookPlot,
            is_featurizer=True,
            is_plot_codebook=is_reconstruct,
            plot_config_kwargs=plot_config_kwargs,
            **kwargs,
        )

    def latent_traversals_plot(self, **kwargs):
        """Logs interpolated images.

        Parameters
        ----------
        plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...

        kwargs :
            Additional arguments to lossyless.callbacks.LatentDimInterpolator.
        """
        kwargs_from_cfg = dict(z_dim="encoder.z_dim")
        self.plot_using_callback(
            LatentDimInterpolator,
            is_featurizer=True,
            kwargs_from_cfg=kwargs_from_cfg,
            **kwargs,
        )

    def reconstruct_image_plot(
        self,
        seed=123,
        n_samples=7,
        is_train=False,
        is_plot_real=True,
        n_rows=None,
        x_labels=None,
        y_labels=None,
        filename="rec_imgs.png",
    ):
        """Reconstruct the desired image.
        
        Parameters
        ----------
        seed : int, optional
            Random seed

        n_samples : int, optional
            Number of images to sample.

        is_train : bool, optional
            Whether to show training images rather than test.

        is_plot_real : bool, optional
            Wehter to plot real image in addition to reconstruction.

        n_rows : int, optional
            Number of rows to use. Usually automatic.

        x_labels : str, optional
            Labels for x axis. `None` means automatic.

        y_labels : str, optional
            Labels for y axis. `None` means automatic.

        filename : str, optional
            Where to save the image.
        """
        mode = "featurizer"
        module = self.modules[mode]
        datamodule = self.datamodules[mode]
        cfg = self.cfgs[mode]

        additional_target = cfg.data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]

        assert is_reconstruct

        if is_train:
            dataloader = datamodule.train_dataloader(batch_size=n_samples)
        else:
            dataloader = datamodule.eval_dataloader(
                cfg.evaluation.is_eval_on_test, batch_size=n_samples
            )

        with tmp_seed(seed):
            for batch in dataloader:
                x, _ = batch
                break

        x_hat = module(x, is_features=False)

        save_dir = Path(cfg.load_pretrained.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_colored_img(x):
            if cfg.data.kwargs.dataset_kwargs.is_normalize:
                unnormalizer = UnNormalizer(cfg.data.dataset)
                x = unnormalizer(x)

        file_path = save_dir / Path(f"{self.prfx}{filename}")

        if is_plot_real:
            if y_labels is None:
                y_labels = ["Source", "Reconstructions"]

            if x_labels is None:
                x_labels = ""

            all_images = torch.stack([x, x_hat], dim=0)
            all_images = einops.rearrange(all_images, "mode b ... -> (b mode) ...")
            fig = tensors_to_fig(
                all_images, n_rows=2, y_labels=y_labels, x_labels=x_labels
            )
        else:
            if y_labels is None:
                y_labels = ""

            if x_labels is None:
                x_labels = ["Reconstructions"]

            if n_rows is None:
                n_rows = 1
            fig = tensors_to_fig(
                x_hat, n_rows=n_rows, x_labels=x_labels, y_labels=y_labels
            )

        save_fig(fig, file_path, dpi=self.dpi)

    def reconstruct_image_plot_placeholder(
        self, seed=123, is_single_row=True, add_standard="", add_invariant=""
    ):
        """Placeholder figure so that can copy past other results.
        
        Parameters
        ----------
        seed : int, optional
            Random seed.

        is_single_row : bool, optional
            Whether to have a single row with 3 images. Else has rows each with 7 columns.

        add_standard : str, optional
            Additional name to add to "Standard Compression""

        add_invariant : str, optional
            Additional name to add to "Invariant Compression"
        """
        labels = [
            "Source",
            "Standard Compression" + add_standard,
            "Invariant Compression" + add_invariant,
        ]

        if is_single_row:
            self.reconstruct_image_plot(
                seed=seed,
                n_samples=3,
                is_plot_real=False,
                y_labels="",
                x_labels=labels,
                filename="rec_imgs_allin1_singlerow.png",
            )
        else:
            self.reconstruct_image_plot(
                seed=seed,
                n_samples=7 * 3,
                n_rows=3,
                is_plot_real=False,
                y_labels=labels,
                x_labels="",
                filename="rec_imgs_allin1_multirow.png",
            )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main_cli()
