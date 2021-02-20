"""Entry point to load a pretrained model for inference / plotting.

This should be called by `python load_pretrained.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/load_pretrained.yaml` for details about the configs. or use `python load_pretrained.py -h`.
"""
import logging
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from lossyless.callbacks import (
    CodebookPlot,
    LatentDimInterpolator,
    MaxinvDistributionPlot,
)
from lossyless.helpers import (
    UnNormalizer,
    is_colored_img,
    plot_config,
    tensors_to_fig,
    tmp_seed,
)
from main import main as main_training
from omegaconf import OmegaConf
from utils.helpers import all_logging_disabled
from utils.postplotting import PRETTY_RENAMER, PostPlotter
from utils.postplotting.helpers import save_fig

logger = logging.getLogger(__name__)


@hydra.main(config_name="load_pretrained", config_path="config")
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

    def collect_data(self, cfg, is_only_feat=True, is_force_cpu=True):
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
            cfg.callbacks.additional = []

        if is_only_feat:
            cfg.is_only_feat = True

        with all_logging_disabled(highest_level=logging.INFO):
            self.modules, self.trainers, self.datamodules, self.cfgs = main_training(
                cfg
            )

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
            val = cfg
            for subselect in v.split("."):
                val = val[subselect]
            kwargs[k] = val

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
        self.plot_using_callback(
            CodebookPlot,
            is_featurizer=True,
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

    def reconstruct_image_plot(self, seed=123, n_samples=5, is_train=False):
        """Reconstruct the desired image."""
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

        for i, (xi, xi_hat) in enumerate(zip(x, x_hat)):

            file_path = save_dir / Path(f"{self.prfx}rec_img_{i}.png")

            if is_colored_img(xi):
                if cfg.data.kwargs.dataset_kwargs.is_normalize:
                    unnormalizer = UnNormalizer(cfg.data.dataset)
                    xi = unnormalizer(xi)

            both_images = torch.stack([xi, xi_hat], dim=0)
            fig = tensors_to_fig(
                both_images, n_cols=2, x_labels=["Real", "Reconstruction"]
            )

            save_fig(fig, file_path, dpi=self.dpi)


if __name__ == "__main__":
    main_cli()
