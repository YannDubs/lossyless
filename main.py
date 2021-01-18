import hydra
import logging
import compressai
import omegaconf

from pathlib import Path
import pytorch_lightning as pl
import pl_bolts
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import lossyless

from lossyless.callbacks import (
    WandbReconstructImages,
    WandbLatentDimInterpolator,
    WandbCodebookPlot,
)
from lossyless import CompressionModule
from lossyless.distributions import MarginalVamp
from utils.helpers import create_folders, omegaconf2namespace, set_debug
from utils.data import get_datamodule
import utils


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="config")
def main(cfg):
    begin(cfg)

    # DATA
    datamodule = instantiate_datamodule(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    # COMPRESSION
    compression_module = CompressionModule(hparams=cfg)

    # some of the module compononets might needs data for initialization
    initialize_(compression_module, datamodule)

    trainer = get_trainer(cfg, compression_module, is_compressor=True)

    logger.info("TRAIN / EVALUATE compression rate.")
    trainer.fit(compression_module, datamodule=datamodule)
    # evaluate_compression(trainer, datamodule, cfg)

    # # PREDICTION
    # TODO add load checkpoint if not training
    # prediciton_module = PredictionModule(hparams=cfg, representer=compression_module)

    # logger.info("TRAIN / EVALUATE downstream classification.")
    # trainer.fit(prediciton_module, datamodule=datamodule)
    # evaluate_prediction(trainer, datamodule, cfg)

    finalize(cfg, trainer, compression_module)
    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    if cfg.is_debug:
        set_debug(cfg)

    pl.seed_everything(cfg.seed)
    cfg.paths.work = str(Path.cwd())
    create_folders(cfg.paths.base_dir, ["results", "logs", "pretrained"])

    if cfg.rate.range_coder is not None:
        compressai.set_entropy_coder(cfg.rate.range_coder)

    logger.info(f"Running {cfg.long_name} from {cfg.paths.work}.")


def instantiate_datamodule(cfg):
    """Instantiate dataset."""
    cfgd = cfg.data
    datamodule = get_datamodule(cfgd.dataset)(**cfgd.kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    cfgd.aux_is_clf = datamodule.aux_is_clf
    cfgd.length = len(datamodule.train_dataset)
    cfgd.shape = datamodule.shape
    cfgd.target_is_clf = datamodule.target_is_clf
    cfgd.target_shape = datamodule.target_shape
    cfgd.aux_shape = datamodule.aux_shape

    if hasattr(utils.data.distributions, type(datamodule).__name__):
        cfgd.mode = "distribution"
    elif hasattr(utils.data.images, type(datamodule).__name__):
        cfgd.mode = "image"
    elif hasattr(datamodule, "mode"):
        cfgd.mode = datamodule.mode
    else:
        raise ValueError(
            f"Cannot say whether datamodule={type(datamodule)} is distribution or image. Add a `mode` attribute."
        )

    cfgd.neg_factor = cfgd.length / (2 * cfgd.kwargs.batch_size - 1)

    # TODO clean max_var for multi label multi clf
    # save real shape of `max_var` if you had to flatten it for batching.
    with omegaconf.open_dict(cfg):
        if hasattr(datamodule, "shape_max_var"):
            cfg.distortion.kwargs.n_classes_multilabel = datamodule.shape_max_var
            # max_var is such that all tasks are independent are should sum over them
            cfg.distortion.kwargs.is_sum_over_tasks = True

    return datamodule


def initialize_(compression_module, datamodule):
    """Uses the data module to set some of the model's param."""
    rate_est = compression_module.rate_estimator
    if hasattr(rate_est, "q_Z") and isinstance(rate_est.q_Z, MarginalVamp):
        # initialize vamprior such that pseudoinputs are some random images
        real_batch_size = datamodule.batch_size
        datamodule.batch_size = rate_est.q_Z.n_pseudo
        dataloader = datamodule.train_dataloader()
        X, _ = iter(dataloader).next()
        rate_est.q_Z.set_pseudoinput_(X)
        datamodule.batch_size = real_batch_size


def get_trainer(cfg, module, is_compressor):
    """Instantiate trainer."""

    # CALLBACKS
    callbacks = []

    if is_compressor:
        chckpt_kwargs = cfg.callbacks.ModelCheckpoint_compressor

        additional_target = cfg.data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]
        if "wandb" in cfg.logger.loggers and is_reconstruct:
            if cfg.data.mode == "image":
                callbacks += [
                    WandbLatentDimInterpolator(cfg.encoder.z_dim),
                    WandbReconstructImages(),
                ]
            elif cfg.data.mode == "distribution":
                callbacks += [WandbCodebookPlot()]

    else:
        chckpt_kwargs = cfg.callbacks.predictor_chckpt

    callbacks += [ModelCheckpoint(**chckpt_kwargs)]

    for name in cfg.callbacks.additional:
        cllbck_kwargs = cfg.callbacks.get(name, {})
        try:
            callbacks.append(getattr(lossyless.callbacks, name)(**cllbck_kwargs))
        except AttributeError:
            try:
                callbacks.append(getattr(pl.callbacks, name)(**cllbck_kwargs))
            except AttributeError:
                callbacks.append(getattr(pl_bolts.callbacks, name)(**cllbck_kwargs))

    # LOGGERS
    loggers = []

    if "csv" in cfg.logger.loggers:
        loggers.append(CSVLogger(**cfg.logger.csv))

    if "wandb" in cfg.logger.loggers:
        try:
            loggers.append(WandbLogger(**cfg.logger.wandb))
        except Exception:
            cfg.logger.wandb.offline = True
            loggers.append(WandbLogger(**cfg.logger.wandb))

        if cfg.trainer.track_grad_norm == 2:
            # use wandb rather than lightning gradients
            cfg.trainer.track_grad_norm = -1
            loggers[-1].watch(module.p_ZlX.mapper, log="gradients", log_freq=500)

    # TRAINER
    last_chckpnt = Path(chckpt_kwargs.dirpath) / "last.ckpt"
    if last_chckpnt.exists():
        # resume training
        cfg.trainer.resume_from_checkpoint = last_chckpnt

    trainer = pl.Trainer(
        logger=loggers, checkpoint_callback=True, callbacks=callbacks, **cfg.trainer
    )

    return trainer


def finalize(cfg, trainer, compression_module):
    """Finalizes the script."""
    # logging.shutdown()

    if "wandb" in cfg.logger.loggers:
        import wandb

        if wandb.run is not None:
            wandb.run.finish()  # finish the run if still on

    # send latest checkpoint to main directory


if __name__ == "__main__":
    main()
