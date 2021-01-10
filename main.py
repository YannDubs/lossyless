from functools import partial
import os

import hydra
import logging
import compressai
import omegaconf

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data
import lossyless

from lossyless.callbacks import (
    OnlineEvaluator,
    WandbReconstructImages,
    WandbLatentDimInterpolator,
)
from lossyless import CompressionModule
from lossyless.distributions import MarginalVamp
from utils.helpers import create_folders, omegaconf2namespace
from utils.data import get_datamodule


logger = logging.getLogger(__name__)

from omegaconf import OmegaConf


@hydra.main(config_name="config", config_path="config")
def main(cfg):
    if cfg.is_debug:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    cfg.paths.base_dir = hydra.utils.get_original_cwd()
    create_folders(cfg.paths.base_dir, ["results", "logs", "pretrained"])

    if cfg.rate.range_coder is not None:
        compressai.set_entropy_coder(cfg.rate.range_coder)

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
    # prediciton_module = PredictionModule(hparams=cfg, representer=compression_module)

    # logger.info("TRAIN / EVALUATE downstream classification.")
    # trainer.fit(prediciton_module, datamodule=datamodule)
    # evaluate_prediction(trainer, datamodule, cfg)


def get_trainer(cfg, module, is_compressor):

    callbacks = []

    if is_compressor:
        chckpt_kwargs = cfg.callbacks.ModelCheckpoint_compressor

        if cfg.predictor.is_online_eval:
            callbacks += [OnlineEvaluator(**cfg.callbacks.OnlineEvaluator)]

        de = module.distortion_estimator
        is_img_out = hasattr(de, "is_img_out") and de.is_img_out
        if is_img_out:
            if "wandb" in cfg.logger.loggers:
                callbacks += [
                    WandbLatentDimInterpolator(cfg.encoder.z_dim),
                    WandbReconstructImages(),
                ]
    else:
        chckpt_kwargs = cfg.callbacks.predictor_chckpt

    callbacks += [ModelCheckpoint(**chckpt_kwargs)]

    for name in cfg.callbacks.additional:
        cllbck_kwargs = cfg.callbacks.additional.get(name, {})
        try:
            callbacks.append(getattr(lossyless.callbacks, name)(**cllbck_kwargs))
        except AttributeError:
            callbacks.append(getattr(pl.callbacks, name)(**cllbck_kwargs))

    loggers = []

    if "csv" in cfg.logger.loggers:
        loggers.append(CSVLogger(**cfg.logger.csv))

    if "wandb" in cfg.logger.loggers:
        try:
            loggers.append(WandbLogger(**cfg.logger.wandb))
        except Exception:
            cfg.logger.wandb.offline = True
            loggers.append(WandbLogger(**cfg.logger.wandb))

        if cfg.is_debug:
            loggers[-1].watch(module.p_ZlX, log="gradients", log_freq=500)

    trainer = pl.Trainer(
        logger=loggers, checkpoint_callback=True, callbacks=callbacks, **cfg.trainer
    )

    return trainer


def instantiate_datamodule(cfg):
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

    cfgd.neg_factor = cfgd.length / (2 * cfgd.kwargs.batch_size - 1)

    # TODO clean max_var for multi label multi clf
    # save real shape of `max_var` if you had to flatten it for batching.
    with omegaconf.open_dict(cfg):
        if hasattr(datamodule, "shape_max_var"):
            cfg.distortion.kwargs.n_classes_multilabel = datamodule.shape_max_var

    return datamodule


def initialize_(compression_module, datamodule):
    rate_est = compression_module.rate_estimator
    if hasattr(rate_est, "q_Z") and isinstance(rate_est.q_Z, MarginalVamp):
        # initialize vamprior such that pseudoinputs are some random images
        real_batch_size = datamodule.batch_size
        datamodule.batch_size = rate_est.q_Z.n_pseudo
        dataloader = datamodule.train_dataloader()
        X, _ = iter(dataloader).next()
        rate_est.q_Z.set_pseudoinput_(X)
        datamodule.batch_size = real_batch_size


if __name__ == "__main__":
    main()
