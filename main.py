from functools import partial
import os

import hydra
import logging
import compressai

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data
import lossyless

from lossyless.callbacks import (
    ClfOnlineEvaluator,
    RgrsOnlineEvaluator,
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
    datamodule = instantiate_datamodule(cfg.data)

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
        chckpt_kwargs = cfg.callbacks.compressor_chckpt

        if cfg.predictor.is_online_eval:
            if cfg.data.is_classification:
                callbacks += [ClfOnlineEvaluator(**cfg.callbacks.online_eval)]
            else:
                callbacks += [RgrsOnlineEvaluator(**cfg.callbacks.online_eval)]

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

    for name in cfg.callbacks.additional:
        try:
            callbacks.append(getattr(lossyless.callbacks, name))
        except AttributeError:
            callbacks.append(getattr(pl.callbacks, name))

    callbacks += [ModelCheckpoint(**chckpt_kwargs)]

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


def instantiate_datamodule(cfgd):
    datamodule = get_datamodule(cfgd.dataset)(**cfgd.kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    cfgd.is_classification_aux = datamodule.is_classification_aux
    cfgd.length = len(datamodule.dataset_train)
    cfgd.shape = datamodule.shape
    cfgd.is_classification = datamodule.is_classification
    cfgd.target_shape = datamodule.target_shape
    cfgd.target_aux_shape = datamodule.target_aux_shape

    cfgd.neg_factor = cfgd.length / (2 * cfgd.kwargs.batch_size - 1)
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
