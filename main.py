import hydra
import logging
import compressai
import omegaconf

import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import pl_bolts
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
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
RES_COMPRESS_FILENAME = "results_compression.csv"


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
    evaluate_compression(trainer, datamodule, cfg)

    # # PREDICTION
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
    create_folders(
        cfg.paths.base_dir,
        [
            f"{cfg.paths.results}",
            f"{cfg.paths.pretrained}",
            f"{cfg.paths.logs}",
            f"{cfg.paths.chckpnt}",
        ],
    )

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
        if cfg.logger.name == "wandb" and is_reconstruct:
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
    if cfg.logger.name == "csv":
        logger = CSVLogger(**cfg.logger.csv)

    elif cfg.logger.name == "wandb":
        try:
            logger = WandbLogger(**cfg.logger.wandb)
        except Exception:
            cfg.logger.wandb.offline = True
            logger = WandbLogger(**cfg.logger.wandb)

        if cfg.trainer.track_grad_norm == 2:
            # use wandb rather than lightning gradients
            cfg.trainer.track_grad_norm = -1
            logger.watch(
                module.p_ZlX.mapper,
                log="gradients",
                log_freq=cfg.trainer.log_every_n_steps * 10,
            )

    elif cfg.logger.name == "tensorboard":
        logger = TensorBoardLogger(**cfg.logger.tensorboard)

    else:
        raise ValueError(f"Unkown logger={cfg.logger.name}.")

    # TRAINER
    last_chckpnt = Path(chckpt_kwargs.dirpath) / "last.ckpt"
    if last_chckpnt.exists():
        # resume training
        cfg.trainer.resume_from_checkpoint = last_chckpnt

    trainer = pl.Trainer(
        logger=logger, checkpoint_callback=True, callbacks=callbacks, **cfg.trainer
    )

    return trainer


def evaluate_compression(trainer, datamodule, cfg):
    """Evaluate the compression / representation learning."""
    # the following will load the best model before eval
    # test on test
    test_results = trainer.test()
    trainer.logger.log_metrics(test_results[0])

    # test on train
    train_results = trainer.test(test_dataloaders=datamodule.train_dataloader())
    train_results = {
        k.replace("test", "testtrain"): v for k, v in train_results[0].items()
    }
    trainer.logger.log_metrics(train_results)

    train_results = {k.replace("testtrain_", ""): v for k, v in train_results.items()}
    test_results = {k.replace("test_", ""): v for k, v in test_results[0].items()}
    results = pd.DataFrame.from_dict(dict(train=train_results, test=test_results))
    path = Path(cfg.paths.results) / RES_COMPRESS_FILENAME
    results.to_csv(path, header=True, index=True)


def finalize(cfg, trainer, compression_module):
    """Finalizes the script."""
    # logging.shutdown()

    if cfg.logger.name == "wandb":
        import wandb

        if wandb.run is not None:
            wandb.run.finish()  # finish the run if still on

    # send best checkpoint(s) to main directory
    dest_path = Path(cfg.paths.pretrained)
    dest_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(dest_path / "best.ckpt", weights_only=True)


if __name__ == "__main__":
    main()
