"""Entropy point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""
import logging
from pathlib import Path

import pandas as pd

import compressai
import hydra
import lossyless
import omegaconf
import pl_bolts
import pytorch_lightning as pl
import utils
from lossyless import CompressionModule
from lossyless.callbacks import (WandbCodebookPlot, WandbLatentDimInterpolator,
                                 WandbMaxinvDistributionPlot,
                                 WandbReconstructImages)
from lossyless.distributions import MarginalVamp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from utils.data import get_datamodule
from utils.estimators import estimate_entropies
from utils.helpers import (create_folders, getattr_from_oneof,
                           omegaconf2namespace, replace_keys, set_debug)

logger = logging.getLogger(__name__)
RES_COMPRESS_FILENAME = "results_compression.csv"


@hydra.main(config_name="main", config_path="config")
def main(cfg):
    begin(cfg)

    # DATA
    datamodule = instantiate_datamodule(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    # COMPRESSION
    compression_module = CompressionModule(hparams=cfg)

    trainer = get_trainer(cfg, compression_module, is_compressor=True)

    # some of the module compononets might needs data or trainer for initialization
    initialize_(compression_module, datamodule, trainer)

    logger.info("TRAIN / EVALUATE compression rate.")
    trainer.fit(compression_module, datamodule=datamodule)
    evaluate_compression(trainer, datamodule, cfg)

    # # PREDICTION
    # TODO should reuse the train/test dataset that were already represented as Z in `evaluate_compression`
    # prediction_module = PredictionModule(hparams=cfg, representer=compression_module)
    # logger.info("TRAIN / EVALUATE downstream classification.")
    # trainer.fit(prediction_module, datamodule=datamodule)
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


def initialize_(compression_module, datamodule, trainer):
    """Uses the data module to set some of the model's param."""

    # auto lr finder

    # marginal vampprior
    rate_est = compression_module.rate_estimator
    if hasattr(rate_est, "q_Z") and isinstance(rate_est.q_Z, MarginalVamp):
        # initialize vamprior such that pseudoinputs are some random images
        real_batch_size = datamodule.batch_size
        datamodule.batch_size = rate_est.q_Z.n_pseudo
        dataloader = datamodule.train_dataloader()
        X, _ = iter(dataloader).next()
        rate_est.q_Z.set_pseudoinput_(X)
        datamodule.batch_size = real_batch_size


def get_callbacks(cfg, is_compressor):
    """Return list of callbacks."""
    callbacks = []

    if is_compressor:
        additional_target = cfg.data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]
        if cfg.logger.name == "wandb" and is_reconstruct:
            if cfg.data.mode == "image":
                callbacks += [
                    WandbLatentDimInterpolator(cfg.encoder.z_dim),
                    WandbReconstructImages(),
                ]
            elif cfg.data.mode == "distribution":
                callbacks += [WandbCodebookPlot(), WandbMaxinvDistributionPlot()]

    curr = "compressor" if is_compressor else "predictor"
    ckwargs = cfg.callbacks[f"ModelCheckpoint_{curr}"]
    callbacks += [ModelCheckpoint(**ckwargs)]

    for name in cfg.callbacks.additional:
        cllbck_kwargs = cfg.callbacks.get(name, {})
        modules = [lossyless.callbacks, pl.callbacks, pl_bolts.callbacks]
        Callback = getattr_from_oneof(modules, name)
        callbacks.append(Callback(**cllbck_kwargs))

    return callbacks


def get_logger(cfg, module):
    """Return coorect logger."""
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

    elif cfg.logger.name == False:
        logger = False

    else:
        raise ValueError(f"Unkown logger={cfg.logger.name}.")

    return logger


def get_trainer(cfg, module, is_compressor):
    """Instantiate trainer."""

    # Resume training ?
    curr = "compressor" if is_compressor else "predictor"
    ckwargs = cfg.callbacks[f"ModelCheckpoint_{curr}"]
    last_chckpnt = Path(ckwargs.dirpath) / "last.ckpt"
    if last_chckpnt.exists():
        cfg.trainer.resume_from_checkpoint = last_chckpnt

    trainer = pl.Trainer(
        logger=get_logger(cfg, module),
        callbacks=get_callbacks(cfg, is_compressor),
        checkpoint_callback=True,
        **cfg.trainer,
    )

    return trainer


def evaluate_compression(trainer, datamodule, cfg):
    """
    Evaluate the compression / representation learning by loging all the metrics from the training 
    and test set from the best bodel. Also computes samples estimates of H_Mlz, H_Ylz which will 
    probably be better estimates than the lower bounds used during training.
    """
    # entropy estimation when Z is stochastic will not be good
    if cfg.evaluation.is_est_entropies and cfg.encoder.fam != "deterministic":
        logger.warn("Turning off `is_est_entropies` because stochastic Z.")
        cfg.evaluation.is_est_entropies = False

    # test on test
    test_res = trainer.test()[0]
    if cfg.evaluation.is_est_entropies:
        H_MlZ, H_YlZ, H_Z = estimate_entropies(trainer, datamodule, is_test=True)
        test_res["test/H_MlZ"] = H_MlZ
        test_res["test/H_YlZ"] = H_YlZ
        test_res["test/H_Z"] = H_Z
    log_metrics(trainer, test_res)

    # test on train
    train_res = trainer.test(test_dataloaders=datamodule.train_dataloader())[0]
    train_res = replace_keys(train_res, "test", "testtrain")
    if cfg.evaluation.is_est_entropies:
        # ? this can be slow on all training set, is it necessary ?
        H_MlZ, H_YlZ, H_Z = estimate_entropies(trainer, datamodule, is_test=False)
        train_res["testtrain/H_MlZ"] = H_MlZ
        train_res["testtrain/H_YlZ"] = H_YlZ
        train_res["testtrain/H_Z"] = H_Z
    log_metrics(trainer, train_res)

    # save results
    train_res = replace_keys(train_res, "testtrain/", "")
    test_res = replace_keys(test_res, "test/", "")
    results = pd.DataFrame.from_dict(dict(train=train_res, test=test_res))
    path = Path(cfg.paths.results) / RES_COMPRESS_FILENAME
    results.to_csv(path, header=True, index=True)
    logger.info(f"Logging compressor results to {path}.")


def log_metrics(trainer, metrics):
    try:
        trainer.logger.log_metrics(metrics)
    except:
        pass


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
