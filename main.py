"""Entropy point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import compressai
import hydra
import lossyless
import pl_bolts
import pytorch_lightning as pl
from lossyless import ClassicalCompressor, LearnableCompressor, Predictor
from lossyless.callbacks import (
    CodebookPlot,
    LatentDimInterpolator,
    MaxinvDistributionPlot,
    ReconstructImages,
)
from lossyless.distributions import MarginalVamp
from lossyless.helpers import orderedset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from utils.data import get_datamodule
from utils.estimators import estimate_entropies
from utils.helpers import (
    get_latest_match,
    getattr_from_oneof,
    learning_rate_finder,
    log_dict,
    omegaconf2namespace,
    replace_keys,
    set_debug,
)

logger = logging.getLogger(__name__)
COMPRESSOR_CKPNT = "best_compressor.ckpt"
PREDICTOR_CKPNT = "best_predictor.ckpt"
COMPRESSOR_RES = "results_compressor.csv"
PREDICTOR_RES = "results_predictor.csv"


@hydra.main(config_name="main", config_path="config")
def main(cfg_hydra):
    ############## STARTUP ##############
    begin(cfg_hydra)

    ############## COMPRESSOR (i.e. sender) ##############
    cfg = set_cfg(cfg_hydra, mode="featurizer")
    datamodule = instantiate_datamodule_(cfg)
    cfg = omegaconf2namespace(cfg)  # ensure real python types (only once cfg are fixed)

    if not cfg.featurizer.is_learnable:
        logger.info(f"Using classical compressor {cfg.featurizer.type} ...")
        compressor = ClassicalCompressor(hparams=cfg)
        comp_trainer = get_trainer(cfg, compressor, is_featurizer=True,)
        placeholder_fit(comp_trainer, compressor, datamodule)

    elif cfg.featurizer.is_train:
        compressor = LearnableCompressor(hparams=cfg)
        comp_trainer = get_trainer(cfg, compressor, is_featurizer=True)
        initialize_compressor_(compressor, datamodule, comp_trainer, cfg)

        logger.info("Train compressor ...")
        comp_trainer.fit(compressor, datamodule=datamodule)
        save_pretrained(cfg, comp_trainer, COMPRESSOR_CKPNT)
    else:
        logger.info("Load pretrained compressor ...")
        compressor = load_pretrained(cfg, LearnableCompressor, COMPRESSOR_CKPNT)
        comp_trainer = get_trainer(cfg, compressor, is_featurizer=True)

    if cfg.evaluation.featurizer.is_evaluate:
        logger.info("Evaluate compressor ...")
        evaluate(
            comp_trainer,
            datamodule,
            cfg,
            COMPRESSOR_RES,
            is_est_entropies(cfg),
            ckpt_path=cfg.evaluation.featurizer.ckpt_path,
        )

    ############## COMMUNICATION (compress and decompress the datamodule) ##############
    if cfg.featurizer.is_on_the_fly:
        # this will perform compression on the fly
        #! one issue is that if using data augmentations you will augment before the featurizer
        # which is less realisitic (normalization is dealt with correctly though)
        onfly_featurizer = compressor
        pre_featurizer = None
    else:
        # compressing once the dataset is more realistic (and quicker) but requires
        # more memory as the compressed dataset will be saved to file
        onfly_featurizer = None
        pre_featurizer = compressor

    ############## DOWNSTREAM PREDICTOR (i.e. receiver) ##############
    cfg = set_cfg(cfg, mode="predictor")
    datamodule = instantiate_datamodule_(cfg, pre_featurizer=pre_featurizer)
    cfg = omegaconf2namespace(cfg)  # ensure real python types (only once cfg are fixed)

    if cfg.predictor.is_train:
        predictor = Predictor(hparams=cfg, featurizer=onfly_featurizer)
        pred_trainer = get_trainer(cfg, predictor, is_featurizer=False)
        initialize_predictor_(predictor, datamodule, pred_trainer, cfg)

        logger.info("Train predictor ...")
        pred_trainer.fit(predictor, datamodule=datamodule)
        save_pretrained(cfg, pred_trainer, PREDICTOR_CKPNT)

    else:
        logger.info("Load pretrained predictor ...")
        predictor = load_pretrained(cfg, Predictor, PREDICTOR_CKPNT)
        pred_trainer = get_trainer(cfg, predictor, is_featurizer=True)

    if cfg.evaluation.predictor.is_evaluate:
        logger.info("Evaluate predictor ...")
        evaluate(
            pred_trainer,
            datamodule,
            cfg,
            PREDICTOR_RES,
            False,
            ckpt_path=cfg.evaluation.predictor.ckpt_path,
        )

    ############## SHUTDOWN ##############
    finalize(
        cfg, modules=[compressor, predictor], trainers=[comp_trainer, pred_trainer]
    )
    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    if cfg.other.is_debug:
        set_debug(cfg)

    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())

    if cfg.rate.range_coder is not None:
        compressai.set_entropy_coder(cfg.rate.range_coder)

    logger.info(f"Running {cfg.long_name} from {cfg.paths.work}.")


def set_cfg(cfg, mode):
    """Set the configurations for a specific mode."""
    cfg = copy.deepcopy(cfg)  # not inplace

    if mode == "featurizer":
        cfg.stage = "feat"
        cfg.long_name = cfg.long_name_feat

        cfg.data.update(cfg.datafeat)
        cfg.trainer.update(cfg.update_trainer_feat)
        cfg.checkpoint.update(cfg.checkpoint_feat)

    elif mode == "predictor":
        cfg.stage = "pred"
        cfg.long_name = cfg.long_name_ored

        cfg.data.update(cfg.datapred)
        cfg.trainer.update(cfg.update_trainer_pred)
        cfg.checkpoint.update(cfg.checkpoint_pred)

        # only need target
        cfg.data.kwargs.dataset_kwargs.additional_target = None

    else:
        raise ValueError(f"Unkown mode={mode}.")

    # make sure all paths exist
    for _, path in cfg.paths.items():
        if isinstance(path, str):
            Path(path).mkdir(parents=True, exist_ok=True)

    Path(cfg.paths.pretrained.save).mkdir(parents=True, exist_ok=True)

    return cfg


def instantiate_datamodule_(cfg, pre_featurizer=None):
    """Instantiate dataset."""

    if pre_featurizer is not None:
        pass  # TODO (probabby give to datamodule)

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
    cfgd.mode = datamodule.mode
    cfgd.neg_factor = cfgd.length / (2 * cfgd.kwargs.batch_size - 1)

    return datamodule


def initialize_compressor_(module, datamodule, trainer, cfg):
    """Additional steps needed for intitalization of the compressor + logging."""

    # marginal vampprior
    rate_est = module.rate_estimator
    if hasattr(rate_est, "q_Z") and isinstance(rate_est.q_Z, MarginalVamp):
        # initialize vamprior such that pseudoinputs are some random images
        real_batch_size = datamodule.batch_size
        datamodule.batch_size = rate_est.q_Z.n_pseudo
        dataloader = datamodule.train_dataloader()
        X, _ = iter(dataloader).next()
        rate_est.q_Z.set_pseudoinput_(X)
        datamodule.batch_size = real_batch_size

    # LOGGING
    # save number of parameters for the main model (not online optimizer but with coder)
    aux_parameters = orderedset(module.rate_estimator.aux_parameters())
    n_param = sum(p.numel() for p in aux_parameters if p.requires_grad)
    n_param += sum(p.numel() for p in module.parameters() if p.requires_grad)
    log_dict(trainer, {"n_param": n_param}, is_param=True)

    # estimate interesting entropies
    entropies = datamodule.dataset.entropies
    entropies = {f"data/{k}": v for k, v in entropies.items()}
    log_dict(trainer, entropies, is_param=True)


def get_callbacks(cfg, is_featurizer):
    """Return list of callbacks."""
    callbacks = []

    if is_featurizer:
        additional_target = cfg.data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]
        can_estimate_Mx = ["representative", "input", "max_var", "max_inv"]
        if cfg.logger.is_can_plot_img:
            if cfg.data.mode == "image" and is_reconstruct:
                callbacks += [
                    LatentDimInterpolator(cfg.encoder.z_dim),
                    ReconstructImages(),
                ]
            elif cfg.data.mode == "distribution":
                callbacks += [
                    CodebookPlot(is_plot_codebook=is_reconstruct),
                ]
                if additional_target in can_estimate_Mx:
                    callbacks += [
                        MaxinvDistributionPlot(),
                    ]

    callbacks += [ModelCheckpoint(**cfg.checkpoint_feat.kwargs)]

    for name in cfg.callbacks.additional:
        cllbck_kwargs = cfg.callbacks.get(name, {})
        modules = [lossyless.callbacks, pl.callbacks, pl_bolts.callbacks]
        Callback = getattr_from_oneof(modules, name)
        callbacks.append(Callback(**cllbck_kwargs))

    return callbacks


def get_logger(cfg, module):
    """Return coorect logger."""
    if cfg.logger.name == "csv":
        logger = CSVLogger(**cfg.logger.kwargs)

    elif cfg.logger.name == "wandb":
        try:
            logger = WandbLogger(**cfg.logger.kwargs)
        except Exception:
            cfg.logger.wandb.offline = True
            logger = WandbLogger(**cfg.logger.kwargs)

        if cfg.trainer.track_grad_norm == 2:
            # use wandb rather than lightning gradients
            cfg.trainer.track_grad_norm = -1
            logger.watch(
                module.p_ZlX.mapper,
                log="gradients",
                log_freq=cfg.trainer.log_every_n_steps * 10,
            )

    elif cfg.logger.name == "tensorboard":
        logger = TensorBoardLogger(**cfg.logger.kwargs)

    elif cfg.logger.name is None:
        logger = False

    else:
        raise ValueError(f"Unkown logger={cfg.logger.name}.")

    return logger


def get_trainer(cfg, module, is_featurizer):
    """Instantiate trainer."""
    # if is_placeholder:
    #     return pl.Trainer()

    # Resume training ?
    last_chckpnt = Path(cfg.callbacks.ModelCheckpoint.dirpath) / "last.ckpt"
    if last_chckpnt.exists():
        cfg.trainer.resume_from_checkpoint = str(last_chckpnt)

    trainer = pl.Trainer(
        logger=get_logger(cfg, module),
        callbacks=get_callbacks(cfg, is_featurizer),
        checkpoint_callback=True,
        **cfg.trainer,
    )

    return trainer


def placeholder_fit(trainer, module, datamodule):
    """Necessary setup of trainer before testing if you don't fit it."""
    trainer.train_loop.setup_fit(module, None, None, datamodule)
    trainer.model = module


def save_pretrained(cfg, trainer, file):
    """Send best checkpoint for compressor to main directory."""
    dest_path = Path(cfg.paths.pretrained.save)
    dest_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(dest_path / file, weights_only=True)


def load_pretrained(cfg, Module, file):
    """Load the best checkpoint from the latest run that has the same name as current run."""
    save_path = Path(cfg.paths.pretrained.load)
    # select the latest checkpoint matching the path
    chckpnt = get_latest_match(save_path / file)

    loaded_module = Module.load_from_checkpoint(chckpnt)

    return loaded_module


def is_est_entropies(cfg):
    # entropy estimation when Z is stochastic will not be good
    if cfg.evaluation.is_est_entropies and cfg.encoder.fam != "deterministic":
        logger.warn("Turning off `is_est_entropies` because stochastic Z.")
        return False
    return cfg.evaluation.is_est_entropies


def evaluate(trainer, datamodule, cfg, file, is_est_entropies=False, ckpt_path="best"):
    """
    Evaluate the trainer by loging all the metrics from the training and test set from the best model. 
    Can also compute sample estimates of soem entropies, which should be better estimates than the 
    lower bounds used during training.
    """
    # test on test
    test_res = trainer.test(ckpt_path=ckpt_path)[0]
    if is_est_entropies:
        append_entropy_est_(test_res, trainer, datamodule, cfg, is_test=True)
    log_dict(trainer, test_res, is_param=False)

    # test on train
    train_res = trainer.test(
        test_dataloaders=datamodule.train_dataloader(), ckpt_path=ckpt_path
    )[0]
    train_res = replace_keys(train_res, "test", "testtrain")
    if cfg.evaluation.is_est_entropies:
        # ? this can be slow on all training set, is it necessary ?
        append_entropy_est_(test_res, trainer, datamodule, cfg, is_test=False)
    log_dict(trainer, train_res, is_param=False)

    # save results
    train_res = replace_keys(train_res, "testtrain/", "")
    test_res = replace_keys(test_res, "test/", "")
    results = pd.DataFrame.from_dict(dict(train=train_res, test=test_res))
    path = Path(cfg.paths.results) / file
    results.to_csv(path, header=True, index=True)
    logger.info(f"Logging results to {path}.")


def append_entropy_est_(results, trainer, datamodule, cfg, is_test):
    """Append entropy estimates to the results."""
    is_discrete_Y = cfg.data.target_is_clf
    is_discrete_M = datamodule.dataset.is_clf_x_t_Mx["max_inv"]
    H_MlZ, H_YlZ, H_Z = estimate_entropies(
        trainer,
        datamodule,
        is_test=is_test,
        is_discrete_M=is_discrete_M,
        is_discrete_Y=is_discrete_Y,
    )
    prfx = "test" if is_test else "testtrain"
    results[f"{prfx}/H_MlZ"] = H_MlZ
    results[f"{prfx}/H_YlZ"] = H_YlZ
    results[f"{prfx}/H_Z"] = H_Z


def initialize_predictor_(module, datamodule, trainer, cfg):
    """Additional steps needed for intitalization of the predictor + logging."""
    if module.hparams.optimizer_predictor.is_lr_find:
        old_lr = module.hparams.optimizer_predictor.lr
        new_lr = learning_rate_finder(module, datamodule, trainer)
        if old_lr is None:
            module.hparams.optimizer_predictor.lr = new_lr
            logger.info(f"Using lr={new_lr} for the predictor.")
        else:
            module.hparams.optimizer_predictor.lr = old_lr


def finalize(cfg, modules, trainers):
    """Finalizes the script."""

    logging.shutdown()

    plt.close("all")

    if cfg.logger.name == "wandb":
        import wandb

        if wandb.run is not None:
            wandb.run.finish()  # finish the run if still on


if __name__ == "__main__":
    main()
