"""Entropy point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""
import copy
import logging
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import compressai
import hydra
import lossyless
import omegaconf
import pl_bolts
import pytorch_lightning as pl
import torch
from lossyless import ClassicalCompressor, LearnableCompressor, Predictor
from lossyless.callbacks import (
    CodebookPlot,
    LatentDimInterpolator,
    MaxinvDistributionPlot,
    ReconstructImages,
)
from lossyless.distributions import MarginalVamp
from lossyless.helpers import check_import
from lossyless.predictors import get_featurizer_predictor
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import (
    DDPPlugin,
    DDPShardedPlugin,
    DDPSpawnPlugin,
    DDPSpawnShardedPlugin,
)
from utils.data import get_datamodule
from utils.estimators import estimate_entropies
from utils.helpers import (
    DataParallelPlugin,
    ModelCheckpoint,
    apply_featurizer,
    cfg_save,
    format_resolver,
    get_latest_match,
    getattr_from_oneof,
    learning_rate_finder,
    log_dict,
    omegaconf2namespace,
    replace_keys,
    set_debug,
)

try:
    import wandb
except ImportError:
    pass


logger = logging.getLogger(__name__)
COMPRESSOR_CHCKPNT = "best_compressor.ckpt"
PREDICTOR_CHCKPNT = "best_predictor.ckpt"
LAST_CHCKPNT = "last.ckpt"
COMPRESSOR_RES = "results_compressor.csv"
PREDICTOR_RES = "results_predictor.csv"
FILE_END = "end.txt"
CONFIG_FILE = "config.yaml"

try:
    GIT_HASH = (
        subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        .decode("utf-8")
        .strip()
    )
except:
    logger.exception("Failed to save git hash with error:")
    GIT_HASH = None


@hydra.main(config_name="main", config_path="config")
def main(cfg):
    ############## STARTUP ##############
    logger.info("Stage : Startup")
    begin(cfg)

    ############## COMPRESSOR (i.e. sender) ##############
    logger.info("Stage : Compressor")
    comp_cfg = set_cfg(cfg, mode="featurizer")
    comp_datamodule = instantiate_datamodule_(comp_cfg)
    comp_cfg = omegaconf2namespace(comp_cfg)  # ensure real python types

    if not comp_cfg.featurizer.is_learnable:
        logger.info(f"Using classical compressor {comp_cfg.featurizer.mode} ...")
        compressor = ClassicalCompressor(hparams=comp_cfg)
        comp_trainer = get_trainer(comp_cfg, compressor, is_featurizer=True,)
        placeholder_fit(comp_trainer, compressor, comp_datamodule)

    elif comp_cfg.featurizer.is_train and not is_trained(comp_cfg, COMPRESSOR_CHCKPNT):
        compressor = LearnableCompressor(hparams=comp_cfg)
        comp_trainer = get_trainer(comp_cfg, compressor, is_featurizer=True)
        initialize_compressor_(compressor, comp_datamodule, comp_trainer, comp_cfg)

        logger.info("Train compressor ...")
        comp_trainer.fit(compressor, datamodule=comp_datamodule)
        save_pretrained(comp_cfg, comp_trainer, COMPRESSOR_CHCKPNT)
    else:
        logger.info("Load pretrained compressor ...")
        compressor = load_pretrained(comp_cfg, LearnableCompressor, COMPRESSOR_CHCKPNT)
        comp_trainer = get_trainer(comp_cfg, compressor, is_featurizer=True)
        placeholder_fit(comp_trainer, compressor, comp_datamodule)
        comp_cfg.evaluation.featurizer.ckpt_path = None  # eval loaded model

    if comp_cfg.evaluation.featurizer.is_evaluate:
        logger.info("Evaluate compressor ...")
        feat_res = evaluate(
            comp_trainer,
            comp_datamodule,
            comp_cfg,
            COMPRESSOR_RES,
            is_est_entropies(comp_cfg),
            ckpt_path=comp_cfg.evaluation.featurizer.ckpt_path,
            is_featurizer=True,
        )
    else:
        feat_res = dict()

    finalize_stage(comp_cfg, compressor, comp_trainer)
    if comp_cfg.is_only_feat:
        return finalize(
            modules=dict(featurizer=compressor),
            trainers=dict(featurizer=comp_trainer),
            datamodules=dict(featurizer=comp_datamodule),
            cfgs=dict(featurizer=comp_cfg),
            results=dict(featurizer=feat_res),
        )
    if not comp_cfg.is_return:
        comp_datamodule = None  # not used anymore and can be large

    ############## COMMUNICATION (compress and decompress the datamodule) ##############
    logger.info("Stage : Communication")
    if comp_cfg.featurizer.is_on_the_fly:
        # this will perform compression on the fly
        #! one issue is that if using data augmentations you will augment before the featurizer
        # which is less realisitic (normalization is dealt with correctly though)
        onfly_featurizer = compressor
        pre_featurizer = None
    else:
        # compressing once the dataset is more realistic (and quicker) but requires more RAM
        onfly_featurizer = None
        pre_featurizer = comp_trainer

    ############## DOWNSTREAM PREDICTOR (i.e. receiver) ##############
    logger.info("Stage : Predictor")
    pred_cfg = set_cfg(cfg, mode="predictor")
    pred_datamodule = instantiate_datamodule_(pred_cfg, pre_featurizer=pre_featurizer)
    pred_cfg = omegaconf2namespace(
        pred_cfg
    )  # ensure real python types (only once cfg are fixed)

    if pred_cfg.predictor.is_train and not is_trained(comp_cfg, PREDICTOR_CHCKPNT):
        predictor = Predictor(hparams=pred_cfg, featurizer=onfly_featurizer)
        pred_trainer = get_trainer(pred_cfg, predictor, is_featurizer=False)
        initialize_predictor_(predictor, pred_datamodule, pred_trainer, pred_cfg)

        logger.info("Train predictor ...")
        pred_trainer.fit(predictor, datamodule=pred_datamodule)
        save_pretrained(pred_cfg, pred_trainer, PREDICTOR_CHCKPNT)

    else:
        logger.info("Load pretrained predictor ...")
        FeatPred = get_featurizer_predictor(onfly_featurizer)
        predictor = load_pretrained(pred_cfg, FeatPred, PREDICTOR_CHCKPNT)
        pred_trainer = get_trainer(pred_cfg, predictor, is_featurizer=False)
        placeholder_fit(pred_trainer, predictor, pred_datamodule)
        pred_cfg.evaluation.predictor.ckpt_path = None  # eval loaded model

    if pred_cfg.evaluation.predictor.is_evaluate:
        logger.info("Evaluate predictor ...")
        pred_res = evaluate(
            pred_trainer,
            pred_datamodule,
            pred_cfg,
            PREDICTOR_RES,
            False,
            ckpt_path=pred_cfg.evaluation.predictor.ckpt_path,
            is_featurizer=False,
        )
    else:
        pred_res = dict()

    finalize_stage(
        pred_cfg, predictor, pred_trainer, is_save_best=pred_cfg.predictor.is_save_best
    )

    ############## SHUTDOWN ##############

    return finalize(
        modules=dict(featurizer=compressor, predictor=predictor),
        trainers=dict(featurizer=comp_trainer, predictor=pred_trainer),
        datamodules=dict(featurizer=comp_datamodule, predictor=pred_datamodule),
        cfgs=dict(featurizer=comp_cfg, predictor=pred_cfg),
        results=dict(featurizer=feat_res, predictor=pred_res),
    )


def begin(cfg):
    """Script initialization."""
    if cfg.other.is_debug:
        set_debug(cfg)

    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())
    cfg.other.git_hash = GIT_HASH

    if cfg.rate.range_coder is not None:
        compressai.set_entropy_coder(cfg.rate.range_coder)

    logger.info(f"Workdir : {cfg.paths.work}.")

    if cfg.data_pred.name == "data_feat":
        # by default same data for pred and feat
        with omegaconf.open_dict(cfg):
            cfg.data_pred.name = cfg.data_feat.name
            cfg.data_pred = OmegaConf.merge(cfg.data_feat, cfg.data_pred)


def get_stage_name(mode):
    """Return the correct stage name given the mode (feturizer, predictor, ...)"""
    return mode[:4]


def set_cfg(cfg, mode):
    """Set the configurations for a specific mode."""
    cfg = copy.deepcopy(cfg)  # not inplace

    with omegaconf.open_dict(cfg):
        if mode == "featurizer":

            cfg.stage = get_stage_name(mode)
            cfg.long_name = cfg.long_name_feat

            cfg.data = OmegaConf.merge(cfg.data, cfg.data_feat)
            cfg.trainer = OmegaConf.merge(cfg.trainer, cfg.update_trainer_feat)
            cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg.checkpoint_feat)

            logger.info(f"Name : {cfg.long_name}.")

        elif mode == "predictor":
            cfg.stage = get_stage_name(mode)
            cfg.long_name = cfg.long_name_pred

            cfg.data = OmegaConf.merge(cfg.data, cfg.data_pred)
            cfg.trainer = OmegaConf.merge(cfg.trainer, cfg.update_trainer_pred)
            cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg.checkpoint_pred)

            # only need target
            cfg.data.kwargs.dataset_kwargs.additional_target = None

            logger.info(f"Name : {cfg.long_name}.")

        else:
            raise ValueError(f"Unkown mode={mode}.")

    if not cfg.is_no_save:
        # make sure all paths exist
        for _, path in cfg.paths.items():
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)

        Path(cfg.paths.pretrained.save).mkdir(parents=True, exist_ok=True)

    file_end = Path(cfg.paths.logs) / f"{cfg.stage}_{FILE_END}"
    if file_end.is_file():
        logger.info(f"Skipping most of {cfg.stage} as {file_end} exists.")

        with omegaconf.open_dict(cfg):
            if mode == "featurizer":
                cfg.featurizer.is_train = False
                cfg.evaluation.featurizer.is_evaluate = False

            elif mode == "predictor":  # improbable
                cfg.predictor.is_train = False
                cfg.evaluation.predictor.is_evaluate = False

    return cfg


def instantiate_datamodule_(cfg, pre_featurizer=None):
    """Instantiate dataset."""

    cfgd = cfg.data
    cfgt = cfg.trainer

    if cfg.trainer.gpus > 1 and cfg.trainer.get("accelerator", "ddp") == "ddp_spawn":
        # ddp_spawn very slow with multi workers
        cfgd.kwargs.num_workers = 0  # TODO test if true

    datamodule = get_datamodule(cfgd.dataset)(**cfgd.kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    if pre_featurizer is not None:
        datamodule = apply_featurizer(datamodule, pre_featurizer, **cfgd.kwargs)
        datamodule.prepare_data()
        datamodule.setup()

    cfgd.aux_is_clf = datamodule.aux_is_clf
    limit_train_batches = cfgt.get("limit_train_batches", 1)
    cfgd.length = int(len(datamodule.train_dataset) * limit_train_batches)
    cfgd.shape = datamodule.shape
    cfgd.target_is_clf = datamodule.target_is_clf
    cfgd.target_shape = datamodule.target_shape
    cfgd.aux_shape = datamodule.aux_shape
    cfgd.mode = datamodule.mode

    n_devices = max(cfgt.gpus * cfgt.num_nodes, 1)
    eff_batch_size = n_devices * cfgd.kwargs.batch_size * cfgt.accumulate_grad_batches
    train_batches = cfgd.length // eff_batch_size
    cfgd.max_steps = cfgt.max_epochs * train_batches

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
    n_param = sum(
        p.numel() for p in module.get_specific_parameters("all") if p.requires_grad
    )
    log_dict(trainer, {"n_param": n_param}, is_param=True)

    if cfg.evaluation.is_est_entropies:
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
                    # ReconstructImages(),
                ]

                if cfg.trainer.gpus == 1:
                    #! does not work (D)DP because of self.store
                    callbacks += [ReconstructImages()]

            elif cfg.data.mode == "distribution":
                callbacks += [
                    CodebookPlot(is_plot_codebook=is_reconstruct),
                ]
                if additional_target in can_estimate_Mx:
                    callbacks += [
                        MaxinvDistributionPlot(),
                    ]

    callbacks += [ModelCheckpoint(**cfg.checkpoint.kwargs)]

    if not cfg.callbacks.is_force_no_additional_callback:
        for name, kwargs in cfg.callbacks.items():
            try:
                if kwargs.is_use:
                    cllbck_kwargs = kwargs.get("kwargs", {})
                    modules = [lossyless.callbacks, pl.callbacks, pl_bolts.callbacks]
                    Callback = getattr_from_oneof(modules, name)
                    new_callback = Callback(**cllbck_kwargs)

                    if isinstance(new_callback, BaseFinetuning) and not is_featurizer:
                        pass  # don't add finetuner during prediciton
                    else:
                        callbacks.append(new_callback)

            except AttributeError:
                pass

    return callbacks


def get_logger(cfg, module, is_featurizer):
    """Return coorect logger."""

    kwargs = cfg.logger.kwargs
    # useful for different modes (e.g. wandb_kwargs)
    kwargs.update(cfg.logger.get(f"{cfg.logger.name}_kwargs", {}))

    if cfg.logger.name == "csv":
        pllogger = CSVLogger(**kwargs)

    elif cfg.logger.name == "wandb":
        check_import("wandb", "WandbLogger")

        try:
            pllogger = WandbLogger(**kwargs)
        except Exception:
            cfg.logger.kwargs.offline = True
            pllogger = WandbLogger(**kwargs)

        if cfg.trainer.track_grad_norm == 2:
            try:
                # use wandb rather than lightning gradients
                cfg.trainer.track_grad_norm = -1
                to_watch = module.p_ZlX.mapper if is_featurizer else module.predictor
                pllogger.watch(
                    to_watch,
                    log="gradients",
                    log_freq=cfg.trainer.log_every_n_steps * 10,
                )
            except:
                logger.exception("Cannot track gradients. Because:")
                pass

    elif cfg.logger.name == "tensorboard":
        pllogger = TensorBoardLogger(**kwargs)

    elif cfg.logger.name is None:
        pllogger = False

    else:
        raise ValueError(f"Unkown logger={cfg.logger.name}.")

    return pllogger


def get_trainer(cfg, module, is_featurizer):
    """Instantiate trainer."""

    # Resume training ?
    last_chckpnt = Path(cfg.checkpoint.kwargs.dirpath) / LAST_CHCKPNT
    if last_chckpnt.exists():
        cfg.trainer.resume_from_checkpoint = str(last_chckpnt)

    kwargs = dict(**cfg.trainer)

    # PARALLEL PROCESSING
    # cpu
    accelerator = kwargs.get("accelerator", None)
    if accelerator == "ddp_cpu_spawn":  # only for debug
        kwargs["accelerator"] = "ddp_cpu"
        kwargs["plugins"] = DDPSpawnPlugin(
            parallel_devices=[], find_unused_parameters=True
        )

    # gpu
    if kwargs["gpus"] > 1:
        kwargs["sync_batchnorm"] = True
        accelerator = kwargs.get("accelerator", "ddp")
        parallel_devices = [torch.device(f"cuda:{i}") for i in range(kwargs["gpus"])]

        #! ddp does not work yet with compressai https://github.com/InterDigitalInc/CompressAI/issues/30
        if accelerator == "ddp":
            kwargs["accelerator"] = "ddp"
            kwargs["plugins"] = DDPPlugin(
                parallel_devices=parallel_devices, find_unused_parameters=True
            )

        elif accelerator == "ddp_spawn":
            kwargs["accelerator"] = "ddp"
            kwargs["plugins"] = DDPSpawnPlugin(
                parallel_devices=parallel_devices, find_unused_parameters=True,
            )

        elif accelerator == "ddp_sharded":
            kwargs["accelerator"] = "ddp"
            kwargs["plugins"] = DDPShardedPlugin(
                parallel_devices=parallel_devices, find_unused_parameters=True,
            )

        elif accelerator == "ddp_sharded_spawn":
            kwargs["accelerator"] = "ddp"
            kwargs["plugins"] = DDPSpawnShardedPlugin(
                parallel_devices=parallel_devices, find_unused_parameters=True,
            )

        elif accelerator == "dp":
            kwargs["plugins"] = DataParallelPlugin(parallel_devices=parallel_devices)

    # TRAINER
    trainer = pl.Trainer(
        logger=get_logger(cfg, module, is_featurizer),
        callbacks=get_callbacks(cfg, is_featurizer),
        checkpoint_callback=True,
        **kwargs,
    )

    return trainer


def placeholder_fit(trainer, module, datamodule):
    """Necessary setup of trainer before testing if you don't fit it."""
    trainer.train_loop.setup_fit(module, None, None, datamodule)
    trainer.model = module


def save_pretrained(cfg, trainer, file):
    """Send best checkpoint for compressor to main directory."""

    # restore best checkpoint
    best = trainer.checkpoint_callback.best_model_path
    trainer.resume_from_checkpoint = best
    trainer.checkpoint_connector.restore_weights()

    # save
    dest_path = Path(cfg.paths.pretrained.save)
    dest_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(dest_path / file, weights_only=True)


def is_trained(cfg, file):
    """Test whether already saved the checkpoint, if yes then you already trained but might have preempted."""
    dest_path = Path(cfg.paths.pretrained.save)
    return (dest_path / file).is_file()


def load_pretrained(cfg, Module, file, **kwargs):
    """Load the best checkpoint from the latest run that has the same name as current run."""
    save_path = Path(cfg.paths.pretrained.load)
    # select the latest checkpoint matching the path
    chckpnt = get_latest_match(save_path / file)

    loaded_module = Module.load_from_checkpoint(chckpnt, **kwargs)

    return loaded_module


def is_est_entropies(cfg):
    # entropy estimation when Z is stochastic will not be good
    if cfg.evaluation.is_est_entropies and cfg.encoder.fam != "deterministic":
        logger.warning("Turning off `is_est_entropies` because stochastic Z.")
        return False
    return cfg.evaluation.is_est_entropies


def evaluate(
    trainer,
    datamodule,
    cfg,
    file,
    is_est_entropies=False,
    ckpt_path="best",
    is_featurizer=True,
):
    """
    Evaluate the trainer by loging all the metrics from the training and test set from the best model.
    Can also compute sample estimates of soem entropies, which should be better estimates than the
    lower bounds used during training. Only estimate entropies if `is_featurizer`.
    """
    try:
        # Evaluation
        eval_dataloader = datamodule.eval_dataloader(cfg.evaluation.is_eval_on_test)
        test_res = trainer.test(test_dataloaders=eval_dataloader, ckpt_path=ckpt_path)[
            0
        ]
        if is_est_entropies and is_featurizer:
            append_entropy_est_(test_res, trainer, datamodule, cfg, is_test=True)
        log_dict(trainer, test_res, is_param=False)

        test_res_rep = replace_keys(test_res, "test/", "")
        tosave = dict(test=test_res_rep)

        # Evaluation on train
        if cfg.data.length < 1e5:
            # don't eval on data if big
            train_res = trainer.test(
                test_dataloaders=datamodule.train_dataloader(), ckpt_path=ckpt_path
            )[0]
            train_res = replace_keys(train_res, "test", "testtrain")
            if is_est_entropies and is_featurizer:
                append_entropy_est_(train_res, trainer, datamodule, cfg, is_test=False)
            log_dict(trainer, train_res, is_param=False)

            train_res = replace_keys(train_res, "testtrain/", "")
            tosave["train"] = train_res

        # save results
        results = pd.DataFrame.from_dict(tosave)
        path = Path(cfg.paths.results) / file
        results.to_csv(path, header=True, index=True)
        logger.info(f"Logging results to {path}.")
    except:
        logger.exception("Failed to evaluate. Skipping this error:")
        test_res = dict()

    return test_res


def append_entropy_est_(results, trainer, datamodule, cfg, is_test):
    """Append entropy estimates to the results."""
    is_discrete_Y = cfg.data.target_is_clf
    is_discrete_M = datamodule.dataset.is_clf_x_t_Mx["max_inv"]

    # get the max invariant from the dataset
    dkwargs = {"additional_target": "max_inv"}
    if is_test:
        if cfg.evaluation.is_eval_on_test:
            dataloader = datamodule.test_dataloader(dataset_kwargs=dkwargs)
        else:
            # testing on validation (needed if don't have access to test set)
            dataloader = datamodule.val_dataloader(dataset_kwargs=dkwargs)
    else:
        dataloader = datamodule.train_dataloader(dataset_kwargs=dkwargs)

    H_MlZ, H_YlZ, H_Z = estimate_entropies(
        trainer, dataloader, is_discrete_M=is_discrete_M, is_discrete_Y=is_discrete_Y,
    )
    prfx = "test" if is_test else "testtrain"
    results[f"{prfx}/feat/H_MlZ"] = H_MlZ
    results[f"{prfx}/feat/H_YlZ"] = H_YlZ
    results[f"{prfx}/feat/H_Z"] = H_Z


def initialize_predictor_(module, datamodule, trainer, cfg):
    """Additional steps needed for intitalization of the predictor + logging."""
    if module.hparams.optimizer_pred.is_lr_find:
        old_lr = module.hparams.optimizer_pred.kwargs.lr
        new_lr = learning_rate_finder(module, datamodule, trainer)

        if (old_lr is None) and (new_lr is None):
            raise ValueError(f"Couldn't find new lr and no old lr given.")

        if old_lr is None and (new_lr is not None):
            module.hparams.optimizer_pred.kwargs.lr = new_lr
            logger.info(f"Using lr={new_lr} for the predictor.")
        else:
            module.hparams.optimizer_pred.kwargs.lr = old_lr


def finalize_stage(cfg, module, trainer, is_save_best=True):
    """Finalize the current stage."""
    logger.info(f"Finalizing {cfg.stage}.")

    assert (
        cfg.checkpoint.kwargs.dirpath != cfg.paths.pretrained.save
    ), "This will remove diesired checkpoints"

    for checkpoint in Path(cfg.checkpoint.kwargs.dirpath).glob("*.ckpt"):
        checkpoint.unlink()  # remove all checkpoints as best is already saved elsewhere

    # don't keep the pretrained model
    if not is_save_best:
        dest_path = Path(cfg.paths.pretrained.save)
        for checkpoint in dest_path.glob("*.ckpt"):
            checkpoint.unlink()  # remove all checkpoints

    if not cfg.is_no_save:
        # save end file to make sure that you don't retrain if preemption
        file_end = Path(cfg.paths.logs) / f"{cfg.stage}_{FILE_END}"
        file_end.touch(exist_ok=True)

        # save config to results
        cfg_save(cfg, Path(cfg.paths.results) / f"{cfg.stage}_{CONFIG_FILE}")


def finalize(modules, trainers, datamodules, cfgs, results):
    """Finalizes the script."""
    cfg = cfgs["featurizer"]  # this is always in

    logger.info("Stage : Shutdown")

    plt.close("all")

    if cfg.logger.name == "wandb":
        if wandb.run is not None:
            wandb.run.finish()  # finish the run if still on

    logger.info("Finished.")
    logging.shutdown()

    all_results = dict()
    for partial_results in results.values():
        all_results.update(partial_results)

    if cfg.is_return:
        return modules, trainers, datamodules, cfgs
    else:
        return get_hypopt_monitor(cfg, all_results)


def get_hypopt_monitor(cfg, all_results):
    """Return the corret monitor for hyperparameter tuning."""
    out = []
    for i, result_key in enumerate(cfg.monitor_return):
        res = all_results[result_key]
        try:
            direction = cfg.monitor_direction[i]
            if not math.isfinite(res):
                # make sure that infinte or nan monitor are not selected by hypopt
                if direction == "minimize":
                    res = float("inf")
                else:
                    res = -float("inf")
        except IndexError:
            pass

        out.append(res)

    if len(out) == 1:
        return out[0]  # return single value rather than tuple
    return tuple(out)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main()
