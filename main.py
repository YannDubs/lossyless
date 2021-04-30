"""Entry point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""
import copy
import csv
from datetime import datetime
import logging
import math
import subprocess
from time import sleep
import os
from pathlib import Path

import compressai
import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import pl_bolts
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities import rank_zero_warn

import lossyless
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
from utils.data import get_datamodule
from utils.data.images import GalaxyDataset
from utils.helpers import (
    ModelCheckpoint,
    apply_featurizer,
    cfg_save,
    format_resolver,
    get_latest_match,
    getattr_from_oneof,
    log_dict,
    omegaconf2namespace,
    replace_keys,
    set_debug,
    remove_rf
)

try:
    import wandb
except ImportError:
    pass

try:
    import kaggle
except:
    logger.warning("WARNING: import kaggle failed, only important if you run GalaxyZoo experiments.")


logger = logging.getLogger(__name__)
BEST_CHCKPNT = "best_{stage}.ckpt"
RESULTS_FILE = "results_{stage}.csv"
LAST_CHCKPNT = "last.ckpt"
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
    finalize_kwargs = dict(modules={}, trainers={}, datamodules={}, cfgs={}, results={})

    ############## COMPRESSOR (i.e. sender) ##############
    logger.info("Stage : Compressor")
    stage = "featurizer"
    comp_cfg = set_cfg(cfg, stage)
    comp_datamodule = instantiate_datamodule_(comp_cfg)
    comp_cfg = omegaconf2namespace(comp_cfg)  # ensure real python types

    if not comp_cfg.featurizer.is_learnable:
        logger.info(f"Using classical compressor {comp_cfg.featurizer.mode} ...")
        compressor = ClassicalCompressor(hparams=comp_cfg)
        comp_trainer = get_trainer(
            comp_cfg,
            compressor,
            is_featurizer=True,
        )
        placeholder_fit(comp_trainer, compressor, comp_datamodule)

    elif comp_cfg.featurizer.is_train and not is_trained(comp_cfg, stage):
        compressor = LearnableCompressor(hparams=comp_cfg)
        comp_trainer = get_trainer(comp_cfg, compressor, is_featurizer=True)
        initialize_compressor_(compressor, comp_datamodule, comp_trainer, comp_cfg)

        logger.info("Train compressor ...")
        comp_trainer.fit(compressor, datamodule=comp_datamodule)
        save_pretrained(comp_cfg, comp_trainer, stage)
    else:
        logger.info("Load pretrained compressor ...")
        compressor = load_pretrained(comp_cfg, LearnableCompressor, stage)
        comp_trainer = get_trainer(comp_cfg, compressor, is_featurizer=True)
        placeholder_fit(comp_trainer, compressor, comp_datamodule)
        comp_cfg.evaluation.featurizer.ckpt_path = None  # eval loaded model

    if comp_cfg.evaluation.featurizer.is_evaluate:
        logger.info("Evaluate compressor ...")
        feat_res = evaluate(comp_trainer, comp_datamodule, comp_cfg, stage)
    else:
        feat_res = load_results(comp_cfg, stage)

    finalize_stage_(
        stage,
        comp_cfg,
        compressor,
        comp_trainer,
        comp_datamodule,
        feat_res,
        finalize_kwargs,
        is_save_best=True,
    )
    if comp_cfg.is_only_feat:
        return finalize(**finalize_kwargs)

    del comp_datamodule  # not used anymore and can be large

    ############## COMMUNICATION (compress and decompress the datamodule) ##############
    logger.info("Stage : Communication")
    stage = "communication"
    comm_cfg = set_cfg(cfg, stage)
    comm_datamodule = instantiate_datamodule_(comm_cfg)
    comm_cfg = omegaconf2namespace(comm_cfg)

    if comp_cfg.featurizer.is_on_the_fly:
        # this will perform compression on the fly. Issue is that augmentations will be applies
        # before the featurizer which is less realisitic (normalization is dealt correctly though)
        onfly_featurizer = compressor
        pre_featurizer = None
    else:
        # compressing once the dataset is more realistic (and quicker) but requires more RAM
        onfly_featurizer = None
        pre_featurizer = comp_trainer

    if comm_cfg.evaluation.communication.is_evaluate:
        logger.info("Evaluate communication ...")
        comm_res = evaluate(comp_trainer, comm_datamodule, comm_cfg, stage)
    else:
        comm_res = load_results(comm_cfg, stage)

    finalize_stage_(
        stage, comm_cfg, None, None, comm_datamodule, comm_res, finalize_kwargs
    )

    del comm_datamodule  # not used anymore and can be large

    ############## DOWNSTREAM PREDICTOR (i.e. receiver) ##############
    logger.info("Stage : Predictor")
    stage = "predictor"
    pred_cfg = set_cfg(cfg, stage)
    pred_datamodule = instantiate_datamodule_(pred_cfg, pre_featurizer=pre_featurizer)
    pred_cfg = omegaconf2namespace(pred_cfg)

    if pred_cfg.predictor.is_train and not is_trained(comp_cfg, stage):
        predictor = Predictor(hparams=pred_cfg, featurizer=onfly_featurizer)
        pred_trainer = get_trainer(pred_cfg, predictor, is_featurizer=False)

        logger.info("Train predictor ...")
        pred_trainer.fit(predictor, datamodule=pred_datamodule)
        save_pretrained(pred_cfg, pred_trainer, stage)

    else:
        logger.info("Load pretrained predictor ...")
        FeatPred = get_featurizer_predictor(onfly_featurizer)
        predictor = load_pretrained(pred_cfg, FeatPred, stage)
        pred_trainer = get_trainer(pred_cfg, predictor, is_featurizer=False)
        placeholder_fit(pred_trainer, predictor, pred_datamodule)
        pred_cfg.evaluation.predictor.ckpt_path = None  # eval loaded model

    if pred_cfg.evaluation.predictor.is_evaluate:
        logger.info("Evaluate predictor ...")
        pred_res = evaluate(pred_trainer, pred_datamodule, pred_cfg, stage)
    else:
        pred_res = load_results(pred_cfg, stage)

    finalize_stage_(
        stage,
        pred_cfg,
        predictor,
        pred_trainer,
        pred_datamodule,
        pred_res,
        finalize_kwargs,
    )

    ############## SHUTDOWN ##############

    return finalize(**finalize_kwargs)


def begin(cfg):
    """Script initialization."""
    if cfg.other.is_debug:
        set_debug(cfg)

    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())
    cfg.other.git_hash = GIT_HASH

    logger.info(f"Workdir : {cfg.paths.work}.")

    if cfg.data_pred.name == "data_feat":
        # by default same data for pred and feat
        with omegaconf.open_dict(cfg):
            cfg.data_pred.name = cfg.data_feat.name
            cfg.data_pred = OmegaConf.merge(cfg.data_feat, cfg.data_pred)


def get_stage_name(mode):
    """Return the correct stage name given the mode (feturizer, predictor, ...)"""
    return mode[:4]


def set_cfg(cfg, stage):
    """Set the configurations for a specific mode."""
    cfg = copy.deepcopy(cfg)  # not inplace

    with omegaconf.open_dict(cfg):
        if stage == "featurizer":

            cfg.stage = get_stage_name(stage)
            cfg.long_name = cfg.long_name_feat

            cfg.data = OmegaConf.merge(cfg.data, cfg.data_feat)
            cfg.trainer = OmegaConf.merge(cfg.trainer, cfg.update_trainer_feat)
            cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg.checkpoint_feat)

            logger.info(f"Name : {cfg.long_name}.")

        elif stage == "communication":
            cfg.stage = get_stage_name(stage)
            cfg.long_name = cfg.long_name_comm

            # currntly only communicate data_pred. But easy to change
            cfg.data = OmegaConf.merge(cfg.data, cfg.data_pred)

            # follwoing is not actually used but simply ensures that interpolation keys are possible
            # e.g. need a checkpoint.kwargs.monitot for defining schedulers
            cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg.checkpoint_pred)

            logger.info(f"Name : {cfg.long_name}.")

        elif stage == "predictor":
            cfg.stage = get_stage_name(stage)
            cfg.long_name = cfg.long_name_pred

            cfg.data = OmegaConf.merge(cfg.data, cfg.data_pred)
            cfg.trainer = OmegaConf.merge(cfg.trainer, cfg.update_trainer_pred)
            cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg.checkpoint_pred)

            # only need target
            cfg.data.kwargs.dataset_kwargs.additional_target = None

            logger.info(f"Name : {cfg.long_name}.")

        else:
            raise ValueError(f"Unkown stage={stage}.")

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
            if stage == "featurizer":
                cfg.featurizer.is_train = False
                cfg.evaluation.featurizer.is_evaluate = False

            elif stage == "communication":
                cfg.evaluation.communication.is_evaluate = False

            elif stage == "predictor":  # improbable
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

    cfgd.aux_is_clf = datamodule.aux_is_clf
    limit_train_batches = cfgt.get("limit_train_batches", 1)
    cfgd.length = int(len(datamodule.train_dataset) * limit_train_batches)
    cfgd.shape = datamodule.shape
    cfgd.target_is_clf = datamodule.target_is_clf
    cfgd.target_shape = datamodule.target_shape
    cfgd.balancing_weights = datamodule.balancing_weights
    cfgd.aux_shape = datamodule.aux_shape
    cfgd.mode = datamodule.mode
    if pre_featurizer is not None:
        datamodule = apply_featurizer(
            datamodule,
            pre_featurizer,
            is_eval_on_test=cfg.evaluation.is_eval_on_test,
            **cfgd.kwargs,
        )
        datamodule.prepare_data()
        datamodule.setup()

        # changes due to the featurization
        cfgd.shape = (datamodule.train_dataset.X.shape[-1],)
        cfgd.mode = "vector"

    n_devices = max(cfgt.gpus * cfgt.num_nodes, 1)
    eff_batch_size = n_devices * cfgd.kwargs.batch_size * cfgt.accumulate_grad_batches
    train_batches = cfgd.length // eff_batch_size
    cfgd.max_steps = cfgt.max_epochs * train_batches

    return datamodule


def initialize_compressor_(module, datamodule, trainer, cfg):
    """Additional steps needed for intitalization of the compressor + logging."""

    # TODO remove if not using vampprior
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


def get_callbacks(cfg, is_featurizer):
    """Return list of callbacks."""
    callbacks = []

    if is_featurizer:
        additional_target = cfg.data.kwargs.dataset_kwargs.additional_target
        is_reconstruct = additional_target in ["representative", "input"]
        if cfg.logger.is_can_plot_img:
            if cfg.data.mode == "image" and is_reconstruct:
                callbacks += [
                    LatentDimInterpolator(cfg.encoder.z_dim),
                ]

                if cfg.trainer.gpus == 1:
                    #! does not work (D)DP because of self.store
                    callbacks += [ReconstructImages()]

            elif cfg.data.mode == "distribution":
                callbacks += [
                    CodebookPlot(
                        is_plot_codebook=is_reconstruct,
                    )
                ]
                if is_reconstruct:
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
    # TODOnly one)
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
                parallel_devices=parallel_devices,
                find_unused_parameters=True,
            )

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


def save_pretrained(cfg, trainer, stage):
    """Send best checkpoint for compressor to main directory."""

    # restore best checkpoint
    best = trainer.checkpoint_callback.best_model_path
    trainer.resume_from_checkpoint = best
    trainer.checkpoint_connector.restore_weights()

    # save
    dest_path = Path(cfg.paths.pretrained.save)
    dest_path.mkdir(parents=True, exist_ok=True)
    filename = BEST_CHCKPNT.format(stage=stage)
    trainer.save_checkpoint(dest_path / filename, weights_only=True)


def is_trained(cfg, stage):
    """Test whether already saved the checkpoint, if yes then you already trained but might have preempted."""
    dest_path = Path(cfg.paths.pretrained.save)
    filename = BEST_CHCKPNT.format(stage=stage)
    return (dest_path / filename).is_file()


def load_pretrained(cfg, Module, stage, **kwargs):
    """Load the best checkpoint from the latest run that has the same name as current run."""
    save_path = Path(cfg.paths.pretrained.load)
    filename = BEST_CHCKPNT.format(stage=stage)
    # select the latest checkpoint matching the path
    chckpnt = get_latest_match(save_path / filename)

    loaded_module = Module.load_from_checkpoint(chckpnt, **kwargs)

    return loaded_module


def evaluate(trainer, datamodule, cfg, stage):
    """Evaluate the trainer by loging all the metrics from the test set from the best model."""
    try:
        trainer.lightning_module.stage = cfg.stage  # logging correct stage
        eval_dataloader = datamodule.eval_dataloader(cfg.evaluation.is_eval_on_test)
        ckpt_path = cfg.evaluation[stage].ckpt_path

        if isinstance(datamodule.test_dataset,GalaxyDataset) and stage == "predictor":
            logger.warning("Testing on Galaxy test set.")
            test_dataloader = datamodule.eval_dataloader(True)
            model = load_best_ckpt(trainer, ckpt_path=ckpt_path)
            test_preds = trainer.predict(model=model, dataloaders=test_dataloader)
            test_res = kaggle_eval(predictions=test_preds, message="test_pred_{}".format(datetime.now()))
        else: # most common case:
            test_res = trainer.test(test_dataloaders=eval_dataloader, ckpt_path=ckpt_path)[0]

        # ensure that select only correct stage (important when communicating)
        test_res = {k: v for k, v in test_res.items() if f"/{cfg.stage}/" in k}

        log_dict(trainer, test_res, is_param=False)

        test_res_rep = replace_keys(test_res, "test/", "")
        tosave = dict(test=test_res_rep)

        # save results
        results = pd.DataFrame.from_dict(tosave)
        filename = RESULTS_FILE.format(stage=stage)
        path = Path(cfg.paths.results) / filename
        results.to_csv(path, header=True, index=True)
        logger.info(f"Logging results to {path}.")

    except:
        logger.exception("Failed to evaluate. Skipping this error:")
        test_res = dict()

    return test_res


def load_results(cfg, stage):
    """
    Load the results that were previsously saved or return empty dict. Useful in case you get_trainer
    premempted but still need access to the results.
    """
    try:
        filename = RESULTS_FILE.format(stage=stage)
        path = Path(cfg.paths.results) / filename

        # dict of "test","train" ... where subdicts are keys and results
        results = pd.read_csv(path, index_col=0).to_dict()

        results = {
            f"{mode}/{k}": v
            for mode, sub_dict in results.items()
            for k, v in sub_dict.items()
        }
        return results
    except:
        return dict()


def finalize_stage_(
    stage,
    cfg,
    module,
    trainer,
    datamodule,
    results,
    finalize_kwargs,
    is_save_best=False,
):
    """Finalize the current stage."""
    logger.info(f"Finalizing {stage}.")

    if stage != "communication":

        # no checkpoints during communication
        assert (
            cfg.checkpoint.kwargs.dirpath != cfg.paths.pretrained.save
        ), "This will remove diesired checkpoints"

        # remove all checkpoints as best is already saved elsewhere
        remove_rf(cfg.checkpoint.kwargs.dirpath, not_exist_ok=True) 

        # don't keep the pretrained model
        if not is_save_best:
            remove_rf(cfg.paths.pretrained.save, not_exist_ok=True)

    if not cfg.is_no_save:
        # save end file to make sure that you don't retrain if preemption
        file_end = Path(cfg.paths.logs) / f"{cfg.stage}_{FILE_END}"
        file_end.touch(exist_ok=True)

        # save config to results
        cfg_save(cfg, Path(cfg.paths.results) / f"{cfg.stage}_{CONFIG_FILE}")

    finalize_kwargs["results"][stage] = results
    finalize_kwargs["cfgs"][stage] = cfg

    if cfg.is_return:
        # don't store large stuff if uneccessary
        finalize_kwargs["modules"][stage] = module
        finalize_kwargs["trainers"][stage] = trainer
        finalize_kwargs["datamodules"][stage] = datamodule


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

def load_best_ckpt(trainer, ckpt_path):
    """Returns the model from trainer that is assosiated with ckpt_path."""
    model = trainer.lightning_module
    # if user requests the best checkpoint but we don't have it, error
    if ckpt_path == 'best' and not trainer.checkpoint_callback.best_model_path:
        raise MisconfigurationException(
            'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
        )

    # load best weights
    if ckpt_path is not None:
        # ckpt_path is 'best' so load the best model
        if ckpt_path == 'best':
            ckpt_path = trainer.checkpoint_callback.best_model_path

        if len(ckpt_path) == 0:
            rank_zero_warn(
                f'.test() found no path for the best weights, {ckpt_path}. Please '
                f'specify a path for a checkpoint .test(ckpt_path=PATH)'
            )
            return {}

        trainer.training_type_plugin.barrier()

        ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])
    return model

def kaggle_eval(predictions, message='Kaggle Evaluation {}'.format(datetime.now())):

        tmp_path = os.path.join(os.getcwd(),'test_predictions.csv')

        with open(tmp_path, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(
                ['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2',
                 'Class4.1',
                 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1',
                 'Class7.2',
                 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
                 'Class9.1',
                 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3',
                 'Class11.4', 'Class11.5', 'Class11.6'])

            for prediction in predictions:
                for idx, pred in zip(prediction[0], prediction[1]):
                    row = [idx] + pred[0].tolist()
                    writer.writerow(row)

        kaggle.api.competition_submit(file_name=tmp_path,
                                      message=message,
                                      competition="galaxy-zoo-the-galaxy-challenge")

        sleep(5)   # wait for kaggle to process our submission
        submissions_list = kaggle.api.competitions_submissions_list_with_http_info(id="galaxy-zoo-the-galaxy-challenge")
        my_submission = [s for s in kaggle.api.process_response(submissions_list) if s['description']==message]
        result = my_submission[0]['publicScore']

        return {"test/pred/kaggle_score": float(result), "test/pred/loss": float(result)**2 }

if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main()
