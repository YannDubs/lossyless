import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics.functional import accuracy

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[2]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from lossyless import (  # isort:skip
    get_rate_estimator,
    get_Architecture,
    LearnableCompressor,
    OnlineEvaluator,
    CondDist,
)
from lossyless.helpers import Annealer  # isort:skip
from utils.featurize import get_featurized_data  # isort:skip
from utils.helpers import (  # isort:skip
    omegaconf2namespace,
    replace_keys,
    format_resolver,
)
from main import (  # isort:skip
    LAST_CHCKPNT,
    PREDICTOR_RES,
    finalize_stage,
    get_hypopt_monitor,
    get_trainer,
)

logger = logging.getLogger(__name__)


class Featurizer(LearnableCompressor):
    def get_encoder(self):
        return CondDist(
            self.hparams.encoder.z_dim,
            self.hparams.encoder.z_dim,
            family="deterministic",
            Architecture=get_Architecture(mode="identity"),
        )

    def get_distortion_estimator(self):
        return None

    def step(self, batch):
        z, _ = batch

        # z_hat. shape: [q, batch_size, z_dim]
        z_hat, rates, r_logs, _ = self.rate_estimator(z_in, None, self)

        p = self.hparams.distortion.p_norm
        distortions = torch.norm(z_in - z_hat, p=p, dim=-1)

        loss, logs, _ = self.loss(rates, distortions)

        logs.update(r_logs)
        logs.update(dict(zmin=z_hat.min(), zmax=z_hat.max(), zmean=z_hat.mean()))
        other = dict()

        return loss, logs, other

    def on_test_epoch_start(self):
        # self.rate_estimator.prepare_compressor_()
        # don't compute real rate yet because is off by large amount
        pass


def begin(cfg):
    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())

    if not cfg.is_no_save:
        # make sure all paths exist
        for _, path in cfg.paths.items():
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)


@hydra.main(config_name="selfsupervised_feat", config_path=f"{MAIN_DIR}/config")
def run_classifier(cfg):
    begin(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = get_featurized_data(
        {cfg.dataset: None},
        cfg.model,
        device=device,
        is_half=True,
        data_dir=cfg.paths.data,
    )

    features = features[cfg.dataset]
    X_train = features["train"]["Z"][:]
    Y_train = features["train"]["Y"][:]
    X_test = features["test"]["Z"][:]
    Y_test = features["test"]["Y"][:]

    cfg.data.length = len(X_train)
    train_batches = cfg.data.length // cfg.data.kwargs.batch_size
    cfg.data.max_steps = cfg.trainer.max_epochs * train_batches
    cfg.data.target_shape = len(np.unique(Y_train))
    cfg.encoder.z_dim = X_train.shape[-1]

    cfg = omegaconf2namespace(cfg)

    datamodule = SklearnDataModule(
        X_train, Y_train, x_test=X_test, y_test=Y_test, **cfg.data.kwargs
    )

    model = Featurizer(cfg)

    trainer = get_trainer(cfg, model, False)
    trainer.fit(model, datamodule=datamodule)

    test_res = trainer.test()[0]

    # save results
    test_res_rep = replace_keys(test_res, "test/", "")
    tosave = dict(test=test_res_rep)
    results = pd.DataFrame.from_dict(tosave)
    path = Path(cfg.paths.results) / PREDICTOR_RES
    logger.info(f"Logging results to {path}.")
    results.to_csv(path, header=True, index=True)

    finalize_stage(cfg, model, trainer, is_save_best=False)

    return get_hypopt_monitor(cfg, test_res)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    run_classifier()
