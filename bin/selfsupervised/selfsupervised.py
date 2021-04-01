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

from lossyless.helpers import append_optimizer_scheduler_  # isort:skip
from utils.featurize import get_featurized_data  # isort:skip
from utils.helpers import (  # isort:skip
    omegaconf2namespace,
    replace_keys,
    format_resolver,
)
from lossyless.architectures import get_Architecture  # isort:skip
from main import (  # isort:skip
    LAST_CHCKPNT,
    PREDICTOR_RES,
    finalize_stage,
    get_hypopt_monitor,
    get_trainer,
)

logger = logging.getLogger(__name__)


class Classifier(pl.LightningModule):
    def __init__(
        self, input_dim, num_classes, hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        cfga = self.hparams.architecture
        Model = get_Architecture(cfga.arch, **cfga.arch_kwargs)
        self.model = Model(input_dim, num_classes)

    def forward(self, x):
        logits = self.model(x)
        y_hat = F.softmax(logits, dim=-1)
        return y_hat

    def configure_optimizers(self):
        optimizers, schedulers = [], []

        append_optimizer_scheduler_(
            self.hparams.optimizer,
            self.hparams.scheduler,
            self.parameters(),
            optimizers,
            schedulers,
        )

        return optimizers, schedulers

    def step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(F.softmax(logits, dim=-1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log(f"train/pred/loss", loss)
        self.log(f"train/pred/acc", acc)
        self.log(f"train/pred/err", 1 - acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log(f"val/pred/loss", loss)
        self.log(f"val/pred/acc", acc)
        self.log(f"val/pred/err", 1 - acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log(f"test/pred/loss", loss)
        self.log(f"test/pred/acc", acc)
        self.log(f"test/pred/err", 1 - acc)
        return loss


def begin(cfg):
    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())

    if not cfg.is_no_save:
        # make sure all paths exist
        for _, path in cfg.paths.items():
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)


@hydra.main(config_name="selfsupervised", config_path=f"{MAIN_DIR}/config")
def run_classifier(cfg):
    begin(cfg)
    cfg = omegaconf2namespace(cfg)

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

    datamodule = SklearnDataModule(
        X_train, Y_train, x_test=X_test, y_test=Y_test, **cfg.data_kwargs
    )

    model = Classifier(
        input_dim=X_train.shape[-1], num_classes=len(np.unique(Y_train)), hparams=cfg,
    )

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
