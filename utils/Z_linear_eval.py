"""Very quick linear evaluation of CLIP + bottleneck."""
import logging
import math
import os
import sys
import time
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils.fixes import loguniform
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10

import hydra
from omegaconf import OmegaConf

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from lossyless import CondDist, LearnableCompressor, get_Architecture  # isort:skip
from utils.data import get_datamodule  # isort:skip
from main import begin, set_cfg, RESULTS_FILE  # isort:skip
from utils.load_pretrained import PretrainedAnalyser  # isort:skip
from utils.helpers import (  # isort:skip
    format_resolver,
    omegaconf2namespace,
    replace_keys,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{MAIN_DIR}/config", config_name="clip_linear")
def main(cfg):

    analyser = PretrainedAnalyser()
    logger.info(f"Collecting the data ..")
    stage = "predictor"

    analyser.collect_data(cfg, **cfg.load_pretrained.collect_data)

    Z_train = analyser.datamodules["predictor"].train_dataset.X
    Y_train = analyser.datamodules["predictor"].train_dataset.Y
    Z_val = analyser.datamodules["predictor"].val_dataset.X
    Y_val = analyser.datamodules["predictor"].val_dataset.Y
    Z_test = analyser.datamodules["predictor"].test_dataset.X
    Y_test = analyser.datamodules["predictor"].test_dataset.Y

    Z = np.concatenate((Z_train, Z_val))
    Y = np.concatenate((Y_train, Y_val))

    parameters = {"C": loguniform(1e-3, 1e-0), "class_weight": ["balanced", None]}
    svc = LinearSVC(random_state=cfg.seed, dual=False)

    if len(Z) > 5e4:
        # use validation for large data
        val_fold = PredefinedSplit([-1] * len(Z_train) + [0] * len(Z_val))
    else:
        # use cross validation otherwise
        val_fold = 5

    clf = RandomizedSearchCV(
        svc,
        parameters,
        scoring="accuracy",
        n_jobs=8,
        cv=val_fold,
        n_iter=cfg.clip_linear.n_predictors,
    )

    clf.fit(Z, Y)

    Y_pred = clf.predict(Z_test)

    bacc = balanced_accuracy_score(Y_test, Y_pred)
    acc = accuracy_score(Y_test, Y_pred)

    analyser.results["test/pred/balanced_acc"] = bacc
    analyser.results["test/pred/acc"] = acc
    analyser.results["test/pred/err"] = 1 - acc
    analyser.results["test/pred/balanced_err"] = 1 - bacc

    # save results
    test_res_rep = replace_keys(analyser.results, "test/", "")
    tosave = dict(test=test_res_rep)
    results = pd.DataFrame.from_dict(tosave)
    results.to_csv(results_file, header=True, index=True)
    logger.info(f"Logged results to {path}.")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main()
