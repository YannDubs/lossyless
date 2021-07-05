"""Save the rate estimator from a pretrained model to hub.

This should be called by `python utils/save_hub.py <conf>` where <conf> sets all configs used to 
pretrain the model in the first place`.
"""
import logging
import os
import sys
from pathlib import Path

import torch

import hydra
from omegaconf import OmegaConf

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)


from utils.load_pretrained import PretrainedAnalyser  # isort:skip
from utils.helpers import format_resolver  # isort:skip

logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{MAIN_DIR}/config", config_name="load_pretrained")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    try:
        return main(cfg)
    except:
        logger.exception("Failed to save pretrained with this error:")
        # don't raise error because if multirun want the rest to work
        pass


def main(cfg):

    analyser = PretrainedAnalyser(**cfg.load_pretrained.kwargs)

    logger.info(f"Collecting the data ..")
    analyser.collect_data(cfg, **cfg.load_pretrained.collect_data)

    rate_estimator = analyser.modules["featurizer"].rate_estimator
    model_name = f"{MAIN_DIR}/hub/beta{cfg.featurizer.loss.beta:.0e}/factorized_rate.pt"
    torch.save(
        rate_estimator.state_dict(), model_name,
    )

    logger.info(f"Saved pretrained model to {model_name}...")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main_cli()
