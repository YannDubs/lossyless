"""Purges some results if you have to rerun something for rerunning"""
import shutil
from pathlib import Path

import hydra


@hydra.main(config_name="purge", config_path="../config")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    return main(cfg)


def main(cfg):
    pattern = "**/datafeat_*" if cfg.is_purge_featurizer else "**/datapred_*"

    if cfg.match is not None:
        pattern += f"/**/*{cfg.match}*"

    for folder in cfg.folders:
        for p in Path(folder).glob(f"**/{pattern}"):
            shutil.rmtree(p)


if __name__ == "__main__":
    main_cli()
