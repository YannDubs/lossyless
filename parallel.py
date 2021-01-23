"""
Entry point that replaces `main.py` in case your jobs are very small and should be run in parallel
on the SAME GPU => multiple runs for a single job.

E.g. usage to run 3 subjobs for every job (every GPU) each with different seed: 
`python parallel.py +parallel=small_seeds <other param>`
"""
import logging
import subprocess
from contextlib import ExitStack
from pathlib import Path

import hydra
from utils.helpers import create_folders

logger = logging.getLogger(__name__)


@hydra.main(config_name="main", config_path="config")
def main(cfg):

    # all overriden parameters to keep
    params = cfg.parallel.override_dirname.split()
    p_to_remove = ["server", "launcher", "parallel"]
    params = [p for p in params if not any(rm in p for rm in p_to_remove)]

    key = cfg.parallel.key
    to_sweep = cfg[key]

    logger.info(f"Sweeping {key}={to_sweep} on single GPU from {str(Path.cwd())}.")

    # create one directory for each subjob
    create_folders("./", [f"{val}" for val in to_sweep])

    processes = []
    with ExitStack() as stack:
        # equivalent to `with open ...` for each logging file
        files_out = [
            stack.enter_context(open(f"./{val}/logs.out", "a+")) for val in to_sweep
        ]
        files_err = [
            stack.enter_context(open(f"./{val}/logs.err", "a+")) for val in to_sweep
        ]

        for i, (val, out_f, err_f) in enumerate(zip(to_sweep, files_out, files_err)):

            run_dir = f"hydra.run.dir=./{val}/"
            val = f"{key}={val}"
            job_id = f"job_id={i}_{cfg.job_id}"  # keep job id if preemting
            base_dir = f"paths.base_dir={cfg.paths.base_dir}"
            p_to_add = [run_dir, val, job_id, base_dir]

            main_script = str(Path(cfg.paths.base_dir) / "main.py")
            command = ["python", "-u", main_script]
            command += params + p_to_add

            logger.info(f"Command: {' '.join(command) }")
            pi = subprocess.Popen(command, bufsize=0, stdout=out_f, stderr=err_f,)
            processes.append(pi)

    exit_codes = [p.wait() for p in processes]
    logger.info(f"All finished with exit codes: {exit_codes}")


if __name__ == "__main__":
    main()
