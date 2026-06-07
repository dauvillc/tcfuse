"""Shared submitit launcher utilities."""

from pathlib import Path
from time import localtime, strftime
from typing import Any

import submitit


def make_executor(cfg: dict[str, Any], job_name: str) -> submitit.AutoExecutor:
    """Create a submitit AutoExecutor configured from cfg["setup"].

    Args:
        cfg: Full Hydra config dict (must contain a "setup" key).
        job_name: SLURM job name and submitit log sub-directory prefix.

    Returns:
        Configured AutoExecutor ready for job submission.
    """
    timestamp = strftime("%Y%m%d_%H-%M-%S", localtime())
    log_dir = Path("submitit") / f"{job_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ex = submitit.AutoExecutor(folder=str(log_dir), slurm_max_num_timeout=20)
    # job_name overrides setup.name so squeue shows the script-specific label.
    setup = {**cfg["setup"], "name": job_name}
    ex.update_parameters(**setup)
    return ex
