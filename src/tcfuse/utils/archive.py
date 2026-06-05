"""Archive utilities for submitting tarball-creation jobs to the default SLURM CPU partition.

After a preprocessing or training script successfully writes its outputs to SCRATCH,
call :func:`submit_archive_job` to asynchronously copy the data to STORE as a .tar.gz.
The function is a no-op when ``cfg["archive"]`` is False, so it is safe to call
unconditionally in all scripts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import submitit

from tcfuse.utils.submitit_utils import make_executor

# Jean-Zay archive job constants.
# No slurm_partition is set — omitting it lets SLURM schedule on the default CPU partition,
# which is much easier to obtain than the archive/prepost nodes.
_ARCHIVE_ACCOUNT = "xyw@cpu"
_ARCHIVE_SETUP_COMMANDS = ["module load pytorch-gpu/py3/2.8.0"]
_ARCHIVE_TIMEOUT_MIN = 240


def _create_tarball(src_path: Path, tar_path: Path) -> None:
    """Create a gzip tarball of src_path at tar_path.

    Writes atomically: archives to a .tmp file, then renames to the final path.
    Cleans up the temporary file on failure.

    Args:
        src_path: Directory to archive (must exist).
        tar_path: Destination tarball path (e.g. /store/archives/pmw_amsr2.tar.gz).
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    # Temporary path prevents a partially-written tarball from appearing at the final path.
    tmp_path = tar_path.parent / (tar_path.name + ".tmp")
    try:
        # -C <parent> so the archive contains src_path.name/, not the absolute path.
        subprocess.run(
            ["tar", "-czf", str(tmp_path), "-C", str(src_path.parent), src_path.name],
            check=True,
        )
        tmp_path.rename(tar_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    size_mb = tar_path.stat().st_size / 1e6
    print(f"Archived {src_path} → {tar_path} ({size_mb:.1f} MB)")


def submit_archive_job(
    src_path: Path,
    tar_path: Path,
    cfg: dict[str, Any],
    job_name: str,
) -> submitit.Job | None:
    """Submit a tarball-creation job to the SLURM archive partition.

    Does nothing and returns None when ``cfg["archive"]`` is False or absent.

    Args:
        src_path: Directory to archive (must exist at job execution time).
        tar_path: Destination tarball path.
        cfg: Full Hydra config dict; must contain a "setup" key when archive is enabled.
        job_name: SLURM job name and submitit log sub-directory prefix.

    Returns:
        Submitted submitit Job, or None if archiving is disabled.
    """
    if not cfg.get("archive", False):
        return None

    # Build a minimal archive-specific executor config.
    # No slurm_partition — omitting it lets SLURM use the default CPU partition,
    # which is far easier to schedule than archive/prepost nodes.
    # --cpu-bind=none prevents the srun step from inheriting module-set CPU affinity
    # that conflicts with a single-CPU allocation.
    archive_cfg: dict[str, Any] = {
        **cfg,
        "setup": {
            "slurm_account": _ARCHIVE_ACCOUNT,
            "timeout_min": _ARCHIVE_TIMEOUT_MIN,
            "cpus_per_task": 1,
            "slurm_ntasks_per_node": 1,
            "name": job_name,
            "slurm_setup": _ARCHIVE_SETUP_COMMANDS,
            "slurm_srun_args": ["--cpu-bind=none"],
        },
    }

    executor = make_executor(archive_cfg, job_name)
    job = executor.submit(_create_tarball, src_path, tar_path)
    print(f"Archive job submitted (id={job.job_id}): {src_path} → {tar_path}")
    return job
