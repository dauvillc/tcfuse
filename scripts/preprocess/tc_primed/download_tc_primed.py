#!/usr/bin/env python3
"""
download_tcprimed.py
--------------------

Bulk-download files from the NOAA TC-PRIMED public S3 bucket.
Destination is read from paths.raw_datasets.tc_primed in the Hydra config.

Examples
========
# 1)  All 2015 Atlantic storms (local, runs directly)
python scripts/preprocess/tc_primed/download_tc_primed.py \
    submitit=false +season=2015 +basin=AL

# 2)  Everything in v01r01/final (≈1.6 TB - be sure you really want it!)
python scripts/preprocess/tc_primed/download_tc_primed.py \
    submitit=false +workers=32

# 3)  On Jean-Zay — submits to the prepost partition (has network access)
python scripts/preprocess/tc_primed/download_tc_primed.py \
    setup=jz_prepost +season=2015 +basin=AL
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import boto3
import hydra
from botocore import UNSIGNED
from botocore.client import Config
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.utils.archive import submit_archive_job
from tcfuse.utils.submitit_utils import make_executor

def _s3_client() -> Any:
    # One client per thread — avoids connection-pool exhaustion; stored on the thread
    # object directly so it is never captured by cloudpickle when submitit serializes the job.
    t = threading.current_thread()
    if not hasattr(t, "_s3_client"):
        t._s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))  # type: ignore[attr-defined]
    return t._s3_client  # type: ignore[attr-defined]


BUCKET_NAME = "noaa-nesdis-tcprimed-pds"


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """List all object keys under `prefix` in the public bucket.

    Shows a live count while paginating so the user can see progress
    even before the download starts (listing can be slow for large prefixes).

    Returns:
        List of (key, size_bytes) tuples.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    objects: list[tuple[str, int]] = []
    # Live counter — S3 listing can take tens of seconds for large prefixes
    with tqdm(desc="Listing objects", unit=" files", leave=True) as pbar:
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append((obj["Key"], obj["Size"]))
                pbar.update(1)
    return objects


def download_one(
    key: str,
    size: int,
    dest_root: str,
    byte_pbar: tqdm,
    file_pbar: tqdm,
    prefix: str,
) -> str:
    """Download a single object unless it already exists locally at the same size.

    Args:
        key: S3 object key.
        size: Expected size in bytes (used to detect already-complete downloads).
        dest_root: Local directory root for the download tree.
        byte_pbar: tqdm bar tracking total bytes transferred.
        file_pbar: tqdm bar tracking total files completed.
        prefix: S3 prefix stripped when building the local relative path.

    Returns:
        ``"skipped"`` if the file was already present, ``"downloaded"`` otherwise.
    """
    # Derive local path by stripping the S3 prefix
    local_path = os.path.join(dest_root, os.path.relpath(key, start=prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file exists and the size matches (idempotent re-runs)
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        byte_pbar.update(size)
        file_pbar.update(1)
        return "skipped"

    # Stream bytes from S3, forwarding each chunk to the byte progress bar
    def callback(bytes_transferred: int) -> None:
        byte_pbar.update(bytes_transferred)

    _s3_client().download_file(BUCKET_NAME, key, local_path, Callback=callback)
    file_pbar.update(1)
    return "downloaded"


class DownloadJob:
    """Submitit-compatible callable that downloads TC-PRIMED from S3."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def __call__(self) -> None:
        """Run the download and archive steps."""
        cfg = self.cfg

        # Destination root comes from the paths config
        tc_primed_root = Path(cfg["paths"]["raw_datasets"]["tc_primed"])

        # Optional overrides: season (calendar year, e.g. 2015), basin, workers
        season: int | None = cfg.get("season", None)
        basin: str | None = cfg.get("basin", None)
        workers: int = int(cfg.get("workers", 8))

        # Validate basin requires season
        if basin is not None and season is None:
            raise ValueError("'basin' override requires 'season' to also be set.")

        # Construct the S3 prefix and local destination subdirectory
        if season is not None:
            prefix = f"v01r01/final/{season}/"
            dest_root = tc_primed_root / str(season)
        else:
            prefix = "v01r01/final/"
            dest_root = tc_primed_root

        if basin is not None:
            prefix = os.path.join(prefix, basin) + "/"
            dest_root = dest_root / basin

        dest_root_str = str(dest_root)

        # Print job parameters so the log is self-documenting
        print(f"Source : s3://{BUCKET_NAME}/{prefix}")
        print(f"Dest   : {dest_root_str}/")
        print(f"Workers: {workers}")

        # List all objects (shows a live counter while paginating)
        objects = list_keys(prefix)
        total_files = len(objects)
        total_size = sum(size for _, size in objects)
        print(f"Found {total_files:,} files — {total_size / 1e9:,.2f} GB total.")

        # Nothing to do if the prefix matched no objects
        if total_files == 0:
            print("Nothing to download.")
            return

        # Thread-safe counters for the final summary
        counters: dict[str, int] = {"downloaded": 0, "skipped": 0, "errors": 0}
        error_messages: list[str] = []
        lock = threading.Lock()

        t0 = time.monotonic()

        # Two stacked progress bars: byte throughput (top) and file count (bottom)
        with (
            tqdm(
                total=total_size,
                desc="Bytes  ",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                position=0,
            ) as byte_pbar,
            tqdm(
                total=total_files,
                desc="Files  ",
                unit=" files",
                position=1,
            ) as file_pbar,
        ):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Map each future back to its key for error reporting
                future_to_key = {
                    pool.submit(
                        download_one,
                        key,
                        size,
                        dest_root_str,
                        byte_pbar,
                        file_pbar,
                        prefix,
                    ): key
                    for key, size in objects
                }

                # Collect results as futures complete; continue on per-file errors
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        status = future.result()
                        with lock:
                            counters[status] += 1
                    except Exception as exc:
                        # Advance the file bar even for failures so counts stay consistent
                        file_pbar.update(1)
                        with lock:
                            counters["errors"] += 1
                            error_messages.append(f"{key}: {exc}")

        elapsed = time.monotonic() - t0

        # Print final summary with per-status counts and total elapsed time
        print(
            f"\nFinished in {elapsed:.1f}s — "
            f"{counters['downloaded']:,} downloaded, "
            f"{counters['skipped']:,} skipped (already present), "
            f"{counters['errors']:,} errors."
        )

        # Print individual error details if any occurred
        if error_messages:
            print("Failed files:")
            for msg in error_messages:
                print(f"  {msg}")

        # Archive the raw TC-PRIMED directory to STORE as a tarball
        submit_archive_job(
            tc_primed_root,
            Path(cfg["paths"]["archives"]["raw_tc_primed"]),
            cfg,
            job_name="archive_raw_tc_primed",
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Download TC-PRIMED data to the path configured in paths.raw_datasets.tc_primed."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    job = DownloadJob(cfg)

    # submitit=false → run directly (local debug); otherwise submit to SLURM
    launch_local = not bool(cfg.get("submitit", False))
    if launch_local:
        job()
    else:
        executor = make_executor(cfg, "download_tc_primed")
        submitted = executor.submit(job)
        print(f"Submitted job {submitted.job_id}")


if __name__ == "__main__":
    main()
