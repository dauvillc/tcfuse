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
    submitit=false +year=28 +basin=AL

# 2)  Everything in v01r01/final (≈1.6 TB - be sure you really want it!)
python scripts/preprocess/tc_primed/download_tc_primed.py \
    submitit=false +workers=32

# 3)  On Jean-Zay — submits to the prepost partition (has network access)
python scripts/preprocess/tc_primed/download_tc_primed.py \
    setup=jz_prepost +year=28 +basin=AL
"""

from __future__ import annotations

import os
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

BUCKET_NAME = "noaa-nesdis-tcprimed-pds"


def list_keys(prefix: str):
    """Recursively list object keys under `prefix` in the public bucket."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download_one(s3, key: str, size: int, dest_root: str, pbar: tqdm, prefix: str):
    """Download a single object unless it already exists locally at the same size."""
    local_path = os.path.join(dest_root, os.path.relpath(key, start=prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file exists and the size matches
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        pbar.update(size)
        return

    # Download with progress callback
    def callback(bytes_transferred):
        pbar.update(bytes_transferred)

    s3.download_file(BUCKET_NAME, key, local_path, Callback=callback)


class DownloadJob:
    """Submitit-compatible callable that downloads TC-PRIMED from S3."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def __call__(self) -> None:
        """Run the download and archive steps."""
        cfg = self.cfg

        # Destination root comes from the paths config
        tc_primed_root = Path(cfg["paths"]["raw_datasets"]["tc_primed"])

        # Optional overrides: year (0-indexed from 1987), basin, workers
        year: int | None = cfg.get("year", None)
        basin: str | None = cfg.get("basin", None)
        workers: int = int(cfg.get("workers", 8))

        # Validate basin requires year
        if basin is not None and year is None:
            raise ValueError("'basin' override requires 'year' to also be set.")

        # Construct the S3 prefix and local destination subdirectory
        if year is not None:
            absolute_year = 1987 + year
            prefix = f"v01r01/final/{absolute_year}/"
            dest_root = tc_primed_root / str(absolute_year)
        else:
            prefix = "v01r01/final/"
            dest_root = tc_primed_root

        if basin is not None:
            prefix = os.path.join(prefix, basin) + "/"
            dest_root = dest_root / basin

        dest_root_str = str(dest_root)
        print(f"Downloading from s3://{BUCKET_NAME}/{prefix} to {dest_root_str}/")

        # Anonymous (unsigned) S3 client
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # Find everything we're going to grab
        objects = list(list_keys(prefix))
        total_size = sum(size for _, size in objects)
        print(f"Found {len(objects):,} files - {total_size / 1e9:,.2f} GB.")

        # Multi-threaded download with a byte-level progress bar
        with tqdm(
            total=total_size, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(download_one, s3_client, key, size, dest_root_str, pbar, prefix)
                    for key, size in objects
                ]
                for future in as_completed(futures):
                    future.result()  # Raise any exceptions

        print("Done.")

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
