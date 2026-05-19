"""Shared launch and finalize helpers for preprocessing scripts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.data.sources import SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

INDEX_COLUMNS = (
    "storm_id",
    "snapshot_time_utc",
    "lat",
    "lon",
    "source_name",
    "file_path",
)


def resolve_preproc_cfg(raw_cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra preproc config to a plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))


def make_index_row(
    storm_id: str,
    snapshot_time_utc: str,
    lat: float,
    lon: float,
    source_name: str,
    file_path: Path | str,
) -> dict[str, Any]:
    """Build one row for a per-source ``index.parquet``."""
    return {
        "storm_id": storm_id,
        "snapshot_time_utc": snapshot_time_utc,
        "lat": lat,
        "lon": lon,
        "source_name": source_name,
        "file_path": str(file_path),
    }


def scan_snapshots_index(source_dir: Path, source_name: str) -> pd.DataFrame:
    """Rebuild the per-source index by scanning written snapshot HDF5 files."""
    snapshots_dir = source_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    rows: list[dict[str, Any]] = []
    for path in sorted(snapshots_dir.glob("*.h5")):
        with h5py.File(path, "r") as snapshot_file:
            attrs = dict(snapshot_file.attrs)
        rows.append(
            make_index_row(
                str(attrs["storm_id"]),
                str(attrs["snapshot_time_utc"]),
                float(np.asarray(attrs["lat"]).item()),
                float(np.asarray(attrs["lon"]).item()),
                source_name,
                path,
            )
        )

    if not rows:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    return pd.DataFrame(rows)


def map_files[R, F](
    worker: Callable[..., R | None],
    files: Sequence[F],
    *static_args: object,
    num_workers: int,
    desc: str,
) -> list[R | None]:
    """Map ``worker`` over files with optional multiprocessing.

    ``*static_args`` are broadcast to every file (not zipped per file). Per-file
    metadata must be bundled into ``files`` (e.g. ``zip(paths, infos)``).
    """
    if num_workers <= 1:
        return [worker(file, *static_args) for file in tqdm(files, desc=desc)]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    worker,
                    files,
                    *[repeat(arg) for arg in static_args],
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc=desc,
            )
        )


def launch_local_or_slurm[T](
    cfg: dict[str, Any],
    job_name: str,
    local_fn: Callable[[], T],
    slurm_fn: Callable[[], T],
) -> T:
    """Run locally or submit a single SLURM job via submitit."""
    if not bool(cfg.get("submitit", False)):
        return local_fn()

    from tcfuse.utils.submitit_utils import make_executor

    executor = make_executor(cfg, job_name)
    job = executor.submit(slurm_fn)
    return job.result()


def finalize_source(
    source_name: str,
    source_type: str,
    kind: SourceKind,
    channels: list[str],
    sources_root: Path,
    cfg: dict[str, Any],
    char_vars: dict[str, Any] | None = None,
) -> int:
    """Write index/metadata from on-disk snapshots and archive the source directory."""
    source_dir = sources_root / source_name
    index_df = scan_snapshots_index(source_dir, source_name)
    if index_df.empty:
        return 0

    source_meta = SourceMetadata(
        source_name,
        source_type,
        kind,
        channels,
        index_df,
        char_vars=char_vars or {},
    )
    source_meta.write(sources_root)
    print(f"Wrote index ({source_name}): {len(index_df)} rows → {source_dir / 'index.parquet'}")

    tar_path = Path(cfg["paths"]["archives"]["preprocessed_sources"]) / f"{source_name}.tar.gz"
    submit_archive_job(source_dir, tar_path, cfg, job_name=f"archive_{source_name}")
    return len(index_df)


def submit_slurm_jobs(
    cfg: dict[str, Any],
    job_name: str,
    tasks: Iterable[tuple[str, Callable[..., Any], tuple[Any, ...]]],
) -> dict[str, Any]:
    """Submit multiple submitit jobs and return results keyed by task label."""
    from tcfuse.utils.submitit_utils import make_executor

    executor = make_executor(cfg, job_name)
    jobs = {label: executor.submit(fn, *args) for label, fn, args in tasks}
    return {label: job.result() for label, job in tqdm(jobs.items(), desc="collecting results")}
