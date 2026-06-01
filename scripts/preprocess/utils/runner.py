"""Shared launch and finalize helpers for Stage 1 per-source preprocessing scripts.

The per-source ``index.parquet`` is rebuilt by scanning the written HDF5
snapshots at the end of every run; workers therefore no longer need to emit
index rows themselves. Each row carries the IBTrACS SID (from the file root
attrs) plus ``season / basin / subbasin`` looked up in the Stage 0 translation
table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import h5py
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.data.ibtracs import load_atcf_to_sid
from tcfuse.data.sources import SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

# Canonical column set for every Stage 1 per-source index.parquet.
INDEX_COLUMNS: tuple[str, ...] = (
    "sid",
    "source_name",
    "snapshot_time_utc",
    "season",
    "basin",
    "subbasin",
)


def resolve_preproc_cfg(raw_cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra preproc config to a plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))


def load_translation(sources_root: Path) -> dict[str, str]:
    """Load the ATCF→SID translation dict from Stage 0 outputs.

    Args:
        sources_root: Root directory for preprocessed sources.

    Returns:
        Mapping ``{usa_atcf_id: sid}`` for every main-track storm in IBTrACS.
    """
    from tcfuse.data.ibtracs import load_atcf_to_sid_dict

    table = load_atcf_to_sid_dict(sources_root)
    print(
        f"Loaded ATCF→SID translation: {len(table)} entries "
        f"from {sources_root / 'ibtracs' / 'atcf_to_sid.csv'}"
    )
    return table


def _sid_attrs_lookup(sources_root: Path) -> dict[str, dict[str, Any]]:
    """Return ``{sid: {season, basin, subbasin}}`` from the Stage 0 translation table."""
    df = load_atcf_to_sid(sources_root)
    keep = cast(pd.DataFrame, df[["sid", "season", "basin", "subbasin"]])
    keep = cast(pd.DataFrame, keep.drop_duplicates(subset=["sid"]))
    records: dict[str, dict[str, Any]] = {}
    for rec in cast(list[dict[str, Any]], keep.to_dict(orient="records")):
        records[str(rec["sid"])] = {
            "season": int(rec["season"]),
            "basin": str(rec["basin"]),
            "subbasin": str(rec["subbasin"]),
        }
    return records


def scan_source_snapshots(
    source_dir: Path,
    source_name: str,
    sid_attrs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Rebuild the per-source index by scanning written snapshot HDF5 files.

    Args:
        source_dir: Source directory, e.g. ``{sources_root}/pmw_amsr2_gcomw1``.
        source_name: Source identifier.
        sid_attrs: Mapping from SID to ``{season, basin, subbasin}`` produced by
            :func:`_sid_attrs_lookup`.

    Returns:
        DataFrame with the columns listed in :data:`INDEX_COLUMNS`. Snapshots
        whose ``storm_id`` attr is not a known SID are dropped with a warning.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    snapshots_dir = source_dir / "snapshots"
    for path in sorted(snapshots_dir.glob("*.h5")) if snapshots_dir.is_dir() else []:
        with h5py.File(path, "r") as snapshot_file:
            attrs = dict(snapshot_file.attrs)
        sid = str(attrs["storm_id"])
        info = sid_attrs.get(sid)
        # Stale snapshots from interrupted runs may reference unknown SIDs.
        if info is None:
            skipped.append(sid)
            continue
        rows.append(
            {
                "sid": sid,
                "source_name": source_name,
                "snapshot_time_utc": str(attrs["snapshot_time_utc"]),
                "season": info["season"],
                "basin": info["basin"],
                "subbasin": info["subbasin"],
            }
        )

    if skipped:
        unique = sorted(set(skipped))
        print(
            f"[WARN] {source_name}: skipped {len(skipped)} snapshots for "
            f"{len(unique)} unknown SID(s) (e.g. {unique[:3]})."
        )

    if not rows:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


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
                    # Larger chunks amortize process-pool scheduling overhead.
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc=desc,
            )
        )


def launch_local_or_slurm[T](
    cfg: dict[str, Any],
    job_name: str,
    fn: Callable[[], T],
    slurm_fn: Callable[[], T] | None = None,
) -> T:
    """Run locally or submit a single SLURM job via submitit."""
    run_fn = slurm_fn if slurm_fn is not None else fn
    if not bool(cfg.get("submitit", False)):
        return fn()

    from tcfuse.utils.submitit_utils import make_executor

    executor = make_executor(cfg, job_name)
    job = executor.submit(run_fn)
    return job.result()


def finalize_source(
    source_name: str,
    source_type: str,
    kind: SourceKind,
    channels: list[str],
    shape: tuple[int, ...],
    sources_root: Path,
    cfg: dict[str, Any],
    char_vars: dict[str, Any] | None = None,
) -> int:
    """Build the per-source index from on-disk snapshots and archive the source.

    Reads the Stage 0 translation table once to populate ``season/basin/subbasin``
    on every index row. Writes ``metadata.yaml`` via :meth:`SourceMetadata.to_yaml`
    and ``index.parquet`` separately under ``{sources_root}/{source_name}/``.

    Args:
        source_name: Source identifier, e.g. ``"pmw_amsr2_gcomw1"``.
        source_type: Physical category, e.g. ``"microwave"``.
        kind: Dimensionality class of the source.
        channels: Ordered channel names.
        shape: Spatial shape shared by every snapshot of this source
            (excluding channels). Use ``()`` for SCALAR, ``(L,)`` for
            PROFILE, ``(H, W)`` for FIELD.
        sources_root: Root directory for preprocessed sources.
        cfg: Resolved preproc config dict.
        char_vars: Optional instrument-level descriptor variables.

    Returns:
        The number of indexed snapshots (0 when nothing was written).
    """
    source_dir = sources_root / source_name
    sid_attrs = _sid_attrs_lookup(sources_root)
    index_df = scan_source_snapshots(source_dir, source_name, sid_attrs)
    if index_df.empty:
        return 0

    source_meta = SourceMetadata(
        source_name,
        source_type,
        kind,
        channels,
        shape,
        char_vars=char_vars or {},
    )
    source_meta.to_yaml(source_dir / "metadata.yaml")
    index_df.to_parquet(source_dir / "index.parquet", index=False)
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
