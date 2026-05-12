#!/usr/bin/env python3
"""Preprocess geostationary infrared data from TC-PRIMED into the standard HDF5 format.

Reads raw TC-PRIMED NetCDF4 overpass files, extracts infrared brightness temperature
(IRWIN) from the ``infrared`` group, and writes one HDF5 snapshot file per observation
in the project standard format. There are two IR sources depending on the value of
``infrared_availability_flag``:

- ``ir_tcirar`` (flag == 1): TC-IRAR dataset, 4 km resolution
- ``ir_hursat`` (flag == 2): HURSAT dataset, 8 km resolution

No regridding is applied; the IR data is already on a regular equiangular grid.
A consolidated index.parquet is written per source at the end.

Run from the project root:
    python scripts/preprocess/tc_primed/prepare_infrared.py [paths=jz] [submitit=false] [num_workers=4]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.preprocess.tc_primed.utils import (
    list_tc_primed_storm_files,
    should_skip_existing,
)
from tcfuse.data.sources import Source, SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

# Maps infrared_availability_flag value → source name (None = unavailable).
# Index 0 means no IR data; 1 = TC-IRAR (4 km); 2 = HURSAT (8 km).
IR_FLAG_TO_SOURCE: list[str | None] = [None, "ir_tcirar", "ir_hursat"]

# Scalar IFOV (km) per IR provenance dataset.
# IR data is already on a regular equiangular grid so a single value suffices.
IR_SOURCE_IFOVS: dict[str, float] = {
    "ir_tcirar": 4.0,
    "ir_hursat": 8.0,
}


def _read_ir_data(
    ir_grp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read IRWIN brightness temperature and spatial coordinates from an IR netCDF4 group.

    Handles both 1-D coordinate arrays (regular grid: lat shape (H,), lon shape (W,))
    and 2-D arrays (H, W) by broadcasting to a common (H, W) mesh when needed.
    Masked values are replaced with NaN; longitudes are normalised to [-180, 180].

    Args:
        ir_grp: netCDF4 group for the ``infrared`` key of a TC-PRIMED overpass file.

    Returns:
        (irwin, lat2d, lon2d) where each array has shape (H, W) and dtype float64.
        irwin contains brightness temperatures; NaN where masked.
    """
    irwin = np.ma.filled(ir_grp["IRWIN"][:].astype(float), np.nan)
    lat = np.ma.filled(ir_grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(ir_grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180

    # Squeeze any leading size-1 time dimension from IRWIN (e.g. (1, H, W) → (H, W))
    while irwin.ndim > 2 and irwin.shape[0] == 1:
        irwin = irwin[0]

    # Broadcast 1-D coord vectors to 2-D grids if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)  # lat: (H, W), lon: (H, W)

    return irwin, lat, lon


def process_ir_file(
    file: str | Path,
    sources_root: Path,
    skip_existing: bool = False,
    max_age_hours: float | None = None,
) -> dict[str, Any] | None:
    """Process one TC-PRIMED overpass file and write a standard HDF5 IR snapshot.

    Reads the ``infrared`` group, determines whether the source is TC-IRAR or HURSAT
    via ``infrared_availability_flag``, and writes a single FIELD source named
    ``ir_tcirar`` or ``ir_hursat`` to
    ``{sources_root}/{source_name}/snapshots/{storm_id}_{time}.h5``.

    When ``skip_existing`` is True, the output HDF5 is checked before reading image
    data: if it already exists (and is not older than ``max_age_hours`` when set),
    the index row is returned immediately without re-processing.

    Args:
        file: Path to the raw TC-PRIMED NetCDF4 overpass file.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        skip_existing: If True, skip files whose output snapshot already exists on disk
            and satisfies the age constraint. Default False.
        max_age_hours: Maximum age (hours) of an existing snapshot for it to be skipped.
            None means skip unconditionally when the file exists.

    Returns:
        Index row dict for inclusion in the per-source index.parquet, or None if discarded.
    """
    with Dataset(str(file)) as raw:
        # --- Overpass and storm metadata ---
        meta_grp = raw["overpass_metadata"]
        season = int(meta_grp["season"][0])
        basin = str(meta_grp["basin"][0])
        storm_number = int(meta_grp["cyclone_number"][-1])
        storm_id = f"{basin}{storm_number:02d}{season}"
        time_unix_s = float(meta_grp["time"][0])

        storm_grp = raw["overpass_storm_metadata"]
        storm_lat = float(storm_grp["storm_latitude"][0])
        storm_lon = (float(storm_grp["storm_longitude"][0]) + 180) % 360 - 180

        # --- Check IR availability ---
        if "infrared" not in raw.groups:
            return None
        ir_grp = raw["infrared"]
        flag = int(ir_grp["infrared_availability_flag"][0])
        source_name = IR_FLAG_TO_SOURCE[flag] if flag < len(IR_FLAG_TO_SOURCE) else None
        if source_name is None:
            return None

        # --- Early skip check (before reading image data) ---
        overpass_time = pd.Timestamp(time_unix_s, unit="s")
        overpass_time_utc = overpass_time.strftime("%Y%m%dT%H%M%SZ")
        dest_path = Source.path(sources_root, source_name, storm_id, overpass_time_utc)
        if should_skip_existing(dest_path, skip_existing, max_age_hours):
            return {
                "storm_id": storm_id,
                "basin": basin,
                "snapshot_time_utc": overpass_time.isoformat(),
                "lat": storm_lat,
                "lon": storm_lon,
                "source_name": source_name,
                "file_path": str(dest_path),
            }

        # --- Read IR data ---
        irwin, lat2d, lon2d = _read_ir_data(ir_grp)

    # Discard if IRWIN is entirely NaN (closed Dataset before this check)
    if np.all(np.isnan(irwin)):
        return None

    # --- Build Source tensors ---
    h, w = irwin.shape
    values_np = irwin[..., np.newaxis].astype(np.float32)  # (H, W, 1)
    time_broadcast = np.full((h, w), time_unix_s, dtype=np.float32)
    coords_np = np.stack(
        [time_broadcast, lat2d.astype(np.float32), lon2d.astype(np.float32)], axis=-1
    )  # (H, W, 3)
    mask_np = ~np.isnan(irwin)  # (H, W)

    meta: dict[str, Any] = {
        "storm_id": storm_id,
        "basin": basin,
        "snapshot_time_utc": overpass_time.isoformat(),
        "lat": storm_lat,
        "lon": storm_lon,
    }
    # Scalar IFOV (km); no along/across distinction since data is already gridded.
    char_vars: dict[str, Any] = {"ifov": {"irwin": IR_SOURCE_IFOVS[source_name]}}
    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=source_name,
        channels=["irwin"],
        mask=torch.from_numpy(mask_np),
        meta=meta,
        char_vars=char_vars,
    )
    source.write(dest_path)

    return {
        "storm_id": storm_id,
        "basin": basin,
        "snapshot_time_utc": overpass_time.isoformat(),
        "lat": storm_lat,
        "lon": storm_lon,
        "source_name": source_name,
        "file_path": str(dest_path),
    }


def _process_all_files(
    files: list[Path],
    sources_root: Path,
    num_workers: int,
    skip_existing: bool = False,
    max_age_hours: float | None = None,
) -> list[dict[str, Any] | None]:
    """Process all overpass files for IR extraction using ProcessPoolExecutor.

    Designed to be submitted as a single submitit job or called directly
    for local execution.

    Args:
        files: Raw TC-PRIMED overpass file paths to process.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        num_workers: Number of parallel worker processes.
        skip_existing: Forwarded to ``process_ir_file``.
        max_age_hours: Forwarded to ``process_ir_file``.

    Returns:
        List of index row dicts (None entries for discarded snapshots).
    """
    if num_workers <= 1:
        return [
            process_ir_file(f, sources_root, skip_existing, max_age_hours)
            for f in tqdm(files, desc="ir")
        ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    process_ir_file,
                    files,
                    repeat(sources_root),
                    repeat(skip_existing),
                    repeat(max_age_hours),
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc="ir",
            )
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED IR snapshots to the standard HDF5 format."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])

    # Collect a flat, deduplicated list of all overpass files across all storms.
    overpass_files_by_storm, _ = list_tc_primed_storm_files(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )
    all_files: list[Path] = list(
        {f for f in chain.from_iterable(overpass_files_by_storm.values())}
    )
    print(f"Found {len(all_files)} overpass files.")

    launch_local = not bool(cfg.get("submitit", False))
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))
    max_age_hours: float | None = cfg.get("max_age_hours", None)
    if max_age_hours is not None:
        max_age_hours = float(max_age_hours)

    if launch_local:
        results: list[dict[str, Any] | None] = _process_all_files(
            all_files, sources_root, num_workers, skip_existing, max_age_hours
        )
    else:
        from tcfuse.utils.submitit_utils import make_executor as make_submitit_executor

        slurm_executor = make_submitit_executor(cfg, "prepare_infrared")
        print(f"Submitting IR job ({len(all_files)} files)…")
        job = slurm_executor.submit(
            _process_all_files, all_files, sources_root, num_workers, skip_existing, max_age_hours
        )
        results = job.result()

    valid = [r for r in results if r is not None]
    discarded = len(results) - len(valid)
    if discarded:
        pct = 100.0 * discarded / max(len(results), 1)
        print(f"Discarded {discarded}/{len(results)} ({pct:.1f}%)")

    # Split rows by source name and write one index.parquet per source.
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        rows_by_source.setdefault(row["source_name"], []).append(row)

    if rows_by_source:
        for source_name, rows in rows_by_source.items():
            index_df = pd.DataFrame(rows)
            index_path = sources_root / source_name / "index.parquet"
            index_df.to_parquet(index_path, index=False)
            print(f"Wrote index ({source_name}): {len(index_df)} rows → {index_path}")
            source_meta = SourceMetadata(
                source_name,
                "infrared",
                SourceKind.FIELD,
                ["irwin"],
                index_df,
                char_vars={"ifov": {"irwin": IR_SOURCE_IFOVS[source_name]}},
            )
            source_meta.write(sources_root)
            # Archive this source's directory to STORE as a per-source tarball.
            tar_path = (
                Path(cfg["paths"]["archives"]["preprocessed_sources"])
                / f"{source_name}.tar.gz"
            )
            submit_archive_job(
                sources_root / source_name, tar_path, cfg, job_name=f"archive_{source_name}"
            )
    else:
        print("No valid IR snapshots found.")


if __name__ == "__main__":
    main()
