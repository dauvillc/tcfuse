#!/usr/bin/env python3
"""Preprocess C-band SAR wind speed data from CyclObs into the standard HDF5 format.

Reads the CyclObs SAR acquisition metadata CSV, locates each NetCDF overpass file,
extracts the wind_speed field, and writes one HDF5 snapshot per observation in the
project standard format. A consolidated index.parquet and metadata.yaml are written
at the end.

Run from the project root:
    python scripts/preprocess/sar/prepare_sar.py [paths=local] [submitit=false] [num_workers=4]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.data.sources import Source, SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

SOURCE_NAME = "sar_cband"
CHANNELS = ["wind_speed"]


def process_sar_file(
    file: str | Path,
    file_info: dict[str, Any],
    sources_root: Path,
) -> dict[str, Any] | None:
    """Process one CyclObs SAR overpass file and write a standard HDF5 snapshot.

    Extracts the wind_speed field, builds a 2D coordinate grid from the 1D lat/lon
    arrays, applies the validity mask (mask_flag == 0), and writes a single FIELD
    source named ``sar_cband``.

    The snapshot is written to
    ``{sources_root}/sar_cband/snapshots/{storm_id}_{time}.h5``.

    Args:
        file: Path to the CyclObs SAR NetCDF file.
        file_info: Row from the acquisition metadata CSV as a plain dict.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).

    Returns:
        Index row dict for inclusion in the per-source index.parquet, or None if discarded.
    """
    with Dataset(str(file)) as raw:
        # --- Read wind speed (channel dim at index 0) and validity mask ---
        wind_speed = np.ma.filled(raw["wind_speed"][:].astype(float), np.nan)
        # wind_speed may be (1, H, W) or (H, W); normalise to (H, W)
        if wind_speed.ndim == 3:
            wind_speed = wind_speed[0]
        mask_flag = np.array(raw["mask_flag"][:])
        if mask_flag.ndim == 3:
            mask_flag = mask_flag[0]

        # --- 1D lat/lon → 2D meshgrid ---
        lat_1d = np.array(raw["lat"][:], dtype=np.float32)
        lon_1d = np.array(raw["lon"][:], dtype=np.float32)

    # Normalise longitudes to [-180, 180]
    lon_1d = (lon_1d + 180) % 360 - 180

    # Skip if all wind speed values are missing
    if np.all(np.isnan(wind_speed)):
        return None

    # --- Storm and time metadata ---
    storm_id: str = file_info["sid"]  # already in ATCF format (reformatted in main)
    basin: str = file_info["basin"]
    season: str = file_info["season"]
    acq_time: pd.Timestamp = pd.Timestamp(file_info["acquisition_start_time"])
    time_unix_s = float(acq_time.timestamp())
    storm_center = file_info["track_point"]  # WKT string
    from shapely.geometry import Point
    from shapely.wkt import loads as wkt_loads

    center = cast(Point, wkt_loads(storm_center))
    storm_lat, storm_lon = float(center.y), float(center.x)
    storm_lon = (storm_lon + 180) % 360 - 180
    # Convert wind speed from m/s to knots
    vmax_kt = float(file_info["vmax (m/s)"]) * 1.94384

    # --- Build 2D arrays ---
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)  # both (H, W)
    h, w = lat_2d.shape

    # values: (H, W, 1) float32
    values_np = wind_speed[:, :, np.newaxis].astype(np.float32)

    # mask: True where valid (mask_flag == 0) and wind_speed is not NaN
    mask_np = (mask_flag == 0) & ~np.isnan(wind_speed)  # (H, W)

    # coords: (H, W, 3) = [time_unix_s (broadcast), lat, lon]
    time_broadcast = np.full((h, w), time_unix_s, dtype=np.float32)
    coords_np = np.stack([time_broadcast, lat_2d, lon_2d], axis=-1)  # (H, W, 3)

    # --- Build Source and write to disk ---
    snapshot_time_utc = acq_time.strftime("%Y%m%dT%H%M%SZ")
    dest_path = Source.path(sources_root, SOURCE_NAME, storm_id, snapshot_time_utc)
    meta: dict[str, Any] = {
        "storm_id": storm_id,
        "basin": basin,
        "season": season,
        "snapshot_time_utc": acq_time.isoformat(),
        "lat": storm_lat,
        "lon": storm_lon,
        "vmax_kt": vmax_kt,
    }
    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=SOURCE_NAME,
        channels=CHANNELS,
        mask=torch.from_numpy(mask_np),
        meta=meta,
    )
    source.write(dest_path)

    return {
        **meta,
        "source_name": SOURCE_NAME,
        "file_path": str(dest_path),
    }


def _process_all_files(
    files: list[Path],
    file_infos: list[dict[str, Any]],
    sources_root: Path,
    num_workers: int,
) -> list[dict[str, Any] | None]:
    """Process all SAR overpass files using ProcessPoolExecutor or sequentially.

    Designed to be submitted as a single submitit job or called directly
    for local execution.

    Args:
        files: SAR NetCDF file paths to process.
        file_infos: Corresponding acquisition metadata rows as plain dicts.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        num_workers: Number of parallel worker processes.

    Returns:
        List of index row dicts (None entries for discarded snapshots).
    """
    if num_workers <= 1:
        return [
            process_sar_file(f, info, sources_root)
            for f, info in tqdm(zip(files, file_infos), total=len(files), desc=SOURCE_NAME)
        ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    process_sar_file,
                    files,
                    file_infos,
                    repeat(sources_root),
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc=SOURCE_NAME,
            )
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all CyclObs SAR snapshots to the standard HDF5 format."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    cyclobs_dir = Path(cfg["paths"]["raw_datasets"]["cyclobs"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    include_seasons = cfg.get("include_seasons")
    launch_local = not bool(cfg.get("submitit", False))
    num_workers = int(cfg.get("num_workers", 4))

    # Load acquisition metadata and reformat the sid to ATCF format (BBNNYYYY)
    acq_df = pd.read_csv(
        cyclobs_dir / "sar_acquisitions_metadata.csv",
        parse_dates=["acquisition_start_time"],
    )
    # CyclObs sid format: bbNNYYYY (e.g. "al022024") → BBNNYYYY (e.g. "AL022024")
    acq_df["season"] = acq_df["sid"].str[-4:]
    acq_df["basin"] = acq_df["sid"].str[:2].str.upper()
    acq_df["storm_number"] = acq_df["sid"].str[2:4].astype(int)
    acq_df["sid"] = acq_df.apply(
        lambda r: f"{r['basin']}{r['storm_number']:02d}{r['season']}", axis=1
    )

    # Optionally filter to specific seasons
    if include_seasons is not None:
        acq_df = acq_df[acq_df["season"].isin(include_seasons)].reset_index(drop=True)
        print(f"Filtered to seasons {include_seasons}: {len(acq_df)} acquisitions.")

    files = [cyclobs_dir / url.split("/")[-1] for url in acq_df["data_url"]]
    # Convert to plain dicts so rows are picklable by ProcessPoolExecutor
    file_infos: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in acq_df.to_dict("records")
    ]

    print(f"Processing {len(files)} SAR acquisitions…")

    # Launch processing locally or via submitit
    if launch_local:
        results = _process_all_files(files, file_infos, sources_root, num_workers)
    else:
        from tcfuse.utils.submitit_utils import make_executor as make_submitit_executor

        executor = make_submitit_executor(cfg, "prepare_sar")
        job = executor.submit(_process_all_files, files, file_infos, sources_root, num_workers)
        results = job.result()

    valid = [r for r in results if r is not None]
    discarded = len(results) - len(valid)
    if discarded:
        pct = 100.0 * discarded / max(len(results), 1)
        print(f"Discarded {discarded}/{len(results)} ({pct:.1f}%)")

    if valid:
        # Write per-source index.parquet and metadata.yaml
        index_df = pd.DataFrame(valid)
        index_path = sources_root / SOURCE_NAME / "index.parquet"
        index_df.to_parquet(index_path, index=False)
        print(f"Wrote index: {len(index_df)} rows → {index_path}")
        source_meta = SourceMetadata(SOURCE_NAME, "sar", SourceKind.FIELD, CHANNELS, index_df)
        source_meta.write(sources_root)
        # Archive to STORE
        tar_path = Path(cfg["paths"]["archives"]["preprocessed_sources"]) / f"{SOURCE_NAME}.tar.gz"
        submit_archive_job(
            sources_root / SOURCE_NAME, tar_path, cfg, job_name=f"archive_{SOURCE_NAME}"
        )
    else:
        print("No valid SAR snapshots found.")


if __name__ == "__main__":
    main()
