#!/usr/bin/env python3
"""Preprocess C-band SAR wind speed data from CyclObs into the standard HDF5 format."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig
from scripts.preprocess.tc_primed.utils import should_skip_existing
from scripts.preprocess.utils.runner import (
    finalize_source,
    launch_local_or_slurm,
    make_index_row,
    map_files,
    resolve_preproc_cfg,
)
from tcfuse.data.sources import Source, SourceKind

SOURCE_NAME = "sar_cband"
CHANNELS = ["wind_speed"]
_WKT_POINT = re.compile(r"POINT\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", re.IGNORECASE)


def parse_wkt_point(wkt: str) -> tuple[float, float]:
    """Parse a WKT POINT string as ``(lat, lon)`` in degrees."""
    match = _WKT_POINT.match(wkt.strip())
    if match is None:
        raise ValueError(f"Unsupported WKT geometry: {wkt!r}")
    lon, lat = float(match.group(1)), float(match.group(2))
    return lat, lon


def process_sar_file(
    file: str | Path,
    file_info: dict[str, Any],
    sources_root: Path,
    skip_existing: bool = False,
) -> dict[str, Any] | None:
    """Process one CyclObs SAR overpass file and write a standard HDF5 snapshot."""
    storm_id = str(file_info["sid"])
    basin = str(file_info["basin"])
    acq_time = cast(pd.Timestamp, pd.Timestamp(file_info["acquisition_start_time"]))
    time_unix_s = float(acq_time.timestamp())
    storm_lat, storm_lon = parse_wkt_point(str(file_info["track_point"]))
    storm_lon = (storm_lon + 180) % 360 - 180

    snapshot_time_utc = acq_time.strftime("%Y%m%dT%H%M%SZ")
    dest_path = Source.path(sources_root, SOURCE_NAME, storm_id, snapshot_time_utc)
    if should_skip_existing(dest_path, skip_existing):
        return None

    with Dataset(str(file)) as raw:
        wind_speed = np.ma.filled(raw["wind_speed"][:].astype(float), np.nan)
        if wind_speed.ndim == 3:
            wind_speed = wind_speed[0]
        mask_flag = np.array(raw["mask_flag"][:])
        if mask_flag.ndim == 3:
            mask_flag = mask_flag[0]
        lat_1d = np.array(raw["lat"][:], dtype=np.float32)
        lon_1d = np.array(raw["lon"][:], dtype=np.float32)

    lon_1d = (lon_1d + 180) % 360 - 180
    wind_speed = np.asarray(wind_speed, dtype=np.float32)
    wind_speed[mask_flag != 0] = np.nan
    if np.all(np.isnan(wind_speed)):
        return None

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    h, w = lat_2d.shape
    values_np = wind_speed[:, :, np.newaxis].astype(np.float32)
    mask_np = np.isfinite(values_np)
    time_broadcast = np.full((h, w), time_unix_s, dtype=np.float32)
    coords_np = np.stack([time_broadcast, lat_2d, lon_2d], axis=-1)

    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=SOURCE_NAME,
        channels=CHANNELS,
        mask=torch.from_numpy(mask_np),
        meta={
            "storm_id": storm_id,
            "basin": basin,
            "snapshot_time_utc": acq_time.isoformat(),
            "lat": storm_lat,
            "lon": storm_lon,
        },
    )
    source.write(dest_path)

    return make_index_row(
        storm_id,
        acq_time.isoformat(),
        storm_lat,
        storm_lon,
        SOURCE_NAME,
        dest_path,
    )


def _process_sar_item(
    item: tuple[Path, dict[str, Any]],
    sources_root: Path,
    skip_existing: bool = False,
) -> dict[str, Any] | None:
    """Process one ``(file, metadata)`` pair (picklable entry point for ``map_files``)."""
    file, file_info = item
    return process_sar_file(file, file_info, sources_root, skip_existing)


def _process_all_files(
    files: list[Path],
    file_infos: list[dict[str, Any]],
    sources_root: Path,
    num_workers: int,
    skip_existing: bool,
) -> None:
    """Process all SAR overpass files."""
    items = list(zip(files, file_infos, strict=True))
    map_files(
        _process_sar_item,
        items,
        sources_root,
        skip_existing,
        num_workers=num_workers,
        desc=SOURCE_NAME,
    )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all CyclObs SAR snapshots to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    cyclobs_dir = Path(cfg["paths"]["raw_datasets"]["cyclobs"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    acq_df = pd.read_csv(
        cyclobs_dir / "sar_acquisitions_metadata.csv",
        parse_dates=["acquisition_start_time"],
    )
    acq_df["season"] = acq_df["sid"].str[-4:]
    acq_df["basin"] = acq_df["sid"].str[:2].str.upper()
    acq_df["storm_number"] = acq_df["sid"].str[2:4].astype(int)
    acq_df["sid"] = acq_df.apply(
        lambda row: f"{row['basin']}{row['storm_number']:02d}{row['season']}", axis=1
    )

    include_seasons = cfg.get("include_seasons")
    if include_seasons is not None:
        acq_df = acq_df[acq_df["season"].isin(include_seasons)].reset_index(drop=True)
        print(f"Filtered to seasons {include_seasons}: {len(acq_df)} acquisitions.")

    keep_cols = ["sid", "basin", "acquisition_start_time", "track_point"]
    subset = cast(pd.DataFrame, acq_df[keep_cols])
    file_infos = cast(list[dict[str, Any]], subset.to_dict(orient="records"))
    files = [cyclobs_dir / url.split("/")[-1] for url in acq_df["data_url"]]
    print(f"Processing {len(files)} SAR acquisitions…")

    launch_local_or_slurm(
        cfg,
        "prepare_sar",
        lambda: _process_all_files(files, file_infos, sources_root, num_workers, skip_existing),
        lambda: _process_all_files(files, file_infos, sources_root, num_workers, skip_existing),
    )

    if finalize_source(SOURCE_NAME, "sar", SourceKind.FIELD, CHANNELS, sources_root, cfg) == 0:
        print("No valid SAR snapshots found.")


if __name__ == "__main__":
    main()
