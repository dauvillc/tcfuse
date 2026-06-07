#!/usr/bin/env python3
"""Stage 1 — preprocess C-band SAR wind speed snapshots from CyclObs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.utils.field_grid import center_crop_or_pad_2d
from scripts.preprocess.utils.runner import (
    finalize_source,
    launch_local_or_slurm,
    load_translation,
    map_files,
    resolve_preproc_cfg,
)
from tcfuse.data.sources import Source, SourceKind
from tcfuse.utils.time import to_compact_time

SOURCE_NAME = "sar_cband"
CHANNELS = ["wind_speed"]
# Storm-centered square crop on the native regular grid (no regridding).
SAR_CENTER_CROP_HALF_WIDTH_PX = 200


def process_sar_file(
    file: str | Path,
    file_info: dict[str, Any],
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    skip_existing: bool = False,
) -> bool:
    """Process one CyclObs SAR overpass file and write a standard HDF5 snapshot."""
    atcf_id = str(file_info["sid"])
    acq_time = cast(pd.Timestamp, pd.Timestamp(file_info["acquisition_start_time"]))

    sid = atcf_to_sid.get(atcf_id)
    if sid is None:
        warnings.warn(
            f"No IBTrACS SID for ATCF {atcf_id!r} — discarding {file}",
            stacklevel=2,
        )
        return False

    time_utc = to_compact_time(acq_time)
    dest_path = Source.path(sources_root, SOURCE_NAME, sid, time_utc)
    # Skip snapshots already on disk when skip_existing is enabled.
    if skip_existing and dest_path.exists():
        return True

    with Dataset(str(file)) as raw:
        wind_speed = np.ma.filled(raw["wind_speed"][:].astype(float), np.nan)
        if wind_speed.ndim == 3:
            wind_speed = wind_speed[0]
        if "mask_flag" in raw.variables:
            mask_flag = np.array(raw["mask_flag"][:])
            if mask_flag.ndim == 3:
                mask_flag = mask_flag[0]
        else:
            # Some files omit mask_flag entirely; treat all pixels as valid.
            mask_flag = np.zeros(wind_speed.shape, dtype=np.int8)
        lat_1d = np.array(raw["lat"][:], dtype=np.float32)
        lon_1d = np.array(raw["lon"][:], dtype=np.float32)

    lon_1d = (lon_1d + 180) % 360 - 180
    wind_speed = np.asarray(wind_speed, dtype=np.float32)
    # mask_flag == 0 marks valid pixels; non-zero values are masked out as NaN.
    wind_speed[mask_flag != 0] = np.nan
    if np.all(np.isnan(wind_speed)):
        return False

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    side = 2 * SAR_CENTER_CROP_HALF_WIDTH_PX + 1
    wind_speed, lat_2d, lon_2d = center_crop_or_pad_2d(side, side, wind_speed, lat_2d, lon_2d)
    h, w = lat_2d.shape
    values_np = wind_speed[:, :, np.newaxis].astype(np.float32)
    mask_np = np.isfinite(values_np)
    # Spatial coords only: [lat, lon] per pixel — time goes to Source.time_utc.
    coords_np = np.stack([lat_2d, lon_2d], axis=-1)

    source = Source(
        kind=SourceKind.FIELD,
        values=values_np,
        coords=coords_np,
        source_name=SOURCE_NAME,
        channels=CHANNELS,
        mask=mask_np,
        time_utc=acq_time,
        meta={
            "storm_id": sid,
            "time_utc": acq_time.isoformat(),
        },
    )
    source.write(dest_path)
    return True


def _process_sar_item(
    item: tuple[Path, dict[str, Any]],
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    skip_existing: bool = False,
) -> bool:
    """Process one ``(file, metadata)`` pair (picklable entry point for ``map_files``)."""
    file, file_info = item
    return process_sar_file(file, file_info, sources_root, atcf_to_sid, skip_existing)


def _process_all_files(
    files: list[Path],
    file_infos: list[dict[str, Any]],
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    num_workers: int,
    skip_existing: bool,
) -> None:
    """Process all SAR overpass files."""
    items = list(zip(files, file_infos, strict=True))
    map_files(
        _process_sar_item,
        items,
        sources_root,
        atcf_to_sid,
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

    atcf_to_sid = load_translation(sources_root)

    acq_df = pd.read_csv(
        cyclobs_dir / "sar_acquisitions_metadata.csv",
        parse_dates=["acquisition_start_time"],
    )
    acq_df["season"] = acq_df["sid"].str[-4:]
    acq_df["basin"] = acq_df["sid"].str[:2].str.upper()
    acq_df["storm_number"] = acq_df["sid"].str[2:4].astype(int)
    # Rebuild canonical ATCF ids (uppercase basin, zero-padded number) for Stage 0 lookup.
    acq_df["sid"] = acq_df.apply(
        lambda row: f"{row['basin']}{row['storm_number']:02d}{row['season']}", axis=1
    )

    include_seasons = cfg.get("include_seasons")
    if include_seasons is not None:
        acq_df = acq_df[acq_df["season"].isin(include_seasons)].reset_index(drop=True)
        print(f"Filtered to seasons {include_seasons}: {len(acq_df)} acquisitions.")

    keep_cols = ["sid", "acquisition_start_time"]
    subset = cast(pd.DataFrame, acq_df[keep_cols])
    file_infos = cast(list[dict[str, Any]], subset.to_dict(orient="records"))
    files = [cyclobs_dir / url.split("/")[-1] for url in acq_df["data_url"]]
    print(f"Processing {len(files)} SAR acquisitions…")

    launch_local_or_slurm(
        cfg,
        "prep_sar",
        lambda: _process_all_files(
            files, file_infos, sources_root, atcf_to_sid, num_workers, skip_existing
        ),
    )

    sar_side = 2 * SAR_CENTER_CROP_HALF_WIDTH_PX + 1
    if (
        finalize_source(
            SOURCE_NAME,
            "sar",
            SourceKind.FIELD,
            CHANNELS,
            shape=(sar_side, sar_side),
            sources_root=sources_root,
            cfg=cfg,
        )
        == 0
    ):
        print("No valid SAR snapshots found.")


if __name__ == "__main__":
    main()
