#!/usr/bin/env python3
"""Preprocess passive microwave (PMW) data from TC-PRIMED into the standard HDF5 format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.tc_primed.regrid_utils import get_regridding_resolution
from scripts.preprocess.tc_primed.tc_primed_meta import read_tc_primed_overpass_meta
from scripts.preprocess.tc_primed.utils import (
    list_tc_primed_overpass_files_by_sensat,
    should_skip_existing,
)
from scripts.preprocess.utils.regridding import ResamplingError, regrid
from scripts.preprocess.utils.runner import (
    finalize_source,
    make_index_row,
    map_files,
    resolve_preproc_cfg,
    submit_slurm_jobs,
)
from tcfuse.data.sources import Source, SourceKind
from tcfuse.utils.time import to_compact_time

SENSOR_VARIABLES: dict[str, dict[str, tuple[str, list[str]]]] = {
    "AMSR2": {"37": ("S4", ["TB_36.5H", "TB_36.5V"]), "89": ("S5", ["TB_A89.0H", "TB_A89.0V"])},
    "AMSRE": {"37": ("S4", ["TB_36.5H", "TB_36.5V"]), "89": ("S5", ["TB_A89.0H", "TB_A89.0V"])},
    "GMI": {"37": ("S1", ["TB_36.64H", "TB_36.64V"]), "89": ("S1", ["TB_89.0H", "TB_89.0V"])},
    "SSMI": {"37": ("S1", ["TB_37.0H", "TB_37.0V"]), "89": ("S2", ["TB_85.5H", "TB_85.5V"])},
    "SSMIS": {"37": ("S2", ["TB_37.0H", "TB_37.0V"]), "89": ("S4", ["TB_91.665H", "TB_91.665V"])},
    "TMI": {"37": ("S2", ["TB_37.0H", "TB_37.0V"]), "89": ("S3", ["TB_85.5H", "TB_85.5V"])},
}


def _read_pmw_swath(
    grp: Any, variables: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read latitude, longitude, and brightness-temperature arrays from a swath group."""
    lat = np.ma.filled(grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180
    data = {v: np.ma.filled(grp[v][:].astype(float), np.nan) for v in variables}
    return lat, lon, data


def process_pmw_file(
    file: str | Path,
    sensat: str,
    ifovs: dict,
    sources_root: Path,
    skip_existing: bool = False,
) -> dict[str, Any] | None:
    """Process one TC-PRIMED overpass file and write a standard HDF5 snapshot."""
    sensor = sensat.split("_")[0]
    swath_37, vars_37 = SENSOR_VARIABLES[sensor]["37"]
    swath_89, vars_89 = SENSOR_VARIABLES[sensor]["89"]
    source_name = f"pmw_{sensat.lower()}"
    channels = [v.lower() for v in vars_37 + vars_89]

    with Dataset(str(file)) as raw:
        meta = read_tc_primed_overpass_meta(raw)
        storm_id = meta["storm_id"]
        basin = meta["basin"]
        time_unix_s = meta["time_unix_s"]
        storm_lat = meta["storm_lat"]
        storm_lon = meta["storm_lon"]

        overpass_time = pd.Timestamp(time_unix_s, unit="s")
        overpass_time_utc = to_compact_time(time_unix_s, unit="s")
        dest_path = Source.path(sources_root, source_name, storm_id, overpass_time_utc)
        if should_skip_existing(dest_path, skip_existing):
            return None

        lat89, lon89, data89 = _read_pmw_swath(raw["passive_microwave"][swath_89], vars_89)
        if any(np.all(np.isnan(arr)) for arr in data89.values()):
            return None

        regridding_res = get_regridding_resolution(sensat, swath_89, ifovs)
        try:
            (resampled89, out_lats, out_lons), target_area = regrid(
                lat89, lon89, data89, regridding_res
            )
        except ResamplingError as exc:
            raise RuntimeError(f"89 GHz regrid failed for {file}") from exc

        lat37, lon37, data37 = _read_pmw_swath(raw["passive_microwave"][swath_37], vars_37)
        if any(np.all(np.isnan(arr)) for arr in data37.values()):
            return None

        try:
            (resampled37, _, _), _ = regrid(
                lat37, lon37, data37, regridding_res, target_area=target_area
            )
        except ResamplingError as exc:
            raise RuntimeError(f"37 GHz regrid failed for {file}") from exc

        all_vars = vars_37 + vars_89
        merged = {**resampled37, **resampled89}
        values_np = np.stack([merged[v] for v in all_vars], axis=-1).astype(np.float32)
        lats = out_lats.astype(np.float32)
        lons = out_lons.astype(np.float32)

    src_h, src_w = lats.shape
    time_broadcast = np.full((src_h, src_w), time_unix_s, dtype=np.float32)
    coords_np = np.stack([time_broadcast, lats, lons], axis=-1)
    mask_np = np.isfinite(values_np)

    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=source_name,
        channels=channels,
        mask=torch.from_numpy(mask_np),
        meta={
            "storm_id": storm_id,
            "basin": basin,
            "snapshot_time_utc": overpass_time.isoformat(),
            "lat": storm_lat,
            "lon": storm_lon,
        },
    )
    source.write(dest_path)

    return make_index_row(
        storm_id,
        overpass_time.isoformat(),
        storm_lat,
        storm_lon,
        source_name,
        dest_path,
    )


def _process_sensat_files(
    files: list[Path],
    sensat: str,
    ifovs: dict,
    sources_root: Path,
    num_workers: int,
    skip_existing: bool,
) -> list[dict[str, Any] | None]:
    """Process all PMW files for one sensat."""
    return map_files(
        process_pmw_file,
        files,
        sensat,
        ifovs,
        sources_root,
        skip_existing,
        num_workers=num_workers,
        desc=sensat,
    )


def _pmw_source_config(sensat: str, ifovs: dict) -> tuple[str, list[str], dict[str, Any]]:
    """Return source name, channels, and metadata char_vars for one sensat."""
    sensor = sensat.split("_")[0]
    swath_37, vars_37 = SENSOR_VARIABLES[sensor]["37"]
    swath_89, vars_89 = SENSOR_VARIABLES[sensor]["89"]
    source_name = f"pmw_{sensat.lower()}"
    channels = [v.lower() for v in vars_37 + vars_89]
    char_vars = {
        "target_resolution_km": get_regridding_resolution(sensat, swath_89, ifovs),
        "swaths": {"37": swath_37, "89": swath_89},
    }
    return source_name, channels, char_vars


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED PMW snapshots to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    with open(tc_primed_path / "tc_primed_ifovs.yaml") as f:
        ifovs: dict = yaml.safe_load(f)

    pmw_files = list_tc_primed_overpass_files_by_sensat(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )

    supported = {
        sensat: files
        for sensat, files in pmw_files.items()
        if sensat.split("_")[0] in SENSOR_VARIABLES
    }
    for sensat in pmw_files:
        if sensat not in supported:
            print(f"Skipping unsupported sensor: {sensat}")

    def run_all() -> None:
        for sensat, files in supported.items():
            print(f"Processing {sensat} ({len(files)} files)…")
            _process_sensat_files(files, sensat, ifovs, sources_root, num_workers, skip_existing)

    def run_all_slurm() -> None:
        submit_slurm_jobs(
            cfg,
            "prepare_pmw",
            [
                (
                    sensat,
                    _process_sensat_files,
                    (files, sensat, ifovs, sources_root, num_workers, skip_existing),
                )
                for sensat, files in supported.items()
            ],
        )

    if cfg.get("submitit", False):
        run_all_slurm()
    else:
        run_all()

    written = 0
    for sensat in supported:
        source_name, channels, char_vars = _pmw_source_config(sensat, ifovs)
        written += finalize_source(
            source_name, "pmw", SourceKind.FIELD, channels, sources_root, cfg, char_vars
        )

    if written == 0:
        print("No valid snapshots found.")


if __name__ == "__main__":
    main()
