#!/usr/bin/env python3
"""Stage 1 — preprocess Ku/Ka-band radar snapshots from TC-PRIMED."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.tc_primed.regrid_utils import get_regridding_resolution
from scripts.preprocess.tc_primed.tc_primed_meta import read_tc_primed_overpass_meta
from scripts.preprocess.tc_primed.utils import (
    list_tc_primed_overpass_files_by_sensat,
    load_tc_primed_ifovs,
    should_skip_existing,
)
from scripts.preprocess.utils.regridding import ResamplingError, regrid
from scripts.preprocess.utils.runner import (
    finalize_source,
    load_translation,
    map_files,
    resolve_preproc_cfg,
    submit_slurm_jobs,
)
from tcfuse.data.sources import Source, SourceKind
from tcfuse.utils.time import to_compact_time

SENSAT_VARIABLES: dict[str, tuple[str, list[str]]] = {
    "GMI_GPM": (
        "KuGMI",
        ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"],
    ),
    "TMI_TRMM": (
        "KuTMI",
        ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"],
    ),
}


def _read_radar_swath(
    grp: Any, variables: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read latitude, longitude, and radar arrays from a swath group."""
    lat = np.ma.filled(grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180
    data = {v: np.ma.filled(grp[v][:].astype(float), np.nan) for v in variables}
    return lat, lon, data


def process_radar_file(
    file: str | Path,
    sensat: str,
    ifovs: dict,
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    skip_existing: bool = False,
) -> bool:
    """Process one TC-PRIMED overpass file and write a standard HDF5 snapshot.

    Returns ``True`` when a snapshot was written or kept, ``False`` otherwise.
    """
    swath, variables = SENSAT_VARIABLES[sensat]
    sensor_abbrev = sensat.split("_")[0].lower()
    source_name = f"radar_{sensor_abbrev}"
    channels = [v.lower() for v in variables]

    with Dataset(str(file)) as raw:
        meta = read_tc_primed_overpass_meta(raw)
        atcf_id = meta["storm_id"]
        time_unix_s = meta["time_unix_s"]

        sid = atcf_to_sid.get(atcf_id)
        if sid is None:
            warnings.warn(
                f"No IBTrACS SID for ATCF {atcf_id!r} — discarding {file}",
                stacklevel=2,
            )
            return False

        overpass_time = pd.Timestamp(time_unix_s, unit="s")
        overpass_time_utc = to_compact_time(time_unix_s, unit="s")
        dest_path = Source.path(sources_root, source_name, sid, overpass_time_utc)
        if should_skip_existing(dest_path, skip_existing):
            return True

        if "radar_radiometer" not in raw.groups:
            return False
        radar_grp = raw["radar_radiometer"]
        if int(radar_grp["availability_flag"][0]) == 0 or swath not in radar_grp.groups:
            return False

        lat, lon, data = _read_radar_swath(radar_grp[swath], variables)
        if any(np.all(np.isnan(arr)) for arr in data.values()):
            return False

        regridding_res = get_regridding_resolution(sensat, swath, ifovs)
        try:
            (resampled, out_lats, out_lons), _ = regrid(lat, lon, data, regridding_res)
        except ResamplingError as exc:
            raise RuntimeError(f"Radar regrid failed for {file}") from exc

        values_np = np.stack([resampled[v] for v in variables], axis=-1).astype(np.float32)
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
            "storm_id": sid,
            "snapshot_time_utc": overpass_time.isoformat(),
        },
    )
    source.write(dest_path)
    return True


def _process_sensat_files(
    files: list[Path],
    sensat: str,
    ifovs: dict,
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    num_workers: int,
    skip_existing: bool,
) -> list[bool | None]:
    """Process all radar files for one sensat."""
    return map_files(
        process_radar_file,
        files,
        sensat,
        ifovs,
        sources_root,
        atcf_to_sid,
        skip_existing,
        num_workers=num_workers,
        desc=sensat,
    )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED radar snapshots to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    atcf_to_sid = load_translation(sources_root)
    ifovs = load_tc_primed_ifovs()

    radar_files = list_tc_primed_overpass_files_by_sensat(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )
    supported = {
        sensat: files for sensat, files in radar_files.items() if sensat in SENSAT_VARIABLES
    }

    def run_all() -> None:
        for sensat, files in supported.items():
            print(f"Processing {sensat} ({len(files)} files)…")
            _process_sensat_files(
                files, sensat, ifovs, sources_root, atcf_to_sid, num_workers, skip_existing
            )

    def run_all_slurm() -> None:
        submit_slurm_jobs(
            cfg,
            "prepare_radar",
            [
                (
                    sensat,
                    _process_sensat_files,
                    (
                        files,
                        sensat,
                        ifovs,
                        sources_root,
                        atcf_to_sid,
                        num_workers,
                        skip_existing,
                    ),
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
        swath, variables = SENSAT_VARIABLES[sensat]
        source_name = f"radar_{sensat.split('_')[0].lower()}"
        channels = [v.lower() for v in variables]
        char_vars = {"target_resolution_km": get_regridding_resolution(sensat, swath, ifovs)}
        written += finalize_source(
            source_name, "radar", SourceKind.FIELD, channels, sources_root, cfg, char_vars
        )

    if written == 0:
        print("No valid snapshots found.")


if __name__ == "__main__":
    main()
