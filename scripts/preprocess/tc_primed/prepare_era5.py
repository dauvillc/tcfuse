#!/usr/bin/env python3
"""Stage 1 — preprocess TC-PRIMED ERA5 surface patches to standard HDF5 sources."""

from __future__ import annotations

import warnings
from itertools import chain
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.tc_primed.utils import list_tc_primed_storm_files
from scripts.preprocess.utils.runner import (
    finalize_source,
    launch_local_or_slurm,
    load_translation,
    map_files,
    resolve_preproc_cfg,
)
from tcfuse.data.sources import Source, SourceKind
from tcfuse.utils.time import to_compact_time

ERA5_SOURCE_NAME = "era5_surface"
# 2D surface variables in the rectilinear group (no level dimension).
ERA5_2D_CHANNELS: list[str] = [
    "precipitable_water",
    "rain_large_scale",
    "rain_convective",
    "sst",
    "pressure_msl",
    "temperature_2m",
    "dewpoint_2m",
    "u_wind_10m",
    "v_wind_10m",
]
# 30° storm-centered patch at 0.25°/px: 121 grid points per side.
ERA5_GRID_SHAPE: tuple[int, int] = (121, 121)


def process_env_file(
    file: str | Path,
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    skip_existing: bool = False,
) -> bool:
    """Process one TC-PRIMED env file, writing one HDF5 snapshot per synoptic time."""
    with Dataset(str(file)) as raw:
        storm_meta = raw["storm_metadata"]
        basin = str(storm_meta["basin"][:])
        cyclone_number = int(storm_meta["cyclone_number"][:])
        season = int(storm_meta["season"][:])
        atcf_id = f"{basin}{cyclone_number:02d}{season}"

        sid = atcf_to_sid.get(atcf_id)
        if sid is None:
            warnings.warn(
                f"No IBTrACS SID for ATCF {atcf_id!r} — discarding {file}",
                stacklevel=2,
            )
            return False

        rect = raw["rectilinear"]
        times_unix = raw["storm_metadata"]["time"][:]
        n_times = len(times_unix)

        written = False
        for t in range(n_times):
            time_unix_s = int(times_unix[t])
            overpass_time = cast(pd.Timestamp, pd.Timestamp(time_unix_s, unit="s"))
            overpass_time_utc = to_compact_time(time_unix_s, unit="s")
            dest_path = Source.path(sources_root, ERA5_SOURCE_NAME, sid, overpass_time_utc)
            if skip_existing and dest_path.exists():
                written = True
                continue

            # lat/lon are 1D per time step on this rectilinear storm-centered grid.
            lat_1d = np.ma.filled(rect["latitude"][t].astype(float), np.nan)
            lon_1d = (
                np.ma.filled(rect["longitude"][t].astype(float), np.nan) + 180
            ) % 360 - 180
            lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

            values_np = np.stack(
                [
                    np.ma.filled(rect[ch][t].astype(float), np.nan)
                    for ch in ERA5_2D_CHANNELS
                ],
                axis=-1,
            ).astype(np.float32)
            coords_np = np.stack(
                [lat2d.astype(np.float32), lon2d.astype(np.float32)], axis=-1
            )
            mask_np = np.isfinite(values_np)

            source = Source(
                kind=SourceKind.FIELD,
                values=values_np,
                coords=coords_np,
                source_name=ERA5_SOURCE_NAME,
                channels=ERA5_2D_CHANNELS,
                mask=mask_np,
                time_utc=overpass_time,
                meta={
                    "storm_id": sid,
                    "time_utc": overpass_time.isoformat(),
                },
            )
            source.write(dest_path)
            written = True

    return written


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED ERA5 surface patches to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    atcf_to_sid = load_translation(sources_root)

    _, env_files_by_storm = list_tc_primed_storm_files(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )
    all_files = sorted({f for f in chain.from_iterable(env_files_by_storm.values())})
    print(f"Found {len(all_files)} ERA5 env files.")

    launch_local_or_slurm(
        cfg,
        "prep_era5",
        lambda: map_files(
            process_env_file,
            all_files,
            sources_root,
            atcf_to_sid,
            skip_existing,
            num_workers=num_workers,
            desc="era5",
        ),
    )

    written = finalize_source(
        ERA5_SOURCE_NAME,
        "environmental",
        SourceKind.FIELD,
        ERA5_2D_CHANNELS,
        shape=ERA5_GRID_SHAPE,
        sources_root=sources_root,
        cfg=cfg,
    )

    if written == 0:
        print("No valid ERA5 snapshots found.")


if __name__ == "__main__":
    main()
