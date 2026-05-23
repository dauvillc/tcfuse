#!/usr/bin/env python3
"""Preprocess geostationary infrared data from TC-PRIMED into the standard HDF5 format."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.tc_primed.tc_primed_meta import read_tc_primed_overpass_meta
from scripts.preprocess.tc_primed.utils import list_tc_primed_storm_files, should_skip_existing
from scripts.preprocess.utils.runner import (
    finalize_source,
    launch_local_or_slurm,
    make_index_row,
    map_files,
    resolve_preproc_cfg,
)
from tcfuse.data.sources import Source, SourceKind
from tcfuse.utils.time import to_compact_time

IR_FLAG_TO_SOURCE: list[str | None] = [None, "ir_tcirar", "ir_hursat"]
IR_SOURCE_IFOVS: dict[str, float] = {"ir_tcirar": 4.0, "ir_hursat": 8.0}


def _read_ir_data(ir_grp: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read IRWIN brightness temperature and spatial coordinates from an IR group."""
    irwin = np.ma.filled(ir_grp["IRWIN"][:].astype(float), np.nan)
    lat = np.ma.filled(ir_grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(ir_grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180

    while irwin.ndim > 2 and irwin.shape[0] == 1:
        irwin = irwin[0]

    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    return irwin, lat, lon


def process_ir_file(
    file: str | Path,
    sources_root: Path,
    skip_existing: bool = False,
) -> dict[str, Any] | None:
    """Process one TC-PRIMED overpass file and write a standard HDF5 IR snapshot."""
    with Dataset(str(file)) as raw:
        meta = read_tc_primed_overpass_meta(raw)
        storm_id = meta["storm_id"]
        basin = meta["basin"]
        time_unix_s = meta["time_unix_s"]
        storm_lat = meta["storm_lat"]
        storm_lon = meta["storm_lon"]

        if "infrared" not in raw.groups:
            return None
        ir_grp = raw["infrared"]
        flag = int(ir_grp["infrared_availability_flag"][0])
        source_name = IR_FLAG_TO_SOURCE[flag] if flag < len(IR_FLAG_TO_SOURCE) else None
        if source_name is None:
            return None

        overpass_time = pd.Timestamp(time_unix_s, unit="s")
        overpass_time_utc = to_compact_time(time_unix_s, unit="s")
        dest_path = Source.path(sources_root, source_name, storm_id, overpass_time_utc)
        if should_skip_existing(dest_path, skip_existing):
            return None

        irwin, lat2d, lon2d = _read_ir_data(ir_grp)

    if np.all(np.isnan(irwin)):
        return None

    h, w = irwin.shape
    values_np = irwin[..., np.newaxis].astype(np.float32)
    time_broadcast = np.full((h, w), time_unix_s, dtype=np.float32)
    coords_np = np.stack(
        [time_broadcast, lat2d.astype(np.float32), lon2d.astype(np.float32)], axis=-1
    )
    mask_np = np.isfinite(values_np)

    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=source_name,
        channels=["irwin"],
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


def _process_all_files(
    files: list[Path],
    sources_root: Path,
    num_workers: int,
    skip_existing: bool,
) -> list[dict[str, Any] | None]:
    """Process all overpass files for IR extraction."""
    return map_files(
        process_ir_file,
        files,
        sources_root,
        skip_existing,
        num_workers=num_workers,
        desc="ir",
    )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED IR snapshots to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    overpass_files_by_storm, _ = list_tc_primed_storm_files(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )
    all_files = sorted({f for f in chain.from_iterable(overpass_files_by_storm.values())})
    print(f"Found {len(all_files)} overpass files.")

    launch_local_or_slurm(
        cfg,
        "prepare_infrared",
        lambda: _process_all_files(all_files, sources_root, num_workers, skip_existing),
        lambda: _process_all_files(all_files, sources_root, num_workers, skip_existing),
    )

    written = 0
    for source_name in ("ir_tcirar", "ir_hursat"):
        written += finalize_source(
            source_name,
            "infrared",
            SourceKind.FIELD,
            ["irwin"],
            sources_root,
            cfg,
            {"ifov_km": IR_SOURCE_IFOVS[source_name]},
        )

    if written == 0:
        print("No valid IR snapshots found.")


if __name__ == "__main__":
    main()
