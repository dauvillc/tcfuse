#!/usr/bin/env python3
"""Stage 1 — preprocess geostationary infrared snapshots from TC-PRIMED."""

from __future__ import annotations

import warnings
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from omegaconf import DictConfig

from scripts.preprocess.tc_primed.utils import (
    list_tc_primed_storm_files,
    read_tc_primed_overpass_meta,
)
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

# infrared_availability_flag → source_name (0 = none, 1 = TCIRAR, 2 = HURSAT).
IR_FLAG_TO_SOURCE: list[str | None] = [None, "ir_tcirar", "ir_hursat"]
IR_SOURCE_IFOVS: dict[str, float] = {"ir_tcirar": 4.0, "ir_hursat": 8.0}
# Native grid spacing differs by product; crop half-width is in pixels, not km.
IR_CENTER_CROP_HALF_WIDTH_PX: dict[str, int] = {"ir_tcirar": 200, "ir_hursat": 100}


def _read_ir_data(ir_grp: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read IRWIN brightness temperature and spatial coordinates from an IR group."""
    irwin = np.ma.filled(ir_grp["IRWIN"][:].astype(float), np.nan)
    lat = np.ma.filled(ir_grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(ir_grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180

    # TC-PRIMED IR arrays vary: leading singleton dims or 1-D lat/lon vectors.
    while irwin.ndim > 2 and irwin.shape[0] == 1:
        irwin = irwin[0]

    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    return irwin, lat, lon


def process_ir_file(
    file: str | Path,
    sources_root: Path,
    atcf_to_sid: dict[str, str],
    skip_existing: bool = False,
) -> bool:
    """Process one TC-PRIMED overpass file and write a standard HDF5 IR snapshot."""
    with Dataset(str(file)) as raw:
        meta = read_tc_primed_overpass_meta(raw)
        atcf_id = meta["storm_id"]
        time_unix_s = meta["time_unix_s"]

        if "infrared" not in raw.groups:
            return False
        ir_grp = raw["infrared"]
        flag = int(ir_grp["infrared_availability_flag"][0])
        source_name = IR_FLAG_TO_SOURCE[flag] if flag < len(IR_FLAG_TO_SOURCE) else None
        if source_name is None:
            return False

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
        if skip_existing and dest_path.exists():
            return True

        irwin, lat2d, lon2d = _read_ir_data(ir_grp)

    if np.all(np.isnan(irwin)):
        return False

    # Center-crop or pad to fixed storm-centered square on the native IR grid.
    half_width_px = IR_CENTER_CROP_HALF_WIDTH_PX[source_name]
    side = 2 * half_width_px + 1
    irwin, lat2d, lon2d = center_crop_or_pad_2d(side, side, irwin, lat2d, lon2d)

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
            "storm_id": sid,
            "snapshot_time_utc": overpass_time.isoformat(),
        },
    )
    source.write(dest_path)
    return True


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED IR snapshots to the standard HDF5 format."""
    cfg = resolve_preproc_cfg(raw_cfg)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    atcf_to_sid = load_translation(sources_root)

    overpass_files_by_storm, _ = list_tc_primed_storm_files(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )
    all_files = sorted({f for f in chain.from_iterable(overpass_files_by_storm.values())})
    print(f"Found {len(all_files)} overpass files.")

    launch_local_or_slurm(
        cfg,
        "prepare_infrared",
        lambda: map_files(
            process_ir_file,
            all_files,
            sources_root,
            atcf_to_sid,
            skip_existing,
            num_workers=num_workers,
            desc="ir",
        ),
    )

    written = 0
    for source_name in ("ir_tcirar", "ir_hursat"):
        half = IR_CENTER_CROP_HALF_WIDTH_PX[source_name]
        side = 2 * half + 1
        written += finalize_source(
            source_name,
            "infrared",
            SourceKind.FIELD,
            ["irwin"],
            shape=(side, side),
            sources_root=sources_root,
            cfg=cfg,
            char_vars={"ifov_km": IR_SOURCE_IFOVS[source_name]},
        )

    if written == 0:
        print("No valid IR snapshots found.")


if __name__ == "__main__":
    main()
