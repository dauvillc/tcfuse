#!/usr/bin/env python3
"""Preprocess passive microwave (PMW) data from TC-PRIMED into the standard HDF5 format.

Reads raw TC-PRIMED NetCDF4 overpass files, extracts 37 GHz and 89 GHz brightness
temperatures, regrids both to a common equiangular grid at the 89 GHz native resolution
(determined from tc_primed_ifovs.yaml), and writes one HDF5 snapshot file per observation
in the project standard format.  A consolidated index.parquet is written to the dataset
root at the end.

Run from the project root:
    python scripts/preprocess/tc_primed/prepare_pmw.py [paths=jz] [submitit=false] [num_workers=4]
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
import yaml
from netCDF4 import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.preprocess.tc_primed.utils import (
    list_tc_primed_overpass_files_by_sensat,
    should_skip_existing,
)
from scripts.preprocess.utils.regridding import ResamplingError, regrid
from tcfuse.data.sources import Source, SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

# Sensor → {"37": (swath_name, [variable_names]), "89": (swath_name, [variable_names])}
# Only 37 GHz and 89 GHz (or the closest equivalents for each sensor) are extracted.
SENSOR_VARIABLES: dict[str, dict[str, tuple[str, list[str]]]] = {
    "AMSR2": {"37": ("S4", ["TB_36.5H", "TB_36.5V"]), "89": ("S5", ["TB_A89.0H", "TB_A89.0V"])},
    "AMSRE": {"37": ("S4", ["TB_36.5H", "TB_36.5V"]), "89": ("S5", ["TB_A89.0H", "TB_A89.0V"])},
    "GMI": {"37": ("S1", ["TB_36.64H", "TB_36.64V"]), "89": ("S1", ["TB_89.0H", "TB_89.0V"])},
    "SSMI": {"37": ("S1", ["TB_37.0H", "TB_37.0V"]), "89": ("S2", ["TB_85.5H", "TB_85.5V"])},
    "SSMIS": {"37": ("S2", ["TB_37.0H", "TB_37.0V"]), "89": ("S4", ["TB_91.665H", "TB_91.665V"])},
    "TMI": {"37": ("S2", ["TB_37.0H", "TB_37.0V"]), "89": ("S3", ["TB_85.5H", "TB_85.5V"])},
}


def _get_regridding_resolution(sensat: str, swath_89: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a given sensor/swath.

    Uses the minimum IFOV value of the 89 GHz swath as the target resolution,
    so the regridded grid preserves the finest detail available.

    Args:
        sensat: Sensor/satellite string (e.g. "AMSR2_GCOMW1").
        swath_89: Swath identifier for the 89 GHz channel (e.g. "S5").
        ifovs: IFOV lookup dict loaded from tc_primed_ifovs.yaml.

    Returns:
        Minimum IFOV value in km.
    """
    ifov_entry = ifovs[sensat][swath_89]
    # There are two possibilities:
    # (1) ifov_entry is a list of values, meaning the same IFOV for all channels in the swath
    # (2) a dict {channel: ifov values}, meaning different IFOVs per channel
    # We'll always take the minimum value across all channels, to get the finest resolution.
    if isinstance(ifov_entry, dict):
        return min(min(vals) for vals in ifov_entry.values())
    return min(ifov_entry)


def _get_channel_ifovs(
    sensat: str, swath: str, variables: list[str], ifovs: dict
) -> dict[str, list[float]]:
    """Return per-channel IFOV values (km) for a given PMW swath.

    Each entry in the returned dict is a 4-element list
    ``[along_track_1, across_track_1, along_track_2, across_track_2]`` in km.

    Args:
        sensat: Sensor/satellite string (e.g. "AMSR2_GCOMW1").
        swath: Swath identifier (e.g. "S4").
        variables: Channel variable names in the swath (e.g. ["TB_36.5H", "TB_36.5V"]).
        ifovs: Full IFOV dict loaded from tc_primed_ifovs.yaml.

    Returns:
        Dict mapping lower-cased channel name to its IFOV list.
    """
    entry = ifovs[sensat][swath]
    result = {}
    for v in variables:
        # Entry is either a dict keyed by channel name (e.g. GMI S1) or a shared list.
        ifov = entry[v] if isinstance(entry, dict) else entry
        result[v.lower()] = list(ifov)
    return result


def _read_pmw_swath(
    grp: Any, variables: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read latitude, longitude, and brightness-temperature arrays from a netCDF4 swath group.

    Args:
        grp: netCDF4 group for the swath (e.g. raw["passive_microwave"]["S5"]).
        variables: List of variable names to read (e.g. ["TB_A89.0H", "TB_A89.0V"]).

    Returns:
        (lat, lon, data) where lat and lon are float64 (H, W) arrays and data maps
        each variable name to a float64 (H, W) array. Masked values are filled with NaN.
        Longitudes are normalised to [-180, 180].
    """
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
    max_age_hours: float | None = None,
) -> dict[str, Any] | None:
    """Process one TC-PRIMED overpass file and write a standard HDF5 snapshot.

    Extracts 37 GHz and 89 GHz brightness temperatures, regrids both to a common
    equiangular grid at the 89 GHz native resolution, and writes a single HDF5 file
    containing one FIELD source named ``pmw_{sensat}``.  The 37 GHz swath is regridded
    to the same area definition as the 89 GHz swath so the two align exactly.

    For sensors where 37 GHz and 89 GHz share the same swath (e.g. GMI on S1), the
    swath is opened twice — once per frequency band — so the code path is uniform.

    The snapshot is written to
    ``{sources_root}/{source_name}/snapshots/{storm_id}_{time}.h5``.

    When ``skip_existing`` is True, the output HDF5 is checked before reading swath
    data: if it already exists (and is not older than ``max_age_hours`` when set),
    the index row is returned immediately without re-processing.

    Args:
        file: Path to the raw TC-PRIMED NetCDF4 overpass file.
        sensat: Sensor/satellite string, e.g. "AMSR2_GCOMW1".
        ifovs: IFOV lookup dict loaded from tc_primed_ifovs.yaml.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        skip_existing: If True, skip files whose output snapshot already exists on disk
            and satisfies the age constraint. Default False.
        max_age_hours: Maximum age (hours) of an existing snapshot for it to be skipped.
            None means skip unconditionally when the file exists.

    Returns:
        Index row dict for inclusion in the per-source index.parquet, or None if discarded.
    """
    sensor = sensat.split("_")[0]
    swath_37, vars_37 = SENSOR_VARIABLES[sensor]["37"]
    swath_89, vars_89 = SENSOR_VARIABLES[sensor]["89"]
    source_name = f"pmw_{sensat.lower()}"
    # Channels ordered 37 GHz first, then 89 GHz
    channels = [v.lower() for v in vars_37 + vars_89]

    with Dataset(str(file)) as raw:
        # --- Overpass and storm metadata ---
        meta_grp = raw["overpass_metadata"]
        season = int(meta_grp["season"][0])
        basin = str(meta_grp["basin"][0])
        storm_number = int(meta_grp["cyclone_number"][-1])
        storm_id = f"{basin}{storm_number:02d}{season}"
        time_unix_s = float(meta_grp["time"][0])

        # Retrieve some info about the storm that were interpolated from best track.
        storm_grp = raw["overpass_storm_metadata"]
        storm_lat = float(storm_grp["storm_latitude"][0])
        storm_lon = (float(storm_grp["storm_longitude"][0]) + 180) % 360 - 180
        vmax_kt = float(storm_grp["intensity"][0])
        min_pressure_hpa = float(storm_grp["central_min_pressure"][0])
        development_lvl = str(storm_grp["development_level"][0])  # e.g. "TD", "TS", "TY"
        storm_speed_ms = float(storm_grp["storm_speed"][0])
        storm_heading_deg = float(storm_grp["storm_heading"][0])

        # --- Early skip check (before reading swath data) ---
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
                "vmax_kt": vmax_kt,
                "mslp_hpa": min_pressure_hpa,
                "development_level": development_lvl,
                "storm_speed_ms": storm_speed_ms,
                "storm_heading_deg": storm_heading_deg,
                "source_name": source_name,
                "file_path": str(dest_path),
            }

        # --- 89 GHz swath: regrid to equiangular grid, capture target area ---
        lat89, lon89, data89 = _read_pmw_swath(raw["passive_microwave"][swath_89], vars_89)
        if any(np.all(np.isnan(arr)) for arr in data89.values()):
            return None

        regridding_res = _get_regridding_resolution(sensat, swath_89, ifovs)
        try:
            (resampled89, out_lats, out_lons), target_area = regrid(
                lat89, lon89, data89, regridding_res
            )
        except ResamplingError as exc:
            raise RuntimeError(f"89 GHz regrid failed for {file}") from exc

        # --- 37 GHz swath: regrid to the same equiangular area ---
        lat37, lon37, data37 = _read_pmw_swath(raw["passive_microwave"][swath_37], vars_37)
        if any(np.all(np.isnan(arr)) for arr in data37.values()):
            return None

        try:
            (resampled37, _, _), _ = regrid(
                lat37, lon37, data37, regridding_res, target_area=target_area
            )
        except ResamplingError as exc:
            raise RuntimeError(f"37 GHz regrid failed for {file}") from exc

        # Stack channels in order: 37 GHz vars first, then 89 GHz vars
        all_vars = vars_37 + vars_89
        merged = {**resampled37, **resampled89}
        # (H, W, C)
        values_np = np.stack([merged[v] for v in all_vars], axis=-1).astype(np.float32)
        lats = out_lats.astype(np.float32)  # (H, W)
        lons = out_lons.astype(np.float32)  # (H, W)

    # --- Build Source (file closed; all data is in numpy arrays) ---
    src_h, src_w = lats.shape
    time_broadcast = np.full((src_h, src_w), time_unix_s, dtype=np.float32)
    coords_np = np.stack([time_broadcast, lats, lons], axis=-1)  # (H, W, 3)
    # Availability is tracked per channel: True where the value is finite.
    mask_np = np.isfinite(values_np)  # (H, W, C)

    # Per-channel IFOVs (km) for both frequency bands.
    char_vars: dict[str, Any] = {
        "ifov": {
            **_get_channel_ifovs(sensat, swath_37, vars_37, ifovs),
            **_get_channel_ifovs(sensat, swath_89, vars_89, ifovs),
        }
    }

    overpass_time = pd.Timestamp(time_unix_s, unit="s")
    overpass_time_utc = overpass_time.strftime("%Y%m%dT%H%M%SZ")
    dest_path = Source.path(sources_root, source_name, storm_id, overpass_time_utc)
    meta: dict[str, Any] = {
        "storm_id": storm_id,
        "basin": basin,
        "snapshot_time_utc": overpass_time.isoformat(),
        "lat": storm_lat,
        "lon": storm_lon,
        "vmax_kt": vmax_kt,
        "mslp_hpa": min_pressure_hpa,
        "development_level": development_lvl,
        "storm_speed_ms": storm_speed_ms,
        "storm_heading_deg": storm_heading_deg,
    }
    source = Source(
        kind=SourceKind.FIELD,
        values=torch.from_numpy(values_np),
        coords=torch.from_numpy(coords_np),
        source_name=source_name,
        channels=channels,
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
        "vmax_kt": vmax_kt,
        "mslp_hpa": min_pressure_hpa,
        "development_level": development_lvl,
        "storm_speed_ms": storm_speed_ms,
        "storm_heading_deg": storm_heading_deg,
        "source_name": source_name,
        "file_path": str(dest_path),
    }


def _process_sensat_files(
    files: list[Path],
    sensat: str,
    ifovs: dict,
    sources_root: Path,
    num_workers: int,
    skip_existing: bool = False,
    max_age_hours: float | None = None,
) -> list[dict[str, Any] | None]:
    """Process all PMW files for one sensat using ProcessPoolExecutor.

    Designed to be submitted as a single submitit job or called directly
    for local execution.

    Args:
        files: Raw TC-PRIMED overpass file paths to process.
        sensat: Sensor/satellite string.
        ifovs: IFOV lookup dict.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        num_workers: Number of parallel worker processes.
        skip_existing: Forwarded to ``process_pmw_file``.
        max_age_hours: Forwarded to ``process_pmw_file``.

    Returns:
        List of index row dicts (None entries for discarded snapshots).
    """
    if num_workers <= 1:
        return [
            process_pmw_file(f, sensat, ifovs, sources_root, skip_existing, max_age_hours)
            for f in tqdm(files, desc=sensat)
        ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    process_pmw_file,
                    files,
                    repeat(sensat),
                    repeat(ifovs),
                    repeat(sources_root),
                    repeat(skip_existing),
                    repeat(max_age_hours),
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc=sensat,
            )
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED PMW snapshots to the standard HDF5 format."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    ifovs_path = tc_primed_path / "tc_primed_ifovs.yaml"
    sources_root = Path(cfg["paths"]["preprocessed_sources"])

    with open(ifovs_path) as f:
        ifovs: dict = yaml.safe_load(f)

    pmw_files = list_tc_primed_overpass_files_by_sensat(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )

    launch_local = not bool(cfg.get("submitit", False))
    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))
    max_age_hours: float | None = cfg.get("max_age_hours", None)
    if max_age_hours is not None:
        max_age_hours = float(max_age_hours)

    # Collect rows per source name so each source gets its own index.parquet.
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    channels_by_source: dict[str, list[str]] = {}
    char_vars_by_source: dict[str, dict[str, Any]] = {}

    if launch_local:
        for sensat, files in pmw_files.items():
            sensor = sensat.split("_")[0]
            if sensor not in SENSOR_VARIABLES:
                print(f"Skipping unsupported sensor: {sensat}")
                continue
            print(f"Processing {sensat} ({len(files)} files)…")
            results: list[dict[str, Any] | None] = _process_sensat_files(
                files, sensat, ifovs, sources_root, num_workers, skip_existing, max_age_hours
            )
            valid = [r for r in results if r is not None]
            discarded = len(results) - len(valid)
            if discarded:
                pct = 100.0 * discarded / max(len(results), 1)
                print(f"  Discarded {discarded}/{len(results)} ({pct:.1f}%)")
            if not valid:
                if results:
                    print(f"  All snapshots discarded for {sensat}; skipping source")
                continue
            source_name = f"pmw_{sensat.lower()}"
            swath_37, vars_37 = SENSOR_VARIABLES[sensor]["37"]
            swath_89, vars_89 = SENSOR_VARIABLES[sensor]["89"]
            channels_by_source[source_name] = [v.lower() for v in vars_37 + vars_89]
            char_vars_by_source[source_name] = {
                "ifov": {
                    **_get_channel_ifovs(sensat, swath_37, vars_37, ifovs),
                    **_get_channel_ifovs(sensat, swath_89, vars_89, ifovs),
                }
            }
            rows_by_source.setdefault(source_name, []).extend(valid)
    else:
        from tcfuse.utils.submitit_utils import make_executor as make_submitit_executor

        slurm_executor = make_submitit_executor(cfg, "prepare_pmw")
        jobs: dict[str, Any] = {}
        for sensat, files in pmw_files.items():
            sensor = sensat.split("_")[0]
            if sensor not in SENSOR_VARIABLES:
                print(f"Skipping unsupported sensor: {sensat}")
                continue
            print(f"Submitting {sensat} ({len(files)} files)…")
            jobs[sensat] = slurm_executor.submit(
                _process_sensat_files,
                files,
                sensat,
                ifovs,
                sources_root,
                num_workers,
                skip_existing,
                max_age_hours,
            )
        for sensat, job in tqdm(jobs.items(), desc="collecting results"):
            results = job.result()
            valid = [r for r in results if r is not None]
            discarded = len(results) - len(valid)
            if discarded:
                pct = 100.0 * discarded / max(len(results), 1)
                print(f"  {sensat}: discarded {discarded}/{len(results)} ({pct:.1f}%)")
            if not valid:
                if results:
                    print(f"  {sensat}: all snapshots discarded; skipping source")
                continue
            sensor = sensat.split("_")[0]
            source_name = f"pmw_{sensat.lower()}"
            swath_37, vars_37 = SENSOR_VARIABLES[sensor]["37"]
            swath_89, vars_89 = SENSOR_VARIABLES[sensor]["89"]
            channels_by_source[source_name] = [v.lower() for v in vars_37 + vars_89]
            char_vars_by_source[source_name] = {
                "ifov": {
                    **_get_channel_ifovs(sensat, swath_37, vars_37, ifovs),
                    **_get_channel_ifovs(sensat, swath_89, vars_89, ifovs),
                }
            }
            rows_by_source.setdefault(source_name, []).extend(valid)

    if rows_by_source:
        for source_name, rows in rows_by_source.items():
            if not rows:
                continue
            index_df = pd.DataFrame(rows)
            index_path = sources_root / source_name / "index.parquet"
            index_df.to_parquet(index_path, index=False)
            print(f"Wrote index ({source_name}): {len(index_df)} rows → {index_path}")
            source_meta = SourceMetadata(
                source_name,
                "pmw",
                SourceKind.FIELD,
                channels_by_source[source_name],
                index_df,
                char_vars=char_vars_by_source[source_name],
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
        print("No valid snapshots found.")


if __name__ == "__main__":
    main()
