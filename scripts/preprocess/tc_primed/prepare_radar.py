#!/usr/bin/env python3
"""Preprocess Ku/Ka-band radar data from TC-PRIMED into the standard HDF5 format.

Reads raw TC-PRIMED NetCDF4 overpass files, extracts near-surface precipitation
rate, rate uncertainty, and precipitation type from Ku/Ka-band radar swaths
(KuKaGMI for GMI/GPM, KuTMI for TMI/TRMM), regrids to a common equiangular grid
at the native radar resolution (determined from tc_primed_ifovs.yaml), and writes
one HDF5 snapshot file per observation in the project standard format. A
consolidated index.parquet is written to the dataset root at the end.

Run from the project root:
    python scripts/preprocess/tc_primed/prepare_radar.py [paths=jz] [submitit=false] [num_workers=4]
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

from scripts.preprocess.tc_primed.utils import list_tc_primed_overpass_files_by_sensat
from scripts.preprocess.utils.regridding import ResamplingError, regrid
from tcfuse.data.sources import Source, SourceKind, SourceMetadata
from tcfuse.utils.archive import submit_archive_job

# Sensor/satellite → (swath_name, [variable_names])
# Extracts near-surface precipitation rate, rate uncertainty, and precipitation type.
SENSAT_VARIABLES: dict[str, tuple[str, list[str]]] = {
    "GMI_GPM": (
        "KuKaGMI",
        ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"],
    ),
    "TMI_TRMM": (
        "KuTMI",
        ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"],
    ),
}


def _get_regridding_resolution(sensat: str, swath: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a given sensor/swath.

    Uses the minimum IFOV value as the target resolution, so the regridded grid
    preserves the finest detail available.

    Args:
        sensat: Sensor/satellite string (e.g. "GMI_GPM").
        swath: Swath identifier (e.g. "KuKaGMI").
        ifovs: IFOV lookup dict loaded from tc_primed_ifovs.yaml.

    Returns:
        Minimum IFOV value in km.
    """
    ifov_entry = ifovs[sensat][swath]
    # IFOV entry can be either a list of values or a dict {channel: values}.
    # We'll always take the minimum value across all channels.
    if isinstance(ifov_entry, dict):
        return min(min(vals) for vals in ifov_entry.values())
    return min(ifov_entry)


def _read_radar_swath(
    grp: Any, variables: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read latitude, longitude, and radar data arrays from a netCDF4 swath group.

    Args:
        grp: netCDF4 group for the radar swath (e.g. raw["radar_radiometer"]["KuKaGMI"]).
        variables: List of variable names to read.

    Returns:
        (lat, lon, data) where lat and lon are float64 (H, W) arrays and data maps
        each variable name to a float64 (H, W) array. Masked values are filled with NaN.
        Longitudes are normalised to [-180, 180].
    """
    lat = np.ma.filled(grp["latitude"][:].astype(float), np.nan)
    lon = (np.ma.filled(grp["longitude"][:].astype(float), np.nan) + 180) % 360 - 180
    data = {v: np.ma.filled(grp[v][:].astype(float), np.nan) for v in variables}
    return lat, lon, data


def process_radar_file(
    file: str | Path,
    sensat: str,
    ifovs: dict,
    sources_root: Path,
) -> dict[str, Any] | None:
    """Process one TC-PRIMED overpass file and write a standard HDF5 snapshot.

    Extracts near-surface precipitation rate, rate uncertainty, and precipitation type
    from the Ku/Ka-band radar swath, regrids to an equiangular grid at native radar
    resolution, and writes a single HDF5 file containing one FIELD source named
    ``radar_{sensor_lower}``.

    The snapshot is written to
    ``{sources_root}/{source_name}/snapshots/{storm_id}_{time}.h5``.

    Args:
        file: Path to the raw TC-PRIMED NetCDF4 overpass file.
        sensat: Sensor/satellite string, e.g. "GMI_GPM".
        ifovs: IFOV lookup dict loaded from tc_primed_ifovs.yaml.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).

    Returns:
        Index row dict for inclusion in the per-source index.parquet, or None if discarded.
    """
    swath, variables = SENSAT_VARIABLES[sensat]
    # Map sensat to simplified source name: "GMI_GPM" → "radar_gmi", "TMI_TRMM" → "radar_tmi"
    sensor_abbrev = sensat.split("_")[0].lower()
    source_name = f"radar_{sensor_abbrev}"
    channels = [v.lower() for v in variables]

    with Dataset(str(file)) as raw:
        # --- Overpass and storm metadata ---
        meta_grp = raw["overpass_metadata"]
        season = int(meta_grp["season"][0])
        basin = str(meta_grp["basin"][0])
        storm_number = int(meta_grp["cyclone_number"][-1])
        storm_id = f"{basin}{storm_number:02d}{season}"
        time_unix_s = float(meta_grp["time"][0])

        # Retrieve storm info interpolated from best track
        storm_grp = raw["overpass_storm_metadata"]
        storm_lat = float(storm_grp["storm_latitude"][0])
        storm_lon = (float(storm_grp["storm_longitude"][0]) + 180) % 360 - 180
        vmax_kt = float(storm_grp["intensity"][0])
        min_pressure_hpa = float(storm_grp["central_min_pressure"][0])
        development_lvl = str(storm_grp["development_level"][0])
        storm_speed_ms = float(storm_grp["storm_speed"][0])
        storm_heading_deg = float(storm_grp["storm_heading"][0])

        # --- Check if radar data is available ---
        if "radar_radiometer" not in raw.groups:
            return None
        radar_grp = raw["radar_radiometer"]
        if int(radar_grp["availability_flag"][0]) == 0:
            return None
        if swath not in radar_grp.groups:
            return None

        # --- Read radar swath ---
        lat, lon, data = _read_radar_swath(radar_grp[swath], variables)
        if any(np.all(np.isnan(arr)) for arr in data.values()):
            return None

        # --- Regrid to equiangular grid ---
        regridding_res = _get_regridding_resolution(sensat, swath, ifovs)
        try:
            (resampled, out_lats, out_lons), target_area = regrid(lat, lon, data, regridding_res)
        except ResamplingError as exc:
            raise RuntimeError(f"Radar regrid failed for {file}") from exc

        # --- Stack channels in order and prepare coordinates ---
        values_np = np.stack([resampled[v] for v in variables], axis=-1).astype(np.float32)
        lats = out_lats.astype(np.float32)  # (H, W)
        lons = out_lons.astype(np.float32)  # (H, W)

    # --- Build Source (file closed; all data is in numpy arrays) ---
    src_h, src_w = lats.shape
    time_broadcast = np.full((src_h, src_w), time_unix_s, dtype=np.float32)
    coords_np = np.stack([time_broadcast, lats, lons], axis=-1)  # (H, W, 3)
    # A pixel is valid only when all channels are non-NaN
    mask_np = ~np.isnan(values_np).any(axis=-1)  # (H, W)

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
) -> list[dict[str, Any] | None]:
    """Process all radar files for one sensat using ProcessPoolExecutor.

    Designed to be submitted as a single submitit job or called directly
    for local execution.

    Args:
        files: Raw TC-PRIMED overpass file paths to process.
        sensat: Sensor/satellite string.
        ifovs: IFOV lookup dict.
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        num_workers: Number of parallel worker processes.

    Returns:
        List of index row dicts (None entries for discarded snapshots).
    """
    if num_workers <= 1:
        return [
            process_radar_file(f, sensat, ifovs, sources_root) for f in tqdm(files, desc=sensat)
        ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    process_radar_file,
                    files,
                    repeat(sensat),
                    repeat(ifovs),
                    repeat(sources_root),
                    chunksize=max(1, len(files) // (num_workers * 4)),
                ),
                total=len(files),
                desc=sensat,
            )
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Preprocess all TC-PRIMED radar snapshots to the standard HDF5 format."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    tc_primed_path = Path(cfg["paths"]["raw_datasets"]["tc_primed"])
    ifovs_path = tc_primed_path / "tc_primed_ifovs.yaml"
    sources_root = Path(cfg["paths"]["preprocessed_sources"])

    with open(ifovs_path) as f:
        ifovs: dict = yaml.safe_load(f)

    radar_files = list_tc_primed_overpass_files_by_sensat(
        tc_primed_path, include_seasons=cfg.get("include_seasons")
    )

    launch_local = not bool(cfg.get("submitit", False))
    num_workers = int(cfg.get("num_workers", 4))

    # Collect rows per source name so each source gets its own index.parquet.
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    channels_by_source: dict[str, list[str]] = {}

    if launch_local:
        for sensat, files in radar_files.items():
            if sensat not in SENSAT_VARIABLES:
                print(f"Skipping unsupported sensor: {sensat}")
                continue
            print(f"Processing {sensat} ({len(files)} files)…")
            results: list[dict[str, Any] | None] = _process_sensat_files(
                files, sensat, ifovs, sources_root, num_workers
            )
            valid = [r for r in results if r is not None]
            discarded = len(results) - len(valid)
            if discarded:
                pct = 100.0 * discarded / max(len(results), 1)
                print(f"  Discarded {discarded}/{len(results)} ({pct:.1f}%)")
            # Map sensat to source name
            sensor_abbrev = sensat.split("_")[0].lower()
            source_name = f"radar_{sensor_abbrev}"
            _, variables = SENSAT_VARIABLES[sensat]
            channels_by_source[source_name] = [v.lower() for v in variables]
            rows_by_source.setdefault(source_name, []).extend(valid)
    else:
        from tcfuse.utils.submitit_utils import make_executor as make_submitit_executor

        slurm_executor = make_submitit_executor(cfg, "prepare_radar")
        jobs: dict[str, Any] = {}
        for sensat, files in radar_files.items():
            if sensat not in SENSAT_VARIABLES:
                print(f"Skipping unsupported sensor: {sensat}")
                continue
            print(f"Submitting {sensat} ({len(files)} files)…")
            jobs[sensat] = slurm_executor.submit(
                _process_sensat_files, files, sensat, ifovs, sources_root, num_workers
            )
        for sensat, job in tqdm(jobs.items(), desc="collecting results"):
            results = job.result()
            valid = [r for r in results if r is not None]
            discarded = len(results) - len(valid)
            if discarded:
                pct = 100.0 * discarded / max(len(results), 1)
                print(f"  {sensat}: discarded {discarded}/{len(results)} ({pct:.1f}%)")
            # Map sensat to source name
            sensor_abbrev = sensat.split("_")[0].lower()
            source_name = f"radar_{sensor_abbrev}"
            _, variables = SENSAT_VARIABLES[sensat]
            channels_by_source[source_name] = [v.lower() for v in variables]
            rows_by_source.setdefault(source_name, []).extend(valid)

    if rows_by_source:
        for source_name, rows in rows_by_source.items():
            index_df = pd.DataFrame(rows)
            index_path = sources_root / source_name / "index.parquet"
            index_df.to_parquet(index_path, index=False)
            print(f"Wrote index ({source_name}): {len(index_df)} rows → {index_path}")
            source_meta = SourceMetadata(
                source_name, "radar", SourceKind.FIELD, channels_by_source[source_name], index_df
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
