"""Shared helpers for TC-PRIMED preprocessing."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any

import yaml

from scripts.preprocess.utils.regridding import grid_shape_for_extent, normalize_longitude_deg

# Hand-maintained IFOV table shipped with the repo (SENSAT → SWATH → VAR → [4 floats]).
TC_PRIMED_IFOVS_YAML = Path(__file__).resolve().parent / "tc_primed_ifovs.yaml"

# Default half-width (km) of storm-centered PMW/radar grids when not overridden in config.
DEFAULT_STORM_GRID_EXTENT_HALF_KM = 750.0


def read_tc_primed_overpass_meta(raw: Any) -> dict[str, Any]:
    """Read storm and overpass metadata from an open TC-PRIMED NetCDF dataset."""
    # Overpass-level metadata: season, basin, cyclone number history, overpass time.
    meta_grp = raw["overpass_metadata"]
    season = int(meta_grp["season"][0])
    basin = str(meta_grp["basin"][0])
    # cyclone_number is a short history vector; the active number is the last entry.
    storm_number = int(meta_grp["cyclone_number"][-1])
    # TC-PRIMED storm_id matches ATCF format: basin + zero-padded number + season.
    storm_id = f"{basin}{storm_number:02d}{season}"
    time_unix_s = float(meta_grp["time"][0])

    # Storm center position at overpass time (used for storm-centered regridding).
    storm_grp = raw["overpass_storm_metadata"]
    storm_lat = float(storm_grp["storm_latitude"][0])
    storm_lon = normalize_longitude_deg(float(storm_grp["storm_longitude"][0]))

    return {
        "storm_id": storm_id,
        "basin": basin,
        "season": season,
        "time_unix_s": time_unix_s,
        "storm_lat": storm_lat,
        "storm_lon": storm_lon,
    }


def load_tc_primed_ifovs() -> dict[str, dict[str, dict[str, list[float]]]]:
    """Load IFOV table from the repo; keys are SENSAT → SWATH → VAR → [4 floats]."""
    with open(TC_PRIMED_IFOVS_YAML) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    # Skip YAML comment keys (prefixed with underscore).
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def storm_grid_extent_half_km_from_cfg(cfg: dict[str, Any]) -> float:
    """Return half-width (km) of storm-centered PMW/radar grids from preproc config."""
    tc_primed = cfg.get("tc_primed") or {}
    return float(tc_primed.get("storm_grid_extent_half_km", DEFAULT_STORM_GRID_EXTENT_HALF_KM))


def get_regridding_resolution(sensat: str, swath: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a sensor/swath pair."""
    swath_entry = ifovs[sensat][swath]
    # Use the finest IFOV across all variables and their four footprint components.
    return min(min(vals) for vals in swath_entry.values())


def get_storm_centered_grid_shape(
    sensat: str,
    swath: str,
    ifovs: dict,
    extent_half_km: float = DEFAULT_STORM_GRID_EXTENT_HALF_KM,
) -> tuple[int, int]:
    """Return fixed output grid shape ``(height, width)`` for a storm-centered snapshot."""
    resolution_km = get_regridding_resolution(sensat, swath, ifovs)
    return grid_shape_for_extent(extent_half_km, resolution_km)


def list_tc_primed_overpass_files_by_sensat(
    tc_primed_path: Path,
    include_seasons: list[int] | None = None,
) -> dict[str, list[Path]]:
    """List overpass files grouped by sensor / satellite pair (e.g. ``AMSR2_GCOMW1``)."""
    overpass_files, _ = list_tc_primed_storm_files(tc_primed_path, include_seasons=include_seasons)
    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for file in chain(*overpass_files.values()):
        # Filename layout: {season}_{basin}_{number}_{sensor}_{satellite}_....nc
        sensat_pair = "_".join(file.stem.split("_")[3:5])
        grouped_files[sensat_pair].append(file)
    return grouped_files


def list_tc_primed_storm_files(
    tc_primed_path: Path,
    include_seasons: list[int] | None = None,
) -> tuple[dict[tuple[str, str, str], list[Path]], dict[tuple[str, str, str], list[Path]]]:
    """List TC-PRIMED files per storm, split into overpass and environment files."""
    # Walk the raw TC-PRIMED tree: {season}/{basin}/{storm_number}/*.nc
    storm_files: dict[tuple[str, str, str], list[Path]] = {}
    for year in tc_primed_path.iterdir():
        if include_seasons is not None and int(year.stem) not in include_seasons:
            continue
        for basin in year.iterdir():
            for number in basin.iterdir():
                storm_files[(year.stem, basin.stem, number.stem)] = list(number.iterdir())

    # Split each storm's files into overpass vs environment (ERA5) NetCDFs.
    overpass_files: dict[tuple[str, str, str], list[Path]] = {}
    environment_files: dict[tuple[str, str, str], list[Path]] = {}
    for key, files in storm_files.items():
        overpass_files[key] = [
            file
            for file in files
            if "era5" not in file.stem and "env" not in file.stem and file.suffix == ".nc"
        ]
        environment_files[key] = [
            file
            for file in files
            if ("era5" in file.stem or "env" in file.stem) and file.suffix == ".nc"
        ]

    return overpass_files, environment_files
