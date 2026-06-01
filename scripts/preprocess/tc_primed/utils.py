"""Small helpers for TC-PRIMED preprocessing."""

from collections import defaultdict
from itertools import chain
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

TC_PRIMED_IFOVS_YAML = Path(__file__).resolve().parent / "tc_primed_ifovs.yaml"
IFOV_COMPONENT_COUNT = 4


def _validate_ifov_entry(path: str, value: object) -> None:
    """Raise ValueError if an IFOV entry is not a length-4 list of numbers."""
    if not isinstance(value, list):
        raise ValueError(f"IFOV entry at {path} must be a list, got {type(value).__name__}")
    if len(value) != IFOV_COMPONENT_COUNT:
        raise ValueError(
            f"IFOV entry at {path} must have {IFOV_COMPONENT_COUNT} components, got {len(value)}"
        )
    if not all(isinstance(component, Real) for component in value):
        raise ValueError(f"IFOV entry at {path} must contain numeric values")


def _validate_ifovs_table(ifovs: dict[str, Any]) -> None:
    """Validate SENSAT → SWATH → VAR → [4 floats] structure."""
    for sensat, swaths in ifovs.items():
        if not isinstance(swaths, dict):
            raise ValueError(
                f"IFOV swaths for {sensat} must be a dict, got {type(swaths).__name__}"
            )
        for swath, variables in swaths.items():
            swath_path = f"{sensat}/{swath}"
            if not isinstance(variables, dict):
                raise ValueError(
                    f"IFOV variables at {swath_path} must be a dict, got {type(variables).__name__}"
                )
            for var_name, ifov_values in variables.items():
                _validate_ifov_entry(f"{swath_path}/{var_name}", ifov_values)


def load_tc_primed_ifovs() -> dict[str, dict[str, dict[str, list[float]]]]:
    """Load IFOV table from the repo; keys are SENSAT → SWATH → VAR → [4 floats]."""
    with open(TC_PRIMED_IFOVS_YAML) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    ifovs = {key: value for key, value in raw.items() if not key.startswith("_")}
    _validate_ifovs_table(ifovs)
    return ifovs


def should_skip_existing(dest_path: Path, skip_existing: bool) -> bool:
    """Return True if an existing output snapshot should not be reprocessed."""
    return skip_existing and dest_path.exists()


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
    storm_files: dict[tuple[str, str, str], list[Path]] = {}
    for year in tc_primed_path.iterdir():
        if include_seasons is not None and int(year.stem) not in include_seasons:
            continue
        for basin in year.iterdir():
            for number in basin.iterdir():
                storm_files[(year.stem, basin.stem, number.stem)] = list(number.iterdir())

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
