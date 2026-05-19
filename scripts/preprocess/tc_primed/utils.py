"""Small helpers for TC-PRIMED preprocessing."""

from collections import defaultdict
from itertools import chain
from pathlib import Path


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
        if year.stem == "tc_primed_ifovs":
            continue
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
