"""Shared TC-PRIMED overpass metadata readers."""

from __future__ import annotations

from typing import Any


def read_tc_primed_overpass_meta(raw: Any) -> dict[str, Any]:
    """Read storm and overpass metadata from an open TC-PRIMED NetCDF dataset."""
    meta_grp = raw["overpass_metadata"]
    season = int(meta_grp["season"][0])
    basin = str(meta_grp["basin"][0])
    # cyclone_number is a short history vector; the active number is the last entry.
    storm_number = int(meta_grp["cyclone_number"][-1])
    storm_id = f"{basin}{storm_number:02d}{season}"
    time_unix_s = float(meta_grp["time"][0])

    storm_grp = raw["overpass_storm_metadata"]
    storm_lat = float(storm_grp["storm_latitude"][0])
    storm_lon = (float(storm_grp["storm_longitude"][0]) + 180) % 360 - 180

    return {
        "storm_id": storm_id,
        "basin": basin,
        "season": season,
        "time_unix_s": time_unix_s,
        "storm_lat": storm_lat,
        "storm_lon": storm_lon,
    }
