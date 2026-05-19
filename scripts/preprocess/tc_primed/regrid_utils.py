"""Shared regridding helpers for TC-PRIMED prepare scripts."""


def get_regridding_resolution(sensat: str, swath: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a sensor/swath pair."""
    ifov_entry = ifovs[sensat][swath]
    if isinstance(ifov_entry, dict):
        return min(min(vals) for vals in ifov_entry.values())
    return min(ifov_entry)
