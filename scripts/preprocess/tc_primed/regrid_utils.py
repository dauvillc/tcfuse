"""Shared regridding helpers for TC-PRIMED prepare scripts."""


def get_regridding_resolution(sensat: str, swath: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a sensor/swath pair."""
    swath_entry = ifovs[sensat][swath]
    if not isinstance(swath_entry, dict):
        raise TypeError(
            f"IFOV entry at {sensat}/{swath} must be VAR → [4 floats], "
            f"got {type(swath_entry).__name__}"
        )
    return min(min(vals) for vals in swath_entry.values())
