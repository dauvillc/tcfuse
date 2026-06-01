"""Shared regridding helpers for TC-PRIMED prepare scripts."""

from typing import Any

from scripts.preprocess.utils.regridding import grid_shape_for_extent

DEFAULT_STORM_GRID_EXTENT_HALF_KM = 750.0


def storm_grid_extent_half_km_from_cfg(cfg: dict[str, Any]) -> float:
    """Return half-width (km) of storm-centered PMW/radar grids from preproc config."""
    tc_primed = cfg.get("tc_primed") or {}
    return float(tc_primed.get("storm_grid_extent_half_km", DEFAULT_STORM_GRID_EXTENT_HALF_KM))


def get_regridding_resolution(sensat: str, swath: str, ifovs: dict) -> float:
    """Return the target regridding resolution (km) for a sensor/swath pair."""
    swath_entry = ifovs[sensat][swath]
    if not isinstance(swath_entry, dict):
        raise TypeError(
            f"IFOV entry at {sensat}/{swath} must be VAR → [4 floats], "
            f"got {type(swath_entry).__name__}"
        )
    # Use the finest IFOV across all variables and their four footprint components.
    return min(min(vals) for vals in swath_entry.values())


def get_storm_centered_grid_shape(
    sensat: str,
    swath: str,
    ifovs: dict,
    extent_half_km: float = DEFAULT_STORM_GRID_EXTENT_HALF_KM,
) -> tuple[int, int]:
    """Return fixed output grid shape for a storm-centered PMW/radar snapshot.

    Args:
        sensat: TC-PRIMED sensor/satellite id (e.g. ``GMI_GPM``).
        swath: Swath group name used for IFOV lookup.
        ifovs: IFOV table from :func:`load_tc_primed_ifovs`.
        extent_half_km: Half-width of the grid in km along each axis.

    Returns:
        ``(height, width)`` for the equiangular target grid.
    """
    resolution_km = get_regridding_resolution(sensat, swath, ifovs)
    return grid_shape_for_extent(extent_half_km, resolution_km)
