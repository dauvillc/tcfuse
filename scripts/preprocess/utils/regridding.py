"""Implements functions to manipulate gridded data."""

import logging
import math
import traceback
import warnings

import numpy as np
from pyproj.exceptions import ProjError
from pyresample.area_config import create_area_def
from pyresample.bilinear._numpy_resampler import NumpyBilinearResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.utils import check_and_wrap


class DisableLogger:
    """Context manager to disable logging temporarily."""

    def __enter__(self):
        logging.basicConfig(level=logging.ERROR, force=True)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.basicConfig(level=logging.INFO, force=True)


class ResamplingError(ValueError):
    """Exception raised when resampling operations fail."""

    pass


# Disable some warnings that pyresample and CRS are raising
# but can be safely ignored
warnings.simplefilter(action="ignore")


EARTH_RADIUS = 6371228.0  # Earth radius in meters
BILINEAR_RADIUS_OF_INFLUENCE_M = 100_000.0  # 100 km


def _resolution_km_to_degrees(target_resolution_km: float) -> float:
    """Convert target grid spacing in km to degrees at the equator."""
    circumference = 2 * math.pi * EARTH_RADIUS
    meters_per_degree = circumference / 360.0
    return (target_resolution_km * 1000.0) / meters_per_degree


def normalize_longitude_deg(lon: float) -> float:
    """Map longitude to ``[-180, 180]``, using ``180`` instead of ``-180``.

    Pyresample's ``create_area_def`` with ``longlat`` projection produces invalid
    extents when the centre longitude is exactly ``-180°``.
    """
    # Wrap to the standard half-open interval [-180, 180).
    wrapped = (lon + 180.0) % 360.0 - 180.0
    # Collapse the antimeridian duplicate meridian to +180 for pyresample.
    if wrapped == -180.0:
        return 180.0
    return wrapped


def grid_shape_for_extent(extent_half_km: float, resolution_km: float) -> tuple[int, int]:
    """Return (height, width) for a storm-centered grid spanning ±extent_half_km.

    Args:
        extent_half_km: Half-width of the grid in km along each axis (e.g. 750).
        resolution_km: Target pixel spacing in km.

    Returns:
        ``(2 * round(extent_half_km / resolution_km), ...)`` for pyresample ``shape``.
    """
    n_half = round(extent_half_km / resolution_km)
    size = 2 * n_half
    return (size, size)


def create_storm_centered_equiangular_area(
    storm_lon: float,
    storm_lat: float,
    target_resolution_km: float,
    *,
    extent_half_km: float = 750.0,
) -> AreaDefinition:
    """Build a fixed equiangular grid centered on the storm position.

    Uses Plate Carrée (``longlat``) with pixel spacing derived from
    ``target_resolution_km`` at the equator.

    Args:
        storm_lon: Storm center longitude in degrees (``[-180, 180]``).
        storm_lat: Storm center latitude in degrees.
        target_resolution_km: Target pixel spacing in km.
        extent_half_km: Half-width of the grid in km along each cardinal direction.

    Returns:
        Frozen ``AreaDefinition`` for bilinear resampling.
    """
    proj_dict = {
        "proj": "longlat",
        "a": EARTH_RADIUS,
    }
    grid_shape = grid_shape_for_extent(extent_half_km, target_resolution_km)
    res_degrees = _resolution_km_to_degrees(target_resolution_km)
    # Avoid pyresample failure at the antimeridian meridian (-180 vs +180).
    center_lon = normalize_longitude_deg(storm_lon)

    with DisableLogger():
        area = create_area_def(
            area_id="storm_centered_equiangular",
            projection=proj_dict,
            center=(center_lon, storm_lat),
            shape=grid_shape,
            resolution=res_degrees,
            units="degrees",
        )
    if not isinstance(area, AreaDefinition):
        raise TypeError(f"Expected AreaDefinition from create_area_def, got {type(area).__name__}")
    return area


def _shift_area_by_180(area: AreaDefinition) -> AreaDefinition:
    """Return a copy of area with its centre longitude shifted by 180° (antimeridian fix)."""
    # For longlat projection, area_extent is in degrees: (lon_min, lat_min, lon_max, lat_max).
    lon_min, lat_min, lon_max, lat_max = area.area_extent
    centre_lon = (lon_min + lon_max) / 2
    # Same 180° shift as applied to the swath: maps ±180° region to near 0°.
    shifted_lon = (centre_lon + 360 if centre_lon < 0 else centre_lon) - 180
    delta = shifted_lon - centre_lon
    # Shift the extent directly instead of going through create_area_def, which involves a
    # resolution unit-conversion path (pixel_size_x → degrees) that can fail on some
    # pyresample versions or with unusual area geometries.
    return AreaDefinition(
        area.area_id,
        area.description,
        area.proj_id,
        {"proj": "longlat", "a": EARTH_RADIUS},
        area.x_size,
        area.y_size,
        (lon_min + delta, lat_min, lon_max + delta, lat_max),
    )


def regrid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: dict[str, np.ndarray],
    target_area: AreaDefinition,
) -> tuple[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray], AreaDefinition]:
    """Regrid swath data onto a fixed equiangular target area using bilinear resampling.

    Args:
        lat: Source latitude array of shape ``(H, W)``.
        lon: Source longitude array of shape ``(H, W)``.
        data: Dict mapping variable names to arrays of shape ``(H, W)`` or ``(H, W, C)``.
        target_area: Pre-built target ``AreaDefinition`` (e.g. from
            :func:`create_storm_centered_equiangular_area`).

    Returns:
        ``((resampled_data, out_lats, out_lons), target_area)`` where ``resampled_data``
        maps each key in ``data`` to its resampled array, and ``out_lats`` / ``out_lons``
        are ``(H', W')`` coordinate arrays on the target grid.
    """
    lon, lat = check_and_wrap(lon, lat)

    # Resampling can provoke errors for cases where the longitudes
    # cross the antimeridian. We have however the advantage that
    # we can assume that the longitudes here span over strictly less than 180°.
    # Therefore, if there are both negative and positive longitudes, they
    # either cross the 0° meridian or the 180° meridian.
    # - If they cross the 0° meridian, nothing to do it'll be handled correctly.
    # - If they cross the 180° meridian, we'll shift the longitudes to be all positive,
    #   then resample, then shift back.
    lons_were_shifted = False
    if np.any(lon < 0) and np.any(lon > 0):
        lon_span = np.nanmax(lon) - np.nanmin(lon)
        if lon_span > 180:
            lon = np.where(lon < 0, lon + 360, lon) - 180
            lons_were_shifted = True
            try:
                target_area = _shift_area_by_180(target_area)
            except ProjError as e:
                raise ResamplingError(
                    "Failed to shift target area for antimeridian crossing"
                ) from e

    # The storm-centered target area can extend past ±180° even when the swath stays
    # on one side (e.g. storm at 178°, target area reaches 184°). Pyresample drops
    # those out-of-range pixels, making bilinear_s/bilinear_t smaller than NxN, so
    # _reshape_to_target_area's np.reshape raises ValueError. Detect and fix here.
    if not lons_were_shifted:
        lon_min_area, _, lon_max_area, _ = target_area.area_extent
        if lon_max_area > 180.0 or lon_min_area < -180.0:
            centre_lon = (lon_min_area + lon_max_area) / 2.0
            delta = ((centre_lon + 360.0 if centre_lon < 0.0 else centre_lon) - 180.0) - centre_lon
            lon = lon + delta
            lons_were_shifted = True
            try:
                target_area = _shift_area_by_180(target_area)
            except ProjError as e:
                raise ResamplingError(
                    "Failed to shift target area for near-antimeridian storm"
                ) from e

    # Sanity check: guard against accidental high-resolution configs that would
    # produce enormous arrays and silently exhaust memory.
    out_h, out_w = target_area.shape
    if out_h > 1000 or out_w > 1000:
        raise ResamplingError(
            f"Regridding target area has shape ({out_h}, {out_w}), which exceeds the "
            f"1000-pixel safety limit in at least one dimension."
        )

    swath = SwathDefinition(lons=lon, lats=lat)

    resampler = NumpyBilinearResampler(swath, target_area, BILINEAR_RADIUS_OF_INFLUENCE_M)
    resampled_vars: dict[str, np.ndarray] = {}
    for var, arr in data.items():
        try:
            resampled_vars[var] = resampler.resample(
                arr,
                fill_value=float("nan"),  # type: ignore
            )
        except Exception as e:
            print("Longitude:", lon)
            print("Latitude:", lat)
            traceback.print_tb(e.__traceback__)
            traceback.print_exc()
            raise ResamplingError(f"Error resampling variable {var}") from e

    out_lons, out_lats = target_area.get_lonlats()
    if lons_were_shifted:
        # Restore the longitudes, but now without a seam at the antimeridian.
        # After that, longitudes can go slightly beyond 180° but will always be monotonic
        # and contiguous.
        out_lons += 180.0

    # Return shape: ((resampled_data, out_lats, out_lons), target_area).
    result = (resampled_vars, out_lats, out_lons)
    return result, target_area
