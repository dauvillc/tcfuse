"""Implements functions to manipulate gridded data."""

import logging
import math
import traceback
import warnings
from typing import cast

import numpy as np
from pyresample.area_config import create_area_def
from pyresample.bilinear._numpy_resampler import NumpyBilinearResampler
from pyresample.geometry import AreaDefinition, DynamicAreaDefinition, SwathDefinition
from pyresample.image import ImageContainerNearest
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


def regrid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: dict[str, np.ndarray],
    target_resolution: float,
    target_area: AreaDefinition | None = None,
) -> tuple[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray], AreaDefinition]:
    """Regrids swath data to a regular equiangular grid at the given resolution.

    Uses an equiangular (Plate Carrée) projection.

    Args:
        lat: Source latitude array of shape (H, W).
        lon: Source longitude array of shape (H, W).
        data: Dict mapping variable names to arrays of shape (H, W) or (H, W, C).
        target_resolution: Target resolution in km (converted to degrees at the equator).
        has_channel_dimension: Whether arrays in data have a trailing channel dimension.
        target_area: If provided, the target AreaDefinition to use. If None, one is
            created from the swath extent and target_resolution.
        return_area: Whether to return the target area definition alongside the result.

    Returns:
        (resampled_data, out_lats, out_lons) where resampled_data maps each key in data
        to its resampled array, and out_lats/out_lons are (H', W') coordinate arrays.
        If return_area is True, returns ((resampled_data, out_lats, out_lons), target_area).
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
        lon_span = lon.max() - lon.min()
        if lon_span > 180:
            lon = np.where(lon < 0, lon + 360, lon) - 180
            lons_were_shifted = True

    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km

    if target_area is None:
        proj_dict = {
            "proj": "longlat",
            "a": EARTH_RADIUS,
        }

        circumference = 2 * math.pi * EARTH_RADIUS
        meters_per_degree = circumference / 360.0
        res_degrees = (target_resolution * 1000) / meters_per_degree

        with DisableLogger():
            target_area_def = cast(
                DynamicAreaDefinition,
                create_area_def(
                    area_id="dynamic_equiangular",
                    projection=proj_dict,
                    resolution=res_degrees,
                    units="degrees",
                    shape=None,
                ),
            )

        target_area = target_area_def.freeze(swath, antimeridian_mode="modify_extents")

    resampler = NumpyBilinearResampler(swath, target_area, radius_of_influence)
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
        out_lons += 180.0
        out_lons = np.where(out_lons > 180.0, out_lons - 360.0, out_lons)

    result = (resampled_vars, out_lats, out_lons)
    return result, target_area


def regrid_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: dict[str, np.ndarray],
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> dict[str, np.ndarray]:
    """Regrids swath data to a given target grid using nearest-neighbour resampling.

    Args:
        lat: Source latitude array of shape (H, W).
        lon: Source longitude array of shape (H, W).
        data: Dict mapping variable names to arrays of shape (H, W) or (H, W, C).
        grid_lat: Target latitude grid of shape (H', W').
        grid_lon: Target longitude grid of shape (H', W').
        has_channel_dimension: Whether arrays in data have a trailing channel dimension.

    Returns:
        Dict mapping each key in data to its resampled array on the target grid.
    """
    lon, lat = check_and_wrap(lon, lat)

    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km
    target_swath = SwathDefinition(lons=grid_lon, lats=grid_lat)

    resampled_vars: dict[str, np.ndarray] = {}
    for var, arr in data.items():
        try:
            resampler = ImageContainerNearest(
                arr,
                swath,
                radius_of_influence,
                fill_value=float("nan"),  # type: ignore
            )
            resampled_vars[var] = resampler.resample(target_swath).image_data
        except Exception as e:
            print("Longitude:", lon)
            print("Latitude:", lat)
            traceback.print_tb(e.__traceback__)
            traceback.print_exc()
            raise ResamplingError(f"Error resampling variable {var}") from e

    return resampled_vars
