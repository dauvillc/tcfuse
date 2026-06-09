"""Unit tests for storm-centered equiangular regridding helpers."""

import math

import numpy as np
import pytest
from scripts.preprocess.tc_primed.utils import (
    get_storm_centered_grid_shape,
    storm_grid_extent_half_km_from_cfg,
)
from scripts.preprocess.utils.regridding import (
    EARTH_RADIUS,
    create_storm_centered_equiangular_area,
    grid_shape_for_extent,
    normalize_longitude_deg,
    regrid,
)


class TestGridShapeForExtent:
    """Tests for fixed storm-centered grid dimensions."""

    def test_shape_at_five_km_resolution(self) -> None:
        """750 km half-extent at 5 km/pixel yields 300x300 grid."""
        assert grid_shape_for_extent(750.0, 5.0) == (300, 300)

    def test_get_storm_centered_grid_shape_from_ifovs(self) -> None:
        """Grid shape combines IFOV resolution with extent half-width."""
        ifovs = {
            "GMI_GPM": {
                "S1": {
                    "TB_89.0H": [7.2, 4.4, 7.2, 4.4],
                    "TB_89.0V": [7.2, 4.4, 7.2, 4.4],
                }
            }
        }
        assert get_storm_centered_grid_shape("GMI_GPM", "S1", ifovs, 750.0) == (340, 340)


class TestStormGridExtentFromCfg:
    """Tests for preproc config lookup."""

    def test_default_when_missing(self) -> None:
        assert storm_grid_extent_half_km_from_cfg({}) == 750.0

    def test_reads_nested_tc_primed_key(self) -> None:
        cfg = {"tc_primed": {"storm_grid_extent_half_km": 600.0}}
        assert storm_grid_extent_half_km_from_cfg(cfg) == 600.0


class TestCreateStormCenteredEquiangularArea:
    """Tests for pyresample AreaDefinition construction."""

    def test_output_shape_matches_grid_shape(self) -> None:
        area = create_storm_centered_equiangular_area(-80.0, 25.0, 5.0, extent_half_km=750.0)
        lons, lats = area.get_lonlats()
        assert lons.shape == (300, 300)
        assert lats.shape == (300, 300)

    def test_center_pixel_near_storm(self) -> None:
        storm_lat, storm_lon = 25.0, -80.0
        area = create_storm_centered_equiangular_area(
            storm_lon, storm_lat, 5.0, extent_half_km=750.0
        )
        lons, lats = area.get_lonlats()
        cy, cx = 150, 150
        assert lats[cy, cx] == pytest.approx(storm_lat, abs=0.5)
        assert lons[cy, cx] == pytest.approx(storm_lon, abs=0.5)

    def test_lat_lon_span_matches_1500_km_at_equator(self) -> None:
        """Total span along each axis is ~1500 km when using equator degree spacing."""
        resolution_km = 5.0
        area = create_storm_centered_equiangular_area(0.0, 0.0, resolution_km, extent_half_km=750.0)
        lons, lats = area.get_lonlats()
        circumference = 2 * math.pi * EARTH_RADIUS
        meters_per_degree = circumference / 360.0
        expected_span_deg = (2 * 750.0 * 1000.0) / meters_per_degree
        assert (lats.max() - lats.min()) == pytest.approx(expected_span_deg, rel=0.02)
        assert (lons.max() - lons.min()) == pytest.approx(expected_span_deg, rel=0.02)


class TestNormalizeLongitudeDeg:
    """Tests for antimeridian-safe longitude normalization."""

    def test_wraps_positive_longitudes(self) -> None:
        assert normalize_longitude_deg(190.0) == pytest.approx(-170.0)

    def test_maps_minus_180_to_plus_180(self) -> None:
        assert normalize_longitude_deg(-180.0) == 180.0


class TestAntimeridianRegridding:
    """Regression tests for pyresample failures near the antimeridian."""

    def test_create_area_at_minus_180_meridian(self) -> None:
        """Storm centre at -180° must not produce broken pyresample extents."""
        area = create_storm_centered_equiangular_area(-180.0, 10.0, 5.0, extent_half_km=750.0)
        lon_min, lat_min, lon_max, lat_max = area.area_extent
        assert abs(lon_min) <= 360.0
        assert abs(lon_max) <= 360.0
        assert -90.0 <= lat_min <= 90.0
        assert -90.0 <= lat_max <= 90.0

    def test_regrid_swath_crossing_antimeridian(self) -> None:
        """Swath longitudes spanning 180° should regrid without ProjError."""
        storm_lon, storm_lat = 170.0, 20.0
        area = create_storm_centered_equiangular_area(storm_lon, storm_lat, 5.0)
        height, width = 40, 40
        lats = np.linspace(15.0, 25.0, height)[:, None] * np.ones(width)
        lons = np.linspace(160.0, 190.0, width)[None, :] * np.ones((height, 1))
        data = {"tb": np.ones((height, width), dtype=np.float64)}
        (resampled, out_lats, out_lons), _ = regrid(lats, lons, data, area)
        assert resampled["tb"].shape == out_lats.shape == out_lons.shape
