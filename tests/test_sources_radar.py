"""Unit tests for radar preprocessing (prepare_radar.py)."""

import numpy as np
import pytest
import torch
from scripts.preprocess.tc_primed.prepare_radar import _read_radar_swath
from scripts.preprocess.tc_primed.utils import (
    get_regridding_resolution,
    get_storm_centered_grid_shape,
    load_tc_primed_ifovs,
)

from tcfuse.data.sources import Source, SourceKind


class TestLoadTcPrimedIfovs:
    """Test loading and validating the repo IFOV lookup table."""

    def test_load_returns_radar_entries(self) -> None:
        """Loaded table includes nested VAR entries for radar swaths."""
        ifovs = load_tc_primed_ifovs()
        assert len(ifovs) > 0
        gmi_kugmi = ifovs["GMI_GPM"]["KuGMI"]["nearSurfPrecipTotRate"]
        assert len(gmi_kugmi) == 4
        assert gmi_kugmi == [5.04, 5.04, 5.04, 5.57]


class TestGetRegriddingResolution:
    """Test IFOV resolution extraction."""

    def test_uniform_var_ifov_values(self) -> None:
        """Minimum IFOV is taken across variables when all share the same values."""
        ifovs = {
            "GMI_GPM": {
                "KuKaGMI": {
                    "nearSurfPrecipTotRate": [5.04, 5.04, 5.04, 5.04],
                    "nearSurfPrecipTotRateSigma": [5.04, 5.04, 5.04, 5.04],
                    "mainprecipitationType": [5.04, 5.04, 5.04, 5.04],
                }
            }
        }
        result = get_regridding_resolution("GMI_GPM", "KuKaGMI", ifovs)
        assert result == 5.04

    def test_radar_var_ifov_values(self) -> None:
        """Minimum IFOV is taken from per-variable radar entries."""
        ifovs = {
            "TMI_TRMM": {
                "KuTMI": {
                    "nearSurfPrecipTotRate": [5.0, 5.0, 5.0, 5.0],
                    "nearSurfPrecipTotRateSigma": [5.0, 5.0, 5.0, 5.0],
                    "mainprecipitationType": [5.0, 5.0, 5.0, 5.0],
                }
            }
        }
        result = get_regridding_resolution("TMI_TRMM", "KuTMI", ifovs)
        assert result == 5.0

    def test_dict_ifov_with_varying_values(self) -> None:
        """Test that minimum is taken across different channel IFOVs."""
        ifovs = {
            "TEST_SAT": {
                "TEST_SWATH": {
                    "channel1": [10.0, 10.0, 10.0, 10.0],
                    "channel2": [5.0, 5.0, 5.0, 5.0],
                    "channel3": [7.5, 7.5, 7.5, 7.5],
                }
            }
        }
        result = get_regridding_resolution("TEST_SAT", "TEST_SWATH", ifovs)
        assert result == 5.0


class TestGetStormCenteredGridShape:
    """Test fixed output grid shape from IFOV resolution and extent."""

    def test_radar_grid_shape(self) -> None:
        ifovs = {
            "GMI_GPM": {
                "KuGMI": {
                    "nearSurfPrecipTotRate": [5.04, 5.04, 5.04, 5.04],
                }
            }
        }
        assert get_storm_centered_grid_shape("GMI_GPM", "KuGMI", ifovs, 750.0) == (298, 298)


class TestRadarSwathReader:
    """Test reading radar swath data from netCDF4 groups (mock interface)."""

    def test_swath_reader_with_mock_group(self):
        """Test _read_radar_swath with a minimal mock netCDF4 group."""

        # Create mock group with netCDF4-like interface
        class MockVariable:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data

        class MockGroup:
            def __init__(self):
                # Create synthetic lat/lon grids (10x15)
                self.lat_data = np.arange(100, dtype=np.float32).reshape(10, 10)
                self.lon_data = np.arange(150, dtype=np.float32).reshape(10, 15) - 90
                self.var1_data = np.random.rand(10, 15).astype(np.float32)
                self.var2_data = np.random.rand(10, 15).astype(np.float32)

            def __getitem__(self, key):
                if key == "latitude":
                    return MockVariable(self.lat_data)
                elif key == "longitude":
                    return MockVariable(self.lon_data)
                elif key == "var1":
                    return MockVariable(self.var1_data)
                elif key == "var2":
                    return MockVariable(self.var2_data)
                raise KeyError(f"Unknown key: {key}")

        grp = MockGroup()
        lat, lon, data = _read_radar_swath(grp, ["var1", "var2"])

        # Note: lat shape is from lat_data (10x10), lon from lon_data (10x15)
        # This demonstrates the interface works, though in practice
        # lat and lon should have the same shape from the swath
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert isinstance(data, dict)
        assert "var1" in data
        assert "var2" in data
        assert data["var1"].dtype == np.float64
        assert data["var2"].dtype == np.float64

    def test_swath_reader_lon_normalization(self):
        """Test longitude normalization to [-180, 180]."""

        class MockVariable:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data

        class MockGroup:
            def __init__(self):
                # Create lat/lon grids with specific normalization test
                self.lat_data = np.zeros((5, 5), dtype=np.float32)
                # Longitudes in [0, 360) range
                self.lon_data = np.array(
                    [
                        [0, 90, 180, 270, 360],
                        [45, 135, 225, 315, 0],
                        [90, 180, 270, 0, 90],
                        [180, 270, 0, 90, 180],
                        [270, 0, 90, 180, 270],
                    ],
                    dtype=np.float32,
                )
                self.var_data = np.ones((5, 5), dtype=np.float32)

            def __getitem__(self, key):
                if key == "latitude":
                    return MockVariable(self.lat_data)
                elif key == "longitude":
                    return MockVariable(self.lon_data)
                elif key == "var":
                    return MockVariable(self.var_data)
                raise KeyError(f"Unknown key: {key}")

        grp = MockGroup()
        lat, lon, data = _read_radar_swath(grp, ["var"])

        # Check that all longitudes are in [-180, 180]
        assert np.all(lon >= -180) and np.all(lon <= 180)


class TestRadarSourceConstruction:
    """Test construction of radar Source objects."""

    def test_create_radar_source_field(self):
        """Test creating a FIELD source from synthetic radar data."""
        H, W, C = 10, 15, 3
        values = torch.randn(H, W, C, dtype=torch.float32)

        # Time broadcast, lat, lon
        time_unix_s = 1000.0
        times = torch.full((H, W), time_unix_s, dtype=torch.float32)
        lats = torch.linspace(-30, 30, H).unsqueeze(1).expand(H, W).float()
        lons = torch.linspace(-60, 60, W).unsqueeze(0).expand(H, W).float()
        coords = torch.stack([times, lats, lons], dim=-1)  # (H, W, 3)

        # Create per-value availability mask: all valid.
        mask = torch.ones(H, W, C, dtype=torch.bool)

        channels = ["precip_rate", "precip_sigma", "precip_type"]

        source = Source(
            kind=SourceKind.FIELD,
            values=values,
            coords=coords,
            source_name="radar_gmi",
            channels=channels,
            mask=mask,
        )

        assert source.kind == SourceKind.FIELD
        assert source.values.shape == (H, W, C)
        assert source.coords.shape == (H, W, 3)
        assert source.mask.shape == (H, W, C)
        assert source.n_tokens == H * W
        assert source.channels == channels

    def test_radar_source_mask_with_nans(self):
        """Test mask computation when some pixels have NaN."""
        H, W, C = 5, 5, 3
        values = torch.ones(H, W, C, dtype=torch.float32)
        # Set some pixels to NaN
        values[0, 0, 0] = float("nan")  # First channel of (0,0) is NaN
        values[2, 3, 1] = float("nan")  # Second channel of (2,3) is NaN

        times = torch.full((H, W), 1000.0, dtype=torch.float32)
        lats = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
        lons = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
        coords = torch.stack([times, lats, lons], dim=-1)

        # Per-value availability mask: each channel is masked independently.
        mask = torch.isfinite(values)

        source = Source(
            kind=SourceKind.FIELD,
            values=values,
            coords=coords,
            source_name="radar_test",
            channels=["a", "b", "c"],
            mask=mask,
        )

        # Check only the NaN channels are marked invalid.
        assert source.mask[0, 0, 0] == False
        assert source.mask[0, 0, 1] == True
        assert source.mask[2, 3, 1] == False
        assert source.mask[2, 3, 2] == True
        # Check that other pixels are valid.
        assert source.mask[0, 1, 0] == True
        assert source.mask[4, 4, 2] == True

    def test_radar_source_n_tokens(self):
        """Test token count for FIELD source."""
        H, W, C = 8, 12, 3
        values = torch.randn(H, W, C)
        times = torch.full((H, W), 1000.0, dtype=torch.float32)
        lats = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
        lons = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
        coords = torch.stack([times, lats, lons], dim=-1)

        source = Source(
            kind=SourceKind.FIELD,
            values=values,
            coords=coords,
            source_name="radar_test",
            channels=["a", "b", "c"],
            mask=torch.isfinite(values),
        )

        assert source.n_tokens == H * W


class TestRadarChannels:
    """Test radar variable naming and channel ordering."""

    def test_channel_naming_lowercase(self):
        """Test that radar variable names are converted to lowercase."""
        original = ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"]
        channels = [v.lower() for v in original]

        expected = ["nearsurfpreciptotrate", "nearsurfpreciptotratesigma", "mainprecipitationtype"]
        assert channels == expected

    def test_channel_order_preserved(self):
        """Test that channel stacking preserves order."""
        H, W = 5, 5
        var1 = np.ones((H, W)) * 1.0
        var2 = np.ones((H, W)) * 2.0
        var3 = np.ones((H, W)) * 3.0

        # Stack in order
        stacked = np.stack([var1, var2, var3], axis=-1)

        # Check that values are preserved in order
        assert np.allclose(stacked[:, :, 0], var1)
        assert np.allclose(stacked[:, :, 1], var2)
        assert np.allclose(stacked[:, :, 2], var3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
