"""Unit tests for SAR preprocessing grid sizing."""

import numpy as np

from scripts.preprocess.sar.prepare_sar import SAR_CENTER_CROP_HALF_WIDTH_PX
from scripts.preprocess.utils.field_grid import center_crop_or_pad_2d


class TestSarFixedGridSize:
    """SAR snapshots are always 401×401 after meshgrid + center crop/pad."""

    def test_undersized_wind_field_pads_to_401(self) -> None:
        """Undersized native grids become 401×401 with NaN borders."""
        side = 2 * SAR_CENTER_CROP_HALF_WIDTH_PX + 1
        h, w = 300, 350
        wind = np.ones((h, w))
        lat_1d = np.linspace(10, 20, h, dtype=np.float32)
        lon_1d = np.linspace(-80, -70, w, dtype=np.float32)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        wind, lat_2d, lon_2d = center_crop_or_pad_2d(side, side, wind, lat_2d, lon_2d)
        assert wind.shape == (side, side)
        pad_y = (side - h) // 2
        pad_x = (side - w) // 2
        assert np.isnan(wind[:pad_y, :]).all()
        assert wind[pad_y, pad_x] == 1.0
