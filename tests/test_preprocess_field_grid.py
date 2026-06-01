"""Unit tests for native-grid center crop / NaN padding (field_grid.py)."""

import numpy as np
import pytest

from scripts.preprocess.utils.field_grid import center_crop_or_pad_2d, center_fixed_length_1d


class TestCenterFixedLength1d:
    """1-D center crop and symmetric NaN padding."""

    def test_pad_split(self) -> None:
        """Odd padding total splits with floor on the left."""
        out = center_fixed_length_1d(np.arange(398, dtype=float), 401)
        assert out.shape == (401,)
        assert np.isnan(out[0])
        assert np.isnan(out[-2:]).all()
        assert out[1] == 0.0
        assert out[-3] == 397.0


class TestCenterCropOrPad2d:
    """2-D storm-centered fixed-size grids."""

    def test_center_crop(self) -> None:
        """Oversized grids are center-cropped."""
        h, w = 801, 801
        target = 401
        field = np.arange(h * w, dtype=float).reshape(h, w)
        cropped = center_crop_or_pad_2d(target, target, field)[0]
        assert cropped.shape == (target, target)
        assert cropped[target // 2, target // 2] == field[h // 2, w // 2]

    def test_symmetric_pad_square(self) -> None:
        """Undersized square grids are padded to the target with NaN borders."""
        h, w = 350, 350
        target = 401
        pad = (target - h) // 2
        field = np.ones((h, w))
        lat = np.linspace(10, 20, h)[:, None] * np.ones((1, w))
        out_field, out_lat = center_crop_or_pad_2d(target, target, field, lat)
        assert out_field.shape == (target, target)
        assert np.isnan(out_field[:pad, :]).all()
        assert out_field[pad, pad] == 1.0
        assert out_lat[pad, pad] == lat[0, 0]

    def test_mixed_crop_and_pad(self) -> None:
        """One axis cropped and the other padded."""
        field = np.ones((500, 300))
        out = center_crop_or_pad_2d(401, 401, field)[0]
        assert out.shape == (401, 401)

    def test_pad_split_width(self) -> None:
        """Width 398 → 401 uses pad_before=1, pad_after=2."""
        field = np.zeros((100, 398))
        out = center_crop_or_pad_2d(100, 401, field)[0]
        assert out.shape == (100, 401)
        assert np.isnan(out[:, 0]).all()
        assert np.isnan(out[:, -2:]).all()

    def test_hursat_size(self) -> None:
        """201×201 target from 401×401 native grid."""
        field = np.ones((401, 401))
        out = center_crop_or_pad_2d(201, 201, field)[0]
        assert out.shape == (201, 201)
