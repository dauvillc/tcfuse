"""Unit tests for infrared preprocessing (prepare_infrared.py)."""

import numpy as np
import pytest

from scripts.preprocess.tc_primed.prepare_infrared import (
    IR_CENTER_CROP_HALF_WIDTH_PX,
    _crop_center_square,
)


class TestCropCenterSquare:
    """Storm-centered crop on native IR grids."""

    def test_tcirar_crop_size(self) -> None:
        """TC IRAR keeps ±200 px around the image center (401×401)."""
        half = IR_CENTER_CROP_HALF_WIDTH_PX["ir_tcirar"]
        h, w = 801, 801
        irwin = np.arange(h * w, dtype=float).reshape(h, w)
        lat = np.linspace(10, 20, h)[:, None] * np.ones((1, w))
        lon = np.linspace(-80, -70, w)[None, :] * np.ones((h, 1))
        cropped = _crop_center_square(half, irwin, lat, lon)
        assert cropped[0].shape == (2 * half + 1, 2 * half + 1)
        assert cropped[0][half, half] == irwin[h // 2, w // 2]

    def test_hursat_crop_size(self) -> None:
        """HURSAT keeps ±100 px around the image center (201×201)."""
        half = IR_CENTER_CROP_HALF_WIDTH_PX["ir_hursat"]
        h, w = 401, 401
        irwin = np.ones((h, w))
        cropped = _crop_center_square(half, irwin)
        assert cropped[0].shape == (201, 201)

    def test_rejects_field_smaller_than_crop(self) -> None:
        """Oversized crop relative to the native grid raises ValueError."""
        with pytest.raises(ValueError, match="smaller than crop"):
            _crop_center_square(200, np.zeros((100, 100)))
