"""Unit tests for infrared preprocessing (prepare_infrared.py)."""

import numpy as np
from scripts.preprocess.tc_primed.prepare_infrared import IR_CENTER_CROP_HALF_WIDTH_PX
from scripts.preprocess.utils.field_grid import center_crop_or_pad_2d


class TestIrFixedGridSize:
    """Storm-centered fixed-size IR grids via shared field_grid helpers."""

    def test_tcirar_crop_size(self) -> None:
        """TC IRAR keeps ±200 px around the image center (401x401)."""
        half = IR_CENTER_CROP_HALF_WIDTH_PX["ir_tcirar"]
        side = 2 * half + 1
        h, w = 801, 801
        irwin = np.arange(h * w, dtype=float).reshape(h, w)
        lat = np.linspace(10, 20, h)[:, None] * np.ones((1, w))
        lon = np.linspace(-80, -70, w)[None, :] * np.ones((h, 1))
        cropped = center_crop_or_pad_2d(side, side, irwin, lat, lon)
        assert cropped[0].shape == (side, side)
        assert cropped[0][half, half] == irwin[h // 2, w // 2]

    def test_hursat_crop_size(self) -> None:
        """HURSAT keeps ±100 px around the image center (201x201)."""
        half = IR_CENTER_CROP_HALF_WIDTH_PX["ir_hursat"]
        side = 2 * half + 1
        h, w = 401, 401
        irwin = np.ones((h, w))
        cropped = center_crop_or_pad_2d(side, side, irwin)
        assert cropped[0].shape == (side, side)

    def test_pads_field_smaller_than_target(self) -> None:
        """Undersized native grids are NaN-padded to the target size."""
        half = IR_CENTER_CROP_HALF_WIDTH_PX["ir_tcirar"]
        side = 2 * half + 1
        irwin = np.ones((100, 100))
        padded = center_crop_or_pad_2d(side, side, irwin)[0]
        assert padded.shape == (side, side)
        pad = (side - 100) // 2
        assert np.isnan(padded[:pad, :]).all()
        assert padded[pad, pad] == 1.0
