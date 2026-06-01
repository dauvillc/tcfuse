"""Storm-centered crop and symmetric NaN padding on native regular 2-D grids."""

from __future__ import annotations

import numpy as np


def center_fixed_length_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or NaN-pad a 1-D array to ``target_len`` elements."""
    n = int(arr.shape[0])
    # Crop symmetrically when the axis is longer than the target.
    if n > target_len:
        start = (n - target_len) // 2
        return arr[start : start + target_len]
    # Pad with NaN when the axis is shorter than the target.
    if n < target_len:
        pad_total = target_len - n
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(arr, (pad_before, pad_after), constant_values=np.nan)
    return arr


def center_crop_or_pad_2d(
    target_h: int,
    target_w: int,
    *fields: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Center-crop or NaN-pad 2-D fields to ``(target_h, target_w)``.

    All inputs must share the same shape. Processing is row-first, then column-wise,
    with symmetric padding when an axis is shorter than the target.
    """
    if not fields:
        return ()

    # Fix height first, then width — keeps lat/lon aligned with data fields.
    result: tuple[np.ndarray, ...] = fields
    result = _center_fix_axis(result, axis=0, target=target_h)
    result = _center_fix_axis(result, axis=1, target=target_w)
    return result


def _center_fix_axis(
    fields: tuple[np.ndarray, ...],
    axis: int,
    target: int,
) -> tuple[np.ndarray, ...]:
    """Center-crop or NaN-pad all fields along one axis."""
    n = fields[0].shape[axis]
    if n > target:
        start = (n - target) // 2
        # Build a 2-D slice tuple selecting the center window along one axis.
        sl: list[slice] = [slice(None), slice(None)]
        sl[axis] = slice(start, start + target)
        return tuple(field[tuple(sl)] for field in fields)
    if n < target:
        pad_total = target - n
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pad_width: list[tuple[int, int]] = [(0, 0), (0, 0)]
        pad_width[axis] = (pad_before, pad_after)
        return tuple(np.pad(field, pad_width, constant_values=np.nan) for field in fields)
    return fields
