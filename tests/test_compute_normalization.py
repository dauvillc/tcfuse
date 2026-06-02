"""Tests for normalization preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
from pathlib import Path

import h5py

from scripts.preprocess.compute_normalization import _ensure_unbatched_group, _flatten_values_and_mask
from tcfuse.data.sources import SourceKind


def test_flatten_values_and_mask_preserves_channel_availability() -> None:
    """One missing channel should not invalidate the whole field pixel."""
    values = np.array(
        [
            [[1.0, np.nan], [2.0, 20.0]],
            [[3.0, 30.0], [np.nan, 40.0]],
        ],
        dtype=np.float32,
    )
    mask = np.isfinite(values)

    flat_values, flat_mask = _flatten_values_and_mask(values, mask, SourceKind.FIELD)

    first_channel = flat_values[:, 0][flat_mask[:, 0]]
    second_channel = flat_values[:, 1][flat_mask[:, 1]]

    assert first_channel.tolist() == [1.0, 2.0, 3.0]
    assert second_channel.tolist() == [20.0, 30.0, 40.0]


def test_ensure_unbatched_group_rejects_batched_source() -> None:
    """Normalization helpers should fail fast on batched Source snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tmp.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("snapshot")
            group.attrs["batched"] = True
            with pytest.raises(ValueError, match="normalization only supports non-batched"):
                _ensure_unbatched_group(group, "sid/time/source")


def test_ensure_unbatched_group_requires_batched_attr() -> None:
    """Normalization helpers should require explicit batched metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tmp.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("snapshot")
            with pytest.raises(ValueError, match="missing mandatory 'batched'"):
                _ensure_unbatched_group(group, "sid/time/source")
