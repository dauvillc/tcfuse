"""Round-trip tests for the Source HDF5 I/O layer.

All tests use synthetic tensors — no real data required.
Tests import the source factories defined in test_sources.py.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tcfuse.data.sources import Source, SourceKind
from tests.test_sources import make_field_source, make_profile_source, make_scalar_source

# ---------------------------------------------------------------------------
# Shared per-item storm metadata used across tests
# ---------------------------------------------------------------------------

_META = {
    "storm_id": "AL012020",
    "basin": "AL",
    "snapshot_time_utc": "2020-08-01T12:00:00+00:00",
    "lat": 25.0,
    "lon": -80.0,
    "vmax_kt": 65.0,
    "mslp_hpa": 975.0,
}


def _write_read(source: Source) -> Source:
    """Write source to a temp HDF5 file and read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        source.write(path)
        return Source.from_disk(path)


# ---------------------------------------------------------------------------
# Round-trip: values and coords
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_scalar_values_match(self) -> None:
        src = make_scalar_source(C=4)
        result = _write_read(src)
        assert torch.allclose(result.values, src.values, atol=1e-6)

    def test_scalar_coords_match(self) -> None:
        src = make_scalar_source()
        result = _write_read(src)
        assert torch.allclose(result.coords.double(), src.coords.double(), atol=1e-9)

    def test_profile_values_match(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read(src)
        assert torch.allclose(result.values, src.values, atol=1e-6)

    def test_profile_coords_match(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read(src)
        assert torch.allclose(result.coords.double(), src.coords.double(), atol=1e-9)

    def test_field_values_match(self) -> None:
        src = make_field_source(H=12, W=10, C=4)
        result = _write_read(src)
        assert torch.allclose(result.values, src.values, atol=1e-5)

    def test_field_coords_match(self) -> None:
        src = make_field_source(H=12, W=10, C=4)
        result = _write_read(src)
        assert torch.allclose(result.coords, src.coords.float(), atol=1e-5)

    def test_source_kind_preserved_scalar(self) -> None:
        result = _write_read(make_scalar_source())
        assert result.kind is SourceKind.SCALAR

    def test_source_kind_preserved_profile(self) -> None:
        result = _write_read(make_profile_source())
        assert result.kind is SourceKind.PROFILE

    def test_source_kind_preserved_field(self) -> None:
        result = _write_read(make_field_source())
        assert result.kind is SourceKind.FIELD

    def test_source_name_preserved(self) -> None:
        src = make_scalar_source(source_name="era5_surface")
        result = _write_read(src)
        assert result.source_name == "era5_surface"

    def test_nan_values_preserved(self) -> None:
        src = make_profile_source(L=8, C=3)
        src.values[2, 1] = float("nan")
        result = _write_read(src)
        assert math.isnan(float(result.values[2, 1]))


# ---------------------------------------------------------------------------
# Meta round-trip
# ---------------------------------------------------------------------------


class TestMeta:
    def test_meta_written_and_read_back(self) -> None:
        src = make_scalar_source()
        src.meta = dict(_META)
        result = _write_read(src)
        assert result.meta["storm_id"] == "AL012020"
        assert result.meta["basin"] == "AL"
        assert float(result.meta["vmax_kt"]) == pytest.approx(65.0)

    def test_empty_meta_no_error(self) -> None:
        src = make_field_source()
        src.meta = {}
        result = _write_read(src)
        assert result.meta == {}

    def test_read_meta_static_method(self) -> None:
        src = make_scalar_source()
        src.meta = {"storm_id": "EP052021", "lat": 18.5}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            src.write(path)
            meta = Source.read_meta(path)
        assert meta["storm_id"] == "EP052021"
        assert float(meta["lat"]) == pytest.approx(18.5)


# ---------------------------------------------------------------------------
# Canonical path helper
# ---------------------------------------------------------------------------


class TestPath:
    def test_path_structure(self) -> None:
        sources_root = Path("/data/preprocessed")
        p = Source.path(sources_root, "pmw_amsr2", "2016AL10", "20160912T010942Z")
        assert p == Path("/data/preprocessed/pmw_amsr2/snapshots/2016AL10_20160912T010942Z.h5")


# ---------------------------------------------------------------------------
# Mask round-trip
# ---------------------------------------------------------------------------


class TestMaskRoundTrip:
    def test_mask_written_and_read_back(self) -> None:
        L, C = 10, 4
        mask = torch.ones(L, C, dtype=torch.bool)
        mask[3, :] = False
        src = Source(
            kind=SourceKind.PROFILE,
            values=torch.randn(L, C),
            coords=torch.randn(L, 4),
            source_name="dropsonde_masked",
            channels=[f"ch{i}" for i in range(C)],
            mask=mask,
        )
        result = _write_read(src)
        assert result.mask is not None
        assert not result.mask[3, 0]
        assert result.mask[0, 0]

    def test_explicit_mask_round_trips_for_unmasked_values(self) -> None:
        src = make_scalar_source()
        assert src.mask is not None
        result = _write_read(src)
        assert result.mask is not None
        assert result.mask.shape == result.values.shape
        assert torch.equal(result.mask, torch.isfinite(result.values))

    def test_mask_always_written(self) -> None:
        src = make_scalar_source()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            src.write(path)
            with h5py.File(path, "r") as f:
                assert "mask" in f["scalar"][src.source_name]

    def test_missing_mask_raises_when_reading_legacy_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.h5"
            with h5py.File(path, "w") as f:
                group = f.create_group("scalar/best_track")
                group.create_dataset(
                    "values",
                    data=np.array([1.0, np.nan, 3.0], dtype=np.float32),
                )
                group.create_dataset(
                    "coords",
                    data=np.array([0.0, 25.0, -80.0], dtype=np.float64),
                )
                group.attrs["source_name"] = "best_track"
                group.attrs["channels"] = '["a", "b", "c"]'
                group.attrs["char_vars"] = "{}"

            with pytest.raises(ValueError, match="missing mandatory 'mask'"):
                Source.from_disk(path)


# ---------------------------------------------------------------------------
# Channels round-trip
# ---------------------------------------------------------------------------


class TestChannelsRoundTrip:
    def test_channels_written_and_read_back(self) -> None:
        src = make_scalar_source(C=2, source_name="best_track")
        src.channels[0] = "vmax_kt"
        src.channels[1] = "mslp_hpa"
        result = _write_read(src)
        assert result.channels == ["vmax_kt", "mslp_hpa"]

    def test_channels_preserved_for_field_source(self) -> None:
        src = make_field_source(C=4, source_name="pmw_amsr2")
        result = _write_read(src)
        assert result.channels == src.channels


