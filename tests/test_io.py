"""Round-trip tests for the HDF5 I/O layer.

All tests use synthetic tensors — no real data required.
Tests import the source factories defined in test_sources.py.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from tcfuse.data.sources import Source, SourceKind, Snapshot
from tests.test_sources import make_field_source, make_profile_source, make_scalar_source

# ---------------------------------------------------------------------------
# Helpers
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


def _write_read(sources: dict[str, Source], **read_kwargs: object) -> Snapshot:
    """Write sources to a temp HDF5 file and read them back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "snapshots" / "AL012020_20200801T120000Z.h5"
        Snapshot(sources=sources, meta=_META).write(path)
        return Snapshot.from_disk(path, **read_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip: values and coords
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_scalar_values_match(self) -> None:
        src = make_scalar_source(C=4)
        result = _write_read({"best_track": src})
        assert torch.allclose(result["best_track"].values, src.values, atol=1e-6)

    def test_scalar_coords_match(self) -> None:
        src = make_scalar_source()
        result = _write_read({"best_track": src})
        assert torch.allclose(result["best_track"].coords.double(), src.coords.double(), atol=1e-9)

    def test_profile_values_match(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read({"dropsonde_001": src})
        assert torch.allclose(result["dropsonde_001"].values, src.values, atol=1e-6)

    def test_profile_coords_match(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read({"dropsonde_001": src})
        assert torch.allclose(
            result["dropsonde_001"].coords.double(), src.coords.double(), atol=1e-9
        )

    def test_field_values_match(self) -> None:
        src = make_field_source(H=12, W=10, C=4)
        result = _write_read({"pmw_amsr2": src})
        assert torch.allclose(result["pmw_amsr2"].values, src.values, atol=1e-5)

    def test_field_coords_match(self) -> None:
        src = make_field_source(H=12, W=10, C=4)
        result = _write_read({"pmw_amsr2": src})
        assert torch.allclose(result["pmw_amsr2"].coords, src.coords.float(), atol=1e-5)

    def test_source_kind_preserved_scalar(self) -> None:
        src = make_scalar_source()
        result = _write_read({"best_track": src})
        assert result["best_track"].kind is SourceKind.SCALAR

    def test_source_kind_preserved_profile(self) -> None:
        src = make_profile_source()
        result = _write_read({"dropsonde_001": src})
        assert result["dropsonde_001"].kind is SourceKind.PROFILE

    def test_source_kind_preserved_field(self) -> None:
        src = make_field_source()
        result = _write_read({"pmw_amsr2": src})
        assert result["pmw_amsr2"].kind is SourceKind.FIELD

    def test_source_name_preserved(self) -> None:
        src = make_scalar_source(source_name="era5_surface")
        result = _write_read({"era5_surface": src})
        assert result["era5_surface"].source_name == "era5_surface"

    def test_nan_values_preserved(self) -> None:
        src = make_profile_source(L=8, C=3)
        src.values[2, 1] = float("nan")
        result = _write_read({"dropsonde_nan": src})
        assert math.isnan(float(result["dropsonde_nan"].values[2, 1]))


# ---------------------------------------------------------------------------
# Multi-source snapshot
# ---------------------------------------------------------------------------


class TestMultiSource:
    def test_all_three_kinds_round_trip(self) -> None:
        sources = {
            "best_track": make_scalar_source(),
            "dropsonde_001": make_profile_source(),
            "pmw_amsr2": make_field_source(),
        }
        result = _write_read(sources)
        assert set(result) == {"best_track", "dropsonde_001", "pmw_amsr2"}

    def test_multiple_field_sources(self) -> None:
        sources = {
            "pmw_amsr2": make_field_source(H=8, W=8, C=4),
            "era5_surface": make_field_source(H=16, W=16, C=5, source_name="era5_surface"),
        }
        result = _write_read(sources)
        assert result["pmw_amsr2"].values.shape == (8, 8, 4)
        assert result["era5_surface"].values.shape == (16, 16, 5)


# ---------------------------------------------------------------------------
# Missing sources (absent HDF5 group)
# ---------------------------------------------------------------------------


class TestMissingSources:
    def test_absent_source_not_in_result(self) -> None:
        sources = {"best_track": make_scalar_source()}
        result = _write_read(sources)
        assert "pmw_amsr2" not in result

    def test_empty_snapshot_returns_empty_dict(self) -> None:
        result = _write_read({})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Partial read (source_names filter)
# ---------------------------------------------------------------------------


class TestPartialRead:
    def test_only_requested_sources_returned(self) -> None:
        sources = {
            "best_track": make_scalar_source(),
            "pmw_amsr2": make_field_source(),
            "era5_surface": make_field_source(source_name="era5_surface"),
        }
        result = _write_read(sources, source_names=["best_track", "era5_surface"])
        assert "best_track" in result
        assert "era5_surface" in result
        assert "pmw_amsr2" not in result

    def test_request_absent_source_returns_empty(self) -> None:
        sources = {"best_track": make_scalar_source()}
        result = _write_read(sources, source_names=["nonexistent"])
        assert len(result) == 0


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
        result = _write_read({"dropsonde_masked": src})
        assert result["dropsonde_masked"].mask is not None
        assert not result["dropsonde_masked"].mask[3, 0]
        assert result["dropsonde_masked"].mask[0, 0]

    def test_no_mask_when_none(self) -> None:
        src = make_scalar_source()
        assert src.mask is None
        result = _write_read({"best_track": src})
        assert result["best_track"].mask is None


# ---------------------------------------------------------------------------
# Channels round-trip
# ---------------------------------------------------------------------------


class TestChannelsRoundTrip:
    def test_channels_written_and_read_back(self) -> None:
        src = make_scalar_source(C=2, source_name="best_track")
        src.channels[0] = "vmax_kt"
        src.channels[1] = "mslp_hpa"
        result = _write_read({"best_track": src})
        assert result["best_track"].channels == ["vmax_kt", "mslp_hpa"]

    def test_channels_preserved_for_field_source(self) -> None:
        src = make_field_source(C=4, source_name="pmw_amsr2")
        result = _write_read({"pmw_amsr2": src})
        assert result["pmw_amsr2"].channels == src.channels
