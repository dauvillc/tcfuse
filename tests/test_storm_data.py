"""Round-trip tests for the StormData HDF5 I/O layer.

All tests use synthetic tensors — no real data required.
Tests import the source factories defined in test_sources.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tcfuse.data.sources import Source, SourceKind, StormData
from tests.test_sources import make_field_source, make_profile_source, make_scalar_source

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_STORM_ID = "2016AL10"
_BASIN = "AL"
_SEASON = 2016
_TIME_0 = "2016-09-12T01:09:42+00:00"
_TIME_1 = "2016-09-14T15:30:12+00:00"


def _make_storm_data(sources: dict[tuple[str, str], Source]) -> StormData:
    """Convenience constructor for test StormData objects."""
    return StormData(storm_id=_STORM_ID, basin=_BASIN, season=_SEASON, sources=sources)


def _write_read(storm_data: StormData) -> StormData:
    """Write StormData to a temp directory and read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assembled_root = Path(tmpdir)
        storm_data.write(assembled_root)
        return StormData.from_disk(assembled_root, storm_data.storm_id)


# ---------------------------------------------------------------------------
# Canonical path helper
# ---------------------------------------------------------------------------


class TestStormDataPath:
    def test_path_structure(self) -> None:
        root = Path("/data/assembled")
        p = StormData.path(root, "2016AL10")
        assert p == Path("/data/assembled/storm_data/2016AL10.h5")

    def test_path_includes_storm_id_as_stem(self) -> None:
        p = StormData.path(Path("/out"), "2023EP05")
        assert p.stem == "2023EP05"
        assert p.suffix == ".h5"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestStormDataConstruction:
    def test_single_field_source(self) -> None:
        src = make_field_source()
        sd = _make_storm_data({("pmw_ssmi", _TIME_0): src})
        assert len(sd.sources) == 1

    def test_multi_source_multi_kind(self) -> None:
        # SCALAR + PROFILE + FIELD coexist in the same StormData.
        sources = {
            ("best_track", _TIME_0): make_scalar_source(),
            ("dropsonde", _TIME_0): make_profile_source(),
            ("pmw_ssmi", _TIME_0): make_field_source(),
        }
        sd = _make_storm_data(sources)
        assert len(sd.sources) == 3

    def test_same_source_two_snapshots(self) -> None:
        # Same instrument, two different overpass times → two distinct entries.
        sources = {
            ("pmw_ssmi", _TIME_0): make_field_source(),
            ("pmw_ssmi", _TIME_1): make_field_source(),
        }
        sd = _make_storm_data(sources)
        assert len(sd.sources) == 2
        assert ("pmw_ssmi", _TIME_0) in sd.sources
        assert ("pmw_ssmi", _TIME_1) in sd.sources

    def test_season_and_basin_attributes(self) -> None:
        sd = _make_storm_data({})
        assert sd.season == _SEASON
        assert sd.basin == _BASIN


# ---------------------------------------------------------------------------
# Round-trip: tensor data
# ---------------------------------------------------------------------------


class TestStormDataRoundTrip:
    def test_field_values_preserved(self) -> None:
        src = make_field_source(H=8, W=8, C=2)
        result = _write_read(_make_storm_data({("pmw_ssmi", _TIME_0): src}))
        recovered = result.sources[("pmw_ssmi", _TIME_0)]
        assert torch.allclose(recovered.values, src.values, atol=1e-5)

    def test_field_coords_preserved(self) -> None:
        src = make_field_source(H=8, W=8, C=2)
        result = _write_read(_make_storm_data({("pmw_ssmi", _TIME_0): src}))
        recovered = result.sources[("pmw_ssmi", _TIME_0)]
        assert torch.allclose(recovered.coords, src.coords.float(), atol=1e-5)

    def test_scalar_values_preserved(self) -> None:
        src = make_scalar_source(C=4)
        result = _write_read(_make_storm_data({("best_track", _TIME_0): src}))
        recovered = result.sources[("best_track", _TIME_0)]
        assert torch.allclose(recovered.values, src.values, atol=1e-6)

    def test_scalar_coords_preserved(self) -> None:
        src = make_scalar_source()
        result = _write_read(_make_storm_data({("best_track", _TIME_0): src}))
        recovered = result.sources[("best_track", _TIME_0)]
        assert torch.allclose(recovered.coords.double(), src.coords.double(), atol=1e-9)

    def test_profile_values_preserved(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read(_make_storm_data({("dropsonde", _TIME_0): src}))
        recovered = result.sources[("dropsonde", _TIME_0)]
        assert torch.allclose(recovered.values, src.values, atol=1e-6)

    def test_profile_coords_preserved(self) -> None:
        src = make_profile_source(L=15, C=6)
        result = _write_read(_make_storm_data({("dropsonde", _TIME_0): src}))
        recovered = result.sources[("dropsonde", _TIME_0)]
        assert torch.allclose(recovered.coords.double(), src.coords.double(), atol=1e-9)

    def test_mask_preserved(self) -> None:
        src = make_field_source(H=6, W=6, C=2)
        mask = torch.ones(6, 6, dtype=torch.bool)
        mask[2, 3] = False
        src.mask = mask
        result = _write_read(_make_storm_data({("pmw_ssmi", _TIME_0): src}))
        recovered = result.sources[("pmw_ssmi", _TIME_0)]
        assert recovered.mask is not None
        assert not recovered.mask[2, 3]
        assert recovered.mask[0, 0]

    def test_no_mask_when_none(self) -> None:
        src = make_scalar_source()
        assert src.mask is None
        result = _write_read(_make_storm_data({("best_track", _TIME_0): src}))
        recovered = result.sources[("best_track", _TIME_0)]
        assert recovered.mask is None

    def test_source_kind_preserved_field(self) -> None:
        result = _write_read(_make_storm_data({("pmw_ssmi", _TIME_0): make_field_source()}))
        assert result.sources[("pmw_ssmi", _TIME_0)].kind is SourceKind.FIELD

    def test_source_kind_preserved_profile(self) -> None:
        result = _write_read(_make_storm_data({("dropsonde", _TIME_0): make_profile_source()}))
        assert result.sources[("dropsonde", _TIME_0)].kind is SourceKind.PROFILE

    def test_source_kind_preserved_scalar(self) -> None:
        result = _write_read(_make_storm_data({("best_track", _TIME_0): make_scalar_source()}))
        assert result.sources[("best_track", _TIME_0)].kind is SourceKind.SCALAR

    def test_source_name_preserved(self) -> None:
        src = make_field_source(source_name="pmw_amsr2_gcomw1")
        result = _write_read(_make_storm_data({("pmw_amsr2_gcomw1", _TIME_0): src}))
        assert result.sources[("pmw_amsr2_gcomw1", _TIME_0)].source_name == "pmw_amsr2_gcomw1"

    def test_channels_preserved(self) -> None:
        src = make_scalar_source(C=2)
        src.channels = ["vmax_kt", "mslp_hpa"]
        result = _write_read(_make_storm_data({("best_track", _TIME_0): src}))
        assert result.sources[("best_track", _TIME_0)].channels == ["vmax_kt", "mslp_hpa"]

    def test_storm_id_preserved(self) -> None:
        result = _write_read(_make_storm_data({("best_track", _TIME_0): make_scalar_source()}))
        assert result.storm_id == _STORM_ID

    def test_basin_preserved(self) -> None:
        result = _write_read(_make_storm_data({("best_track", _TIME_0): make_scalar_source()}))
        assert result.basin == _BASIN

    def test_season_preserved(self) -> None:
        result = _write_read(_make_storm_data({("best_track", _TIME_0): make_scalar_source()}))
        assert result.season == _SEASON
        assert isinstance(result.season, int)

    def test_snapshot_time_utc_key_preserved(self) -> None:
        # The isoformat timestamp used as the dict key must survive the round-trip
        # exactly, including any timezone suffix.
        result = _write_read(_make_storm_data({("pmw_ssmi", _TIME_0): make_field_source()}))
        assert ("pmw_ssmi", _TIME_0) in result.sources

    def test_multi_source_multi_snapshot_round_trip(self) -> None:
        # 2 sources × 2 snapshots = 4 entries; all must survive.
        sources = {
            ("pmw_ssmi", _TIME_0): make_field_source(),
            ("pmw_ssmi", _TIME_1): make_field_source(),
            ("best_track", _TIME_0): make_scalar_source(),
            ("best_track", _TIME_1): make_scalar_source(),
        }
        result = _write_read(_make_storm_data(sources))
        assert len(result.sources) == 4
        for key in sources:
            assert key in result.sources

    def test_snapshot_meta_round_trip(self) -> None:
        # Source.meta entries (e.g. lat/lon) should survive as snapshot attrs.
        src = make_scalar_source()
        src.meta = {"lat": 25.3, "lon": -80.1, "vmax_kt": 65.0}
        result = _write_read(_make_storm_data({("best_track", _TIME_0): src}))
        recovered_meta = result.sources[("best_track", _TIME_0)].meta
        assert float(recovered_meta["lat"]) == pytest.approx(25.3, abs=1e-6)
        assert float(recovered_meta["vmax_kt"]) == pytest.approx(65.0, abs=1e-6)


# ---------------------------------------------------------------------------
# read_meta: lightweight root-attr access
# ---------------------------------------------------------------------------


class TestReadMeta:
    def test_read_meta_returns_root_attrs(self) -> None:
        sd = _make_storm_data({("best_track", _TIME_0): make_scalar_source()})
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            sd.write(assembled_root)
            meta = StormData.read_meta(assembled_root, _STORM_ID)
        assert meta["storm_id"] == _STORM_ID
        assert meta["basin"] == _BASIN
        assert int(meta["season"]) == _SEASON

    def test_read_meta_does_not_load_tensors(self) -> None:
        # read_meta should be fast / not require source data to be read.
        # We verify it returns only the three root-level keys.
        sd = _make_storm_data({("pmw_ssmi", _TIME_0): make_field_source(H=64, W=64, C=4)})
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            sd.write(assembled_root)
            meta = StormData.read_meta(assembled_root, _STORM_ID)
        assert set(meta.keys()) == {"storm_id", "basin", "season"}


# ---------------------------------------------------------------------------
# atcf_id round-trip
# ---------------------------------------------------------------------------


class TestAtcfId:
    def test_atcf_id_round_trips_when_set(self) -> None:
        sd = StormData(
            storm_id=_STORM_ID,
            basin=_BASIN,
            season=_SEASON,
            sources={("best_track", _TIME_0): make_scalar_source()},
            atcf_id="AL102016",
        )
        result = _write_read(sd)
        assert result.atcf_id == "AL102016"

    def test_atcf_id_is_none_when_absent(self) -> None:
        sd = _make_storm_data({("best_track", _TIME_0): make_scalar_source()})
        result = _write_read(sd)
        assert result.atcf_id is None

    def test_read_meta_includes_atcf_id_when_present(self) -> None:
        sd = StormData(
            storm_id=_STORM_ID,
            basin=_BASIN,
            season=_SEASON,
            sources={("best_track", _TIME_0): make_scalar_source()},
            atcf_id="AL102016",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            sd.write(assembled_root)
            meta = StormData.read_meta(assembled_root, _STORM_ID)
        assert meta.get("atcf_id") == "AL102016"
