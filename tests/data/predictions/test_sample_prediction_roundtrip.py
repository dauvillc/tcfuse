"""Round-trip tests for the SamplePrediction HDF5 I/O layer.

All tests use synthetic tensors — no real data required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from tcfuse.data.predictions import SamplePrediction
from tcfuse.data.sources import Source, SourceKind
from tests.test_sources import make_field_source, make_scalar_source

_SAMPLE_ID = "2016AL10_20160912T000000Z"
_STORM_ID = "2016AL10"
_INIT_TIME = "2016-09-12T00:00:00"
_BASIN = "AL"
_SEASON = 2016
_RUN_ID = "perceiver_pmw_ri_20160912"

_VALID_TIME_0 = "2016-09-12T00:00:00"
_VALID_TIME_6 = "2016-09-12T06:00:00"


def _make_sample(
    pred_sources: dict[tuple[str, str], Source] | None = None,
    target_sources: dict[tuple[str, str], Source] | None = None,
    *,
    atcf_id: str | None = None,
    run_id: str | None = _RUN_ID,
) -> SamplePrediction:
    """Convenience constructor for SamplePrediction test instances."""
    return SamplePrediction(
        sample_id=_SAMPLE_ID,
        storm_id=_STORM_ID,
        init_time_utc=_INIT_TIME,
        basin=_BASIN,
        season=_SEASON,
        atcf_id=atcf_id,
        run_id=run_id,
        pred_sources=pred_sources or {},
        target_sources=target_sources or {},
    )


def _write_read(sample: SamplePrediction) -> SamplePrediction:
    """Write a SamplePrediction to a temp run dir and read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        sample.write(run_root)
        return SamplePrediction.from_disk(run_root, sample.sample_id)


# ---------------------------------------------------------------------------
# Canonical path
# ---------------------------------------------------------------------------


class TestSamplePredictionPath:
    def test_path_structure(self) -> None:
        run_root = Path("/runs/run42")
        path = SamplePrediction.path(run_root, _SAMPLE_ID)
        assert path == Path("/runs/run42/samples/2016AL10_20160912T000000Z.h5")

    def test_path_uses_sample_id_as_stem(self) -> None:
        path = SamplePrediction.path(Path("/runs/r"), "2023EP05_20230815T120000Z")
        assert path.stem == "2023EP05_20230815T120000Z"
        assert path.suffix == ".h5"


# ---------------------------------------------------------------------------
# Round-trip: tensor data + metadata
# ---------------------------------------------------------------------------


class TestSamplePredictionRoundTrip:
    def test_pred_field_values_preserved(self) -> None:
        src = make_field_source(H=4, W=4, C=2)
        sample = _make_sample(pred_sources={("pmw_ssmi", _VALID_TIME_0): src})
        result = _write_read(sample)
        recovered = result.pred_sources[("pmw_ssmi", _VALID_TIME_0)]
        assert np.allclose(recovered.values, src.values, atol=1e-5)

    def test_pred_field_coords_preserved(self) -> None:
        src = make_field_source(H=4, W=4, C=2)
        sample = _make_sample(pred_sources={("pmw_ssmi", _VALID_TIME_0): src})
        result = _write_read(sample)
        recovered = result.pred_sources[("pmw_ssmi", _VALID_TIME_0)]
        assert np.allclose(recovered.coords, src.coords, atol=1e-5)

    def test_target_field_values_preserved(self) -> None:
        target = make_field_source(H=4, W=4, C=2)
        sample = _make_sample(target_sources={("pmw_ssmi", _VALID_TIME_0): target})
        result = _write_read(sample)
        recovered = result.target_sources[("pmw_ssmi", _VALID_TIME_0)]
        assert np.allclose(recovered.values, target.values, atol=1e-5)

    def test_pred_and_target_coexist(self) -> None:
        # Writing the same source under pred/ and target/ creates two distinct copies.
        pred = make_field_source(H=4, W=4, C=2)
        target = make_field_source(H=4, W=4, C=2)
        sample = _make_sample(
            pred_sources={("pmw_ssmi", _VALID_TIME_0): pred},
            target_sources={("pmw_ssmi", _VALID_TIME_0): target},
        )
        result = _write_read(sample)
        assert ("pmw_ssmi", _VALID_TIME_0) in result.pred_sources
        assert ("pmw_ssmi", _VALID_TIME_0) in result.target_sources
        assert np.allclose(
            result.pred_sources[("pmw_ssmi", _VALID_TIME_0)].values,
            pred.values,
            atol=1e-5,
        )
        assert np.allclose(
            result.target_sources[("pmw_ssmi", _VALID_TIME_0)].values,
            target.values,
            atol=1e-5,
        )

    def test_scalar_source_round_trip(self) -> None:
        src = make_scalar_source(C=2)
        src.channels = ["usa_vmax_kt", "usa_mslp_hpa"]
        sample = _make_sample(pred_sources={("ibtracs_best_track", _VALID_TIME_0): src})
        result = _write_read(sample)
        recovered = result.pred_sources[("ibtracs_best_track", _VALID_TIME_0)]
        assert recovered.kind is SourceKind.SCALAR
        assert recovered.channels == ["usa_vmax_kt", "usa_mslp_hpa"]
        assert np.allclose(recovered.values, src.values, atol=1e-6)

    def test_mask_preserved(self) -> None:
        src = make_field_source(H=4, W=4, C=2)
        mask = np.ones((4, 4, 2), dtype=bool)
        mask[1, 2, 0] = False
        src.mask = mask
        sample = _make_sample(pred_sources={("pmw_ssmi", _VALID_TIME_0): src})
        result = _write_read(sample)
        recovered = result.pred_sources[("pmw_ssmi", _VALID_TIME_0)]
        assert not recovered.mask[1, 2, 0]
        assert recovered.mask[0, 0, 0]

    def test_multi_source_multi_time(self) -> None:
        # Two sources x two valid times = four entries; all must survive.
        sources = {
            ("pmw_ssmi", _VALID_TIME_0): make_field_source(H=4, W=4, C=2),
            ("pmw_ssmi", _VALID_TIME_6): make_field_source(H=4, W=4, C=2),
            ("ir_geo", _VALID_TIME_0): make_field_source(H=4, W=4, C=1),
            ("ir_geo", _VALID_TIME_6): make_field_source(H=4, W=4, C=1),
        }
        sample = _make_sample(pred_sources=sources)
        result = _write_read(sample)
        assert len(result.pred_sources) == 4
        for key in sources:
            assert key in result.pred_sources

    def test_root_attrs_preserved(self) -> None:
        sample = _make_sample(
            pred_sources={("pmw_ssmi", _VALID_TIME_0): make_field_source()},
            atcf_id="AL102016",
        )
        result = _write_read(sample)
        assert result.sample_id == _SAMPLE_ID
        assert result.storm_id == _STORM_ID
        assert result.init_time_utc == _INIT_TIME
        assert result.basin == _BASIN
        assert result.season == _SEASON
        assert isinstance(result.season, int)
        assert result.atcf_id == "AL102016"
        assert result.run_id == _RUN_ID

    def test_atcf_id_optional(self) -> None:
        sample = _make_sample(pred_sources={("pmw_ssmi", _VALID_TIME_0): make_field_source()})
        result = _write_read(sample)
        assert result.atcf_id is None

    def test_predicted_source_names(self) -> None:
        sample = _make_sample(
            pred_sources={
                ("pmw_ssmi", _VALID_TIME_0): make_field_source(),
                ("pmw_ssmi", _VALID_TIME_6): make_field_source(),
                ("ir_geo", _VALID_TIME_0): make_field_source(),
            }
        )
        # Same source name across multiple snapshots yields one unique entry.
        assert sample.predicted_source_names == ["ir_geo", "pmw_ssmi"]

    def test_empty_sample_round_trip(self) -> None:
        # A sample with no predictions and no targets should still round-trip.
        sample = _make_sample()
        result = _write_read(sample)
        assert result.pred_sources == {}
        assert result.target_sources == {}
        assert result.sample_id == _SAMPLE_ID


# ---------------------------------------------------------------------------
# read_meta
# ---------------------------------------------------------------------------


class TestReadMeta:
    def test_read_meta_returns_root_attrs(self) -> None:
        sample = _make_sample(
            pred_sources={("pmw_ssmi", _VALID_TIME_0): make_field_source(H=16, W=16, C=4)},
            atcf_id="AL102016",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            sample.write(run_root)
            meta = SamplePrediction.read_meta(run_root, _SAMPLE_ID)

        assert meta["sample_id"] == _SAMPLE_ID
        assert meta["storm_id"] == _STORM_ID
        assert meta["basin"] == _BASIN
        assert meta["season"] == _SEASON
        assert meta["atcf_id"] == "AL102016"
        assert meta["run_id"] == _RUN_ID

    def test_read_meta_does_not_include_extras(self) -> None:
        sample = _make_sample(
            pred_sources={("pmw_ssmi", _VALID_TIME_0): make_field_source()},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            sample.write(run_root)
            meta = SamplePrediction.read_meta(run_root, _SAMPLE_ID)

        # atcf_id is absent from the input so it should not appear in the read result.
        assert "atcf_id" not in meta
        # Only documented root attrs are returned.
        assert set(meta.keys()) <= {
            "sample_id",
            "storm_id",
            "init_time_utc",
            "basin",
            "season",
            "atcf_id",
            "run_id",
        }
