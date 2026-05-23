"""Streaming round-trip tests for the PredictionRun writer/reader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tcfuse.data.predictions import (
    IBTRACS_LONG_COLUMNS,
    PredictionRun,
    SamplePrediction,
    build_long_rows,
)
from tcfuse.data.sources import Source
from tests.test_sources import make_field_source, make_scalar_source

_RUN_ID = "perceiver_pmw_ri_20160912"
_BASIN = "AL"
_SEASON = 2016
_INIT_TIME_0 = "2016-09-12T00:00:00"
_INIT_TIME_1 = "2016-09-12T06:00:00"
_INIT_TIME_2 = "2016-09-12T12:00:00"
_VALID_TIME_0 = "2016-09-12T00:00:00"
_VALID_TIME_6 = "2016-09-12T06:00:00"
_CHANNELS = ["usa_vmax_kt", "usa_mslp_hpa"]


def _make_manifest() -> dict[str, object]:
    """Minimal manifest used across the streaming tests."""
    return {
        "run_id": _RUN_ID,
        "model": {"name": "perceiver", "checkpoint": "/fake/ckpt.pt"},
        "split": "val",
        "leads_hours": [0, 6],
        "ibtracs_channels": _CHANNELS,
        "deterministic": True,
    }


def _make_sample(
    storm_id: str,
    init_time: str,
    *,
    pred_sources: dict[tuple[str, str], Source] | None = None,
    target_sources: dict[tuple[str, str], Source] | None = None,
) -> SamplePrediction:
    """Build a SamplePrediction for one window of one storm."""
    sample_id = f"{storm_id}_{pd.Timestamp(init_time):%Y%m%dT%H%M%SZ}"
    return SamplePrediction(
        sample_id=sample_id,
        storm_id=storm_id,
        init_time_utc=init_time,
        basin=_BASIN,
        season=_SEASON,
        run_id=_RUN_ID,
        pred_sources=pred_sources or {},
        target_sources=target_sources or {},
    )


def _make_ibtracs_block(sample: SamplePrediction) -> pd.DataFrame:
    """Build a tidy-long IBTrACS block matching the sample's window."""
    leads = [
        {
            "lead_hour": 0,
            "valid_time_utc": sample.init_time_utc,
            "pred": {"usa_vmax_kt": 50.0, "usa_mslp_hpa": 990.0},
            "target": {"usa_vmax_kt": 55.0, "usa_mslp_hpa": 985.0},
        },
        {
            "lead_hour": 6,
            "valid_time_utc": pd.Timestamp(sample.init_time_utc) + pd.Timedelta(hours=6),
            "pred": {"usa_vmax_kt": 60.0, "usa_mslp_hpa": 980.0},
            "target": {"usa_vmax_kt": 65.0, "usa_mslp_hpa": 975.0},
        },
    ]
    # Pandas Timestamp -> isoformat to keep schema strings.
    leads[1]["valid_time_utc"] = leads[1]["valid_time_utc"].isoformat()
    return build_long_rows(
        sample_id=sample.sample_id,
        storm_id=sample.storm_id,
        season=sample.season,
        basin=sample.basin,
        init_time_utc=sample.init_time_utc,
        leads=leads,
        channels=_CHANNELS,
    )


# ---------------------------------------------------------------------------
# Streaming write + reopen
# ---------------------------------------------------------------------------


class TestPredictionRunRoundTrip:
    def test_streaming_write_and_reopen(self) -> None:
        # 3 storms with intentionally varied predicted-source sets:
        #   - sample 0: PMW preds, IBTrACS rows
        #   - sample 1: PMW + IR preds, IBTrACS rows
        #   - sample 2: no source preds, IBTrACS rows only
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID

            samples_to_write: list[SamplePrediction] = [
                _make_sample(
                    "2016AL10",
                    _INIT_TIME_0,
                    pred_sources={
                        ("pmw_ssmi", _VALID_TIME_0): make_field_source(H=4, W=4, C=2),
                    },
                    target_sources={
                        ("pmw_ssmi", _VALID_TIME_0): make_field_source(H=4, W=4, C=2),
                    },
                ),
                _make_sample(
                    "2016AL10",
                    _INIT_TIME_1,
                    pred_sources={
                        ("pmw_ssmi", _VALID_TIME_6): make_field_source(H=4, W=4, C=2),
                        ("ir_geo", _VALID_TIME_6): make_field_source(H=4, W=4, C=1),
                    },
                ),
                _make_sample("2016EP05", _INIT_TIME_2),
            ]

            with PredictionRun.create(run_root, _make_manifest()) as run:
                for sample in samples_to_write:
                    run.add_sample(sample, _make_ibtracs_block(sample))

            # Reopen for reading and verify everything.
            reopened = PredictionRun.from_disk(run_root)

            # Index has one row per sample, in write order.
            index = reopened.index
            assert len(index) == 3
            assert index["sample_id"].tolist() == [s.sample_id for s in samples_to_write]
            assert bool(index["has_ibtracs_prediction"].all())

            # Predicted source names per sample match what we wrote.
            assert index.iloc[0]["predicted_source_names"].tolist() == ["pmw_ssmi"]
            assert index.iloc[1]["predicted_source_names"].tolist() == ["ir_geo", "pmw_ssmi"]
            assert index.iloc[2]["predicted_source_names"].tolist() == []

            # IBTrACS table covers all samples.
            ibtracs = reopened.ibtracs
            assert list(ibtracs.columns) == IBTRACS_LONG_COLUMNS
            assert len(ibtracs) == 3 * 2 * 2  # samples x leads x channels
            assert set(ibtracs["sample_id"].unique()) == {s.sample_id for s in samples_to_write}

            # load_sample returns the right per-window file with tensors intact.
            recovered_first = reopened.load_sample(samples_to_write[0].sample_id)
            assert ("pmw_ssmi", _VALID_TIME_0) in recovered_first.pred_sources
            assert ("pmw_ssmi", _VALID_TIME_0) in recovered_first.target_sources

            # iter_samples follows the index order.
            iterated_ids = [sample.sample_id for sample in reopened.iter_samples()]
            assert iterated_ids == [s.sample_id for s in samples_to_write]

    def test_run_without_ibtracs_writes_empty_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run_no_ibtracs"
            with PredictionRun.create(run_root, _make_manifest()) as run:
                sample = _make_sample(
                    "2016AL10",
                    _INIT_TIME_0,
                    pred_sources={("pmw_ssmi", _VALID_TIME_0): make_field_source()},
                )
                # Pass None to signal the model emitted no IBTrACS rows for this sample.
                run.add_sample(sample, None)

            reopened = PredictionRun.from_disk(run_root)
            assert reopened.ibtracs.empty
            assert list(reopened.ibtracs.columns) == IBTRACS_LONG_COLUMNS
            assert not reopened.index.iloc[0]["has_ibtracs_prediction"]

    def test_manifest_is_persisted_with_derived_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID
            with PredictionRun.create(run_root, _make_manifest()) as run:
                sample = _make_sample(
                    "2016AL10",
                    _INIT_TIME_0,
                    pred_sources={
                        ("pmw_ssmi", _VALID_TIME_0): make_field_source(),
                        ("ir_geo", _VALID_TIME_0): make_field_source(),
                    },
                )
                run.add_sample(sample, _make_ibtracs_block(sample))

            with (run_root / "manifest.yaml").open() as f:
                manifest = yaml.safe_load(f)

            assert manifest["run_id"] == _RUN_ID
            assert manifest["n_samples"] == 1
            assert manifest["predicted_sources"] == ["ir_geo", "pmw_ssmi"]
            # created_at_utc is auto-populated.
            assert "created_at_utc" in manifest

    def test_add_sample_after_close_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID
            run = PredictionRun.create(run_root, _make_manifest())
            sample = _make_sample(
                "2016AL10",
                _INIT_TIME_0,
                pred_sources={("ibtracs_best_track", _VALID_TIME_0): make_scalar_source()},
            )
            run.add_sample(sample, _make_ibtracs_block(sample))
            run.close()

            with pytest.raises(RuntimeError):
                run.add_sample(sample, None)

    def test_double_close_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID
            run = PredictionRun.create(run_root, _make_manifest())
            run.close()
            run.close()  # Must not raise.

    def test_close_with_no_samples_writes_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID
            run = PredictionRun.create(run_root, _make_manifest())
            run.close()

            reopened = PredictionRun.from_disk(run_root)
            assert reopened.index.empty
            assert reopened.ibtracs.empty
            assert reopened.manifest["n_samples"] == 0


# ---------------------------------------------------------------------------
# Schema validation on add_sample
# ---------------------------------------------------------------------------


class TestPredictionRunSchemaValidation:
    def test_invalid_ibtracs_schema_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / _RUN_ID
            run = PredictionRun.create(run_root, _make_manifest())
            sample = _make_sample("2016AL10", _INIT_TIME_0)

            # Frame missing the required pred / target / mask columns.
            bad_frame = pd.DataFrame(
                {
                    "sample_id": [sample.sample_id],
                    "storm_id": [sample.storm_id],
                    "season": [_SEASON],
                    "basin": [_BASIN],
                    "init_time_utc": [_INIT_TIME_0],
                    "valid_time_utc": [_INIT_TIME_0],
                    "lead_hour": [0],
                    "channel": ["usa_vmax_kt"],
                }
            )
            with pytest.raises(ValueError):
                run.add_sample(sample, bad_frame)

            run.close()
