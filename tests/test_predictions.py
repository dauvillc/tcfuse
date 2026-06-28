"""Unit tests for the predictions store: SamplePrediction + PredictionRun.

All tests use synthetic numpy Sources — no real data or model is required.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from tcfuse.data.collate import collate_window_samples
from tcfuse.data.dataset import WindowSample
from tcfuse.data.predictions import PredictionRun, SamplePrediction
from tcfuse.data.sources import Source, StormData
from tcfuse.lightning.prediction_writer import _build_sample_prediction
from tests.test_sources import make_field_source, make_scalar_source

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_prediction(
    sample_id: str = "win-0",
    *,
    season: int = 2016,
    basin: str = "AL",
    offset: float = 0.0,
) -> SamplePrediction:
    """Build a SamplePrediction with one FIELD and one SCALAR target source.

    The predicted Sources equal the targets shifted by ``offset`` so metrics are
    analytically known (bias == offset, RMSE/MAE == |offset|, R2 == 1).
    """
    # Ground-truth target Sources (fully available masks from make_* helpers).
    field_gt = make_field_source(H=4, W=4, C=2, source_name="pmw_gmi")
    scalar_gt = make_scalar_source(C=3, source_name="ibtracs")

    # Predicted Sources reuse coords/mask/channels but add a constant offset.
    field_pred = dataclasses.replace(field_gt, values=field_gt.values + offset)
    scalar_pred = dataclasses.replace(scalar_gt, values=scalar_gt.values + offset)

    return SamplePrediction(
        sample_id=sample_id,
        sid="2016AL10",
        season=season,
        basin=basin,
        subbasin="GM",
        window_ref_time_utc="2016-10-05T00:00:00",
        predicted={("pmw_gmi", 0): field_pred, ("ibtracs", 0): scalar_pred},
        target={("pmw_gmi", 0): field_gt, ("ibtracs", 0): scalar_gt},
    )


def _assert_sources_equal(a: Source, b: Source) -> None:
    """Assert two Sources carry identical arrays, channels, kind, and time."""
    assert a.kind is b.kind
    assert a.channels == b.channels
    assert a.time_utc == b.time_utc
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.coords, b.coords)
    np.testing.assert_array_equal(a.mask, b.mask)


# ---------------------------------------------------------------------------
# SamplePrediction round-trip
# ---------------------------------------------------------------------------


def test_sample_prediction_roundtrip(tmp_path: Path) -> None:
    """SamplePrediction.write then from_disk recovers all arrays and metadata."""
    sample = _make_sample_prediction(offset=1.5)
    path = tmp_path / "win-0.h5"
    sample.write(path)

    loaded = SamplePrediction.from_disk(path)

    # Sample-level metadata survives the round trip.
    assert loaded.sample_id == sample.sample_id
    assert loaded.sid == sample.sid
    assert loaded.season == sample.season
    assert loaded.basin == sample.basin
    assert loaded.window_ref_time_utc == sample.window_ref_time_utc

    # Predicted and target Sources match exactly for every key.
    assert set(loaded.predicted) == set(sample.predicted)
    assert set(loaded.target) == set(sample.target)
    for key in sample.predicted:
        _assert_sources_equal(loaded.predicted[key], sample.predicted[key])
        _assert_sources_equal(loaded.target[key], sample.target[key])


# ---------------------------------------------------------------------------
# PredictionRun round-trip
# ---------------------------------------------------------------------------


def test_prediction_run_roundtrip(tmp_path: Path) -> None:
    """PredictionRun create/append/finalize then open recovers index + samples."""
    run = PredictionRun.create(tmp_path / "run", manifest={"split": "test"})
    run.append(_make_sample_prediction("win-0"))
    run.append(_make_sample_prediction("win-1"))
    run.finalize()

    reopened = PredictionRun.open(tmp_path / "run")

    # Manifest carries through and records the sample count.
    assert reopened.manifest["split"] == "test"
    assert reopened.manifest["num_samples"] == 2

    # Index has one row per (sample_id, source_name, source_index): 2 sources x 2 samples.
    assert len(reopened.index) == 4
    assert set(reopened.sample_ids) == {"win-0", "win-1"}

    # Samples reload by id with their Sources intact.
    sample = reopened.load_sample("win-0")
    assert set(sample.predicted) == {("pmw_gmi", 0), ("ibtracs", 0)}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_prediction(tmp_path: Path) -> None:
    """Predictions equal to targets yield ~zero error and R2 == 1."""
    run = PredictionRun.create(tmp_path / "run", manifest={})
    # Two samples so R2Score has enough points to compute.
    run.append(_make_sample_prediction("win-0", offset=0.0))
    run.append(_make_sample_prediction("win-1", offset=0.0))
    run.finalize()

    metrics = PredictionRun.open(tmp_path / "run").compute_metrics()

    # Bias / RMSE / MAE all ~0 for an exact match.
    for metric_name in ("bias", "rmse", "mae"):
        values = metrics.loc[metrics["metric"] == metric_name, "value"]
        np.testing.assert_allclose(values, 0.0, atol=1e-5)
    # R2 == 1 when predictions reproduce targets exactly.
    r2 = metrics.loc[metrics["metric"] == "r2", "value"]
    np.testing.assert_allclose(r2, 1.0, atol=1e-5)


def test_compute_metrics_constant_offset(tmp_path: Path) -> None:
    """A constant offset shows up exactly as the bias for every channel."""
    offset = 2.0
    run = PredictionRun.create(tmp_path / "run", manifest={})
    run.append(_make_sample_prediction("win-0", offset=offset))
    run.append(_make_sample_prediction("win-1", offset=offset))
    run.finalize()

    metrics = PredictionRun.open(tmp_path / "run").compute_metrics()

    # Bias equals the injected offset; MAE equals its magnitude.
    bias = metrics.loc[metrics["metric"] == "bias", "value"]
    np.testing.assert_allclose(bias, offset, atol=1e-5)
    mae = metrics.loc[metrics["metric"] == "mae", "value"]
    np.testing.assert_allclose(mae, abs(offset), atol=1e-5)


def test_compute_metrics_group_by_season(tmp_path: Path) -> None:
    """Grouping by season produces one set of rows per distinct season."""
    run = PredictionRun.create(tmp_path / "run", manifest={})
    run.append(_make_sample_prediction("win-0", season=2016, offset=0.0))
    run.append(_make_sample_prediction("win-1", season=2016, offset=0.0))
    run.append(_make_sample_prediction("win-2", season=2017, offset=0.0))
    run.append(_make_sample_prediction("win-3", season=2017, offset=0.0))
    run.finalize()

    metrics = PredictionRun.open(tmp_path / "run").compute_metrics(group_by=["season"])

    # A season column is present and both seasons appear.
    assert "season" in metrics.columns
    assert set(metrics["season"]) == {2016, 2017}


# ---------------------------------------------------------------------------
# Writer mapping: batched model output -> SamplePrediction
# ---------------------------------------------------------------------------


def _make_window_sample() -> tuple[WindowSample, Source]:
    """Build a WindowSample with one target and one non-target scalar source."""
    # Two scalar sources at different times so chronological ordering is exercised.
    target_src = make_scalar_source(C=3, source_name="ibtracs")
    other_src = dataclasses.replace(
        make_scalar_source(C=3, source_name="ibtracs"),
        time_utc=target_src.time_utc + np.timedelta64(6, "h"),
    )
    storm_data = StormData(
        storm_id="2016AL10",
        basin="AL",
        subbasin="GM",
        season=2016,
        sources={
            ("ibtracs", target_src.time_utc.isoformat()): target_src,
            ("ibtracs", other_src.time_utc.isoformat()): other_src,
        },
    )
    # Earliest snapshot is index 0; mark only it as a target.
    sample = WindowSample(
        storm_data=storm_data,
        sample_id="win-0",
        window_ref_time_utc=target_src.time_utc.isoformat(),
        window_start_time_utc=target_src.time_utc.isoformat(),
        window_end_time_utc=other_src.time_utc.isoformat(),
        sid="2016AL10",
        season=2016,
        basin="AL",
        subbasin="GM",
        is_target={("ibtracs", 0): True, ("ibtracs", 1): False},
        usa_atcf_id=None,
    )
    return sample, target_src


def test_build_sample_prediction_keeps_only_targets() -> None:
    """_build_sample_prediction stores only target slots, mapping output correctly."""
    sample, target_src = _make_window_sample()
    batch = collate_window_samples([sample])

    # Use the collated batch itself as the "prediction": predicted == ground truth.
    sample_pred = _build_sample_prediction(sample, batch, sample_idx=0)

    # Only the target slot (index 0) is stored, not the non-target snapshot.
    assert set(sample_pred.predicted) == {("ibtracs", 0)}
    assert set(sample_pred.target) == {("ibtracs", 0)}
    # Predicted values are the model output column for that slot (here = ground truth).
    np.testing.assert_allclose(
        sample_pred.predicted[("ibtracs", 0)].values, target_src.values, rtol=1e-6
    )
    # The stored target Source is the ground-truth snapshot, unchanged.
    np.testing.assert_array_equal(sample_pred.target[("ibtracs", 0)].values, target_src.values)


def test_window_batch_to_moves_tensors() -> None:
    """WindowBatch.to returns a copy with tensors on the requested device."""
    sample, _ = _make_window_sample()
    batch = collate_window_samples([sample])

    moved = batch.to(torch.device("cpu"))

    # A copy is returned (not the same object) but metadata is preserved.
    assert moved is not batch
    assert moved.sample_ids == batch.sample_ids
    assert moved.sources[("ibtracs", 0)].values.device.type == "cpu"
