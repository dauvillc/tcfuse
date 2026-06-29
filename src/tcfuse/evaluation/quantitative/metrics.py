"""Core quantitative regression metrics (RMSE, MAE, R2, MAPE) over a run.

This plugin reproduces the main point-wise regression metrics from saved
predictions using numpy / scikit-learn only — deliberately **independent** of the
torchmetrics path used for online validation
(:func:`tcfuse.metrics.collection.build_source_metric_collection`).  Keeping it
standalone lets the offline evaluation suite evolve without touching training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from tcfuse.data.predictions.run import PredictionRun
from tcfuse.evaluation.base import Evaluation
from tcfuse.evaluation.flatten import flatten_valid

# Sample-level fields metrics may be grouped by — same set as
# PredictionRun.compute_metrics, so the two evaluation paths agree on grouping.
_GROUPABLE_FIELDS: frozenset[str] = frozenset({"sid", "season", "basin", "subbasin"})

# Output file written into this plugin's results subfolder.
_METRICS_FILENAME = "metrics.csv"


class QuantitativeMetricsEvaluation(Evaluation):
    """Per-source, per-channel RMSE / MAE / R2 / MAPE over a prediction run.

    Streams every window once, accumulating the valid (prediction, target) point
    pairs per ``(group_key, source_name)``, then computes each metric per channel
    with scikit-learn and writes a tidy ``metrics.csv``.

    Args:
        group_by: Optional list of sample-level fields to group metrics by
            (``"sid"``, ``"season"``, ``"basin"``, ``"subbasin"``).  ``None``
            (default) produces a single global result per source/channel.
    """

    name = "quantitative_metrics"

    def __init__(self, group_by: list[str] | None = None) -> None:
        # Validate eagerly so a bad config fails before the (slow) streaming pass.
        self.group_fields = self._validate_group_by(group_by)

    def run(self, run: PredictionRun, output_dir: Path) -> None:
        """Compute the metrics for ``run`` and write ``metrics.csv``."""
        # Accumulate valid point pairs per (group_key, source_name); also record
        # each source's channel names once for labelling the output rows.
        preds_by_key: dict[tuple[Any, ...], dict[str, list[np.ndarray]]] = {}
        targets_by_key: dict[tuple[Any, ...], dict[str, list[np.ndarray]]] = {}
        channels_by_source: dict[str, list[str]] = {}

        # Single streaming pass over every stored window.
        for sample in run.iter_samples():
            # Resolve this sample's group key from its sample-level attributes.
            group_key = tuple(getattr(sample, f) for f in self.group_fields)
            for key, pred_source in sample.predicted.items():
                source_name, _source_index = key
                target_source = sample.target[key]
                # Flatten to (N, C) keeping only fully-available spatial positions.
                preds, targets = flatten_valid(pred_source, target_source)
                if preds.shape[0] == 0:
                    continue
                channels_by_source.setdefault(source_name, pred_source.channels)
                # Append this window's rows to the matching (group, source) bucket.
                preds_by_key.setdefault(group_key, {}).setdefault(source_name, []).append(preds)
                targets_by_key.setdefault(group_key, {}).setdefault(source_name, []).append(targets)

        # Reduce every accumulated bucket into tidy per-channel metric rows.
        rows: list[dict[str, Any]] = []
        for group_key, per_source in preds_by_key.items():
            for source_name, pred_chunks in per_source.items():
                # Concatenate all windows for this (group, source) into (N, C).
                preds = np.concatenate(pred_chunks, axis=0)
                targets = np.concatenate(targets_by_key[group_key][source_name], axis=0)
                channels = channels_by_source[source_name]
                rows.extend(self._metric_rows(preds, targets, source_name, channels, group_key))

        # Persist the tidy table next to this plugin's other outputs.
        metrics = pd.DataFrame(rows)
        metrics_path = output_dir / _METRICS_FILENAME
        metrics.to_csv(metrics_path, index=False)
        print(f"  [{self.name}] wrote {len(metrics)} metric rows to {metrics_path}")

    def _metric_rows(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        source_name: str,
        channels: list[str],
        group_key: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        """Compute the four metrics per channel and emit one tidy row each."""
        rows: list[dict[str, Any]] = []
        # Compute in float64: scikit-learn computes in the input dtype, and a
        # float32 mean / sum-of-squares over millions of large physical values
        # (e.g. brightness temperatures) loses enough precision to noticeably bias
        # RMSE and especially R2. float64 matches an exact reference computation.
        preds = preds.astype(np.float64)
        targets = targets.astype(np.float64)
        # Compute each metric independently so one failure (e.g. R2 with a single
        # sample, MAPE with near-zero targets) does not suppress the others.
        for metric_name, metric_fn in _METRIC_FNS.items():
            try:
                # multioutput="raw_values" keeps one value per channel.
                values = np.atleast_1d(metric_fn(preds, targets))
            except ValueError:
                # Not enough / degenerate data for this metric — skip it here.
                continue
            # Emit one row per channel, prefixed with the group-field columns.
            for channel, value in zip(channels, values):
                row: dict[str, Any] = dict(zip(self.group_fields, group_key))
                row["source_name"] = source_name
                row["channel"] = channel
                row["metric"] = metric_name
                row["value"] = float(value)
                rows.append(row)
        return rows

    @staticmethod
    def _validate_group_by(group_by: list[str] | None) -> list[str]:
        """Return the validated grouping fields (empty list means global)."""
        if group_by is None:
            return []
        # Guard against grouping by a non-existent / non-sample field.
        unknown = set(group_by) - _GROUPABLE_FIELDS
        if unknown:
            raise ValueError(
                f"Cannot group metrics by {sorted(unknown)}; "
                f"allowed fields are {sorted(_GROUPABLE_FIELDS)}."
            )
        return list(group_by)


def _rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-channel root mean squared error.

    Computed as sqrt(MSE) rather than via the ``squared=False`` kwarg, which was
    removed from newer scikit-learn versions.
    """
    mse = mean_squared_error(targets, preds, multioutput="raw_values")
    return np.sqrt(mse)


# Each metric: (preds, targets) -> per-channel values. sklearn's convention is
# (y_true, y_pred), so targets come first.
_METRIC_FNS: dict[str, Any] = {
    # Root mean squared error per channel.
    "rmse": _rmse,
    # Mean absolute error per channel.
    "mae": lambda preds, targets: mean_absolute_error(targets, preds, multioutput="raw_values"),
    # Coefficient of determination per channel.
    "r2": lambda preds, targets: r2_score(targets, preds, multioutput="raw_values"),
    # Mean absolute percentage error per channel.
    "mape": lambda preds, targets: mean_absolute_percentage_error(
        targets, preds, multioutput="raw_values"
    ),
}
