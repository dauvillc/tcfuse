"""Shared per-source regression metric collection used in training and evaluation."""

from __future__ import annotations

import torchmetrics

from tcfuse.metrics.bias import BiasMetric


def build_source_metric_collection(num_channels: int) -> torchmetrics.MetricCollection:
    """Build the per-channel regression metrics for a single source.

    The same collection is used both for online validation logging
    (:class:`~tcfuse.lightning.base_module.BaseLightningModule`) and offline
    evaluation of saved predictions
    (:class:`~tcfuse.data.predictions.run.PredictionRun`), so the metric set
    stays identical across the two code paths.

    Every metric is configured to return a ``(num_channels,)`` tensor so its
    entries map one-to-one onto a source's channel names.

    Args:
        num_channels: Number of channels (last axis of a source's values).

    Returns:
        A :class:`~torchmetrics.MetricCollection` with ``bias``, ``rmse``,
        ``mae``, and ``r2`` metrics, each emitting per-channel values.
    """
    return torchmetrics.MetricCollection(
        {
            # Mean signed error (pred - target) per channel.
            "bias": BiasMetric(num_outputs=num_channels),
            # Root mean squared error per channel.
            "rmse": torchmetrics.MeanSquaredError(squared=False, num_outputs=num_channels),
            # Mean absolute error per channel.
            "mae": torchmetrics.MeanAbsoluteError(num_outputs=num_channels),
            # Coefficient of determination; raw_values keeps one R2 per channel.
            "r2": torchmetrics.R2Score(multioutput="raw_values"),
        }
    )
