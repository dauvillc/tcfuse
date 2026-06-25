"""BiasMetric: mean signed error (pred - target) per output channel."""

from __future__ import annotations

import torch
import torchmetrics


class BiasMetric(torchmetrics.Metric):
    """Mean signed error per output channel: mean(pred - target, dim=0).

    Measures systematic over/under-prediction. Positive bias means the model
    over-predicts on average; negative bias means it under-predicts.

    Mirrors the ``num_outputs`` interface of built-in TorchMetrics regression
    metrics so it can be used inside a :class:`torchmetrics.MetricCollection`
    alongside :class:`~torchmetrics.MeanSquaredError`,
    :class:`~torchmetrics.MeanAbsoluteError`, and :class:`~torchmetrics.R2Score`.

    Args:
        num_outputs: Number of output channels. Inputs to :meth:`update` must
            be ``(N, num_outputs)`` tensors.
    """

    # Declared as class-level annotations so torchmetrics registers them as states.
    sum_error: torch.Tensor  # (num_outputs,) — accumulated sum of (pred - target)
    count: torch.Tensor  # scalar — number of samples seen

    def __init__(self, num_outputs: int = 1) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        # sum_error is (C,) so the per-channel sum accumulates correctly.
        self.add_state("sum_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        # count is a scalar — same number of samples across all channels.
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate signed residuals for one batch.

        Args:
            preds: Predicted values, shape ``(N, num_outputs)``.
            targets: Ground-truth values, shape ``(N, num_outputs)``.
        """
        # Sum residuals along the sample axis; result is (num_outputs,).
        self.sum_error += (preds - targets).sum(dim=0)
        # All channels share the same sample count.
        self.count += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """Return mean signed error per channel, shape ``(num_outputs,)``.

        Returns:
            Tensor of shape ``(num_outputs,)`` containing the bias for each channel.
        """
        return self.sum_error / self.count
