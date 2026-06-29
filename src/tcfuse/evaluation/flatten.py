"""Opt-in helper: flatten a (pred, target) Source pair to point-wise ``(N, C)``.

This is the numpy counterpart of ``PredictionRun._flatten_valid`` (which returns
torch tensors), kept here so evaluation plugins stay independent of torch /
torchmetrics.  It is **not** invoked by the :class:`~tcfuse.evaluation.base.Evaluation`
base class — only plugins that genuinely want point-wise data (e.g. the
quantitative regression metrics) call it.  Plugins that need spatial structure
(power spectra, image diagnostics) ignore this helper and read
``source.values`` / ``source.mask`` directly.
"""

from __future__ import annotations

import numpy as np

from tcfuse.data.sources.source import Source


def flatten_valid(
    pred_source: Source,
    target_source: Source,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a (pred, target) Source pair to ``(N, C)`` over valid positions.

    A spatial position is valid only where **every** target channel is available
    (``target.mask`` True), mirroring the reduction used during training and in
    ``PredictionRun._flatten_valid``.  SCALAR, PROFILE, and FIELD sources are all
    collapsed to a 2-D ``(N, C)`` layout.

    Args:
        pred_source: Predicted Source (physical units).
        target_source: Ground-truth Source (physical units), same shape.

    Returns:
        ``(preds, targets)`` float32 arrays of shape ``(N, C)`` over valid rows.
    """
    # Number of channels lives on the last axis for every SourceKind.
    n_channels = pred_source.values.shape[-1]
    # Collapse all leading (spatial) axes into one row axis: (..., C) -> (N, C).
    pred_2d = pred_source.values.reshape(-1, n_channels)
    target_2d = target_source.values.reshape(-1, n_channels)
    mask_2d = target_source.mask.reshape(-1, n_channels)
    # Keep only rows where every channel is available, so each metric sees a
    # consistent N across channels.
    valid = mask_2d.all(axis=-1)
    preds = np.asarray(pred_2d[valid], dtype=np.float32)
    targets = np.asarray(target_2d[valid], dtype=np.float32)
    return preds, targets
