"""Lightning prediction writer that persists model output as a PredictionRun."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, override

import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter

from tcfuse.data.collate import WindowBatch
from tcfuse.data.dataset import TCWindowDataset, WindowSample
from tcfuse.data.predictions.run import PredictionRun
from tcfuse.data.predictions.sample import SamplePrediction
from tcfuse.data.sources.source import Source


def _build_sample_prediction(
    sample: WindowSample,
    pred_batch: WindowBatch,
    sample_idx: int,
) -> SamplePrediction:
    """Assemble one window's :class:`SamplePrediction` from the batched model output.

    For every target slot of this sample, the predicted values are taken from the
    model's output column ``sample_idx`` and wrapped into a Source that reuses the
    ground-truth coords/mask/channels/time. The ground-truth Source is stored
    as-is so metrics are self-contained.

    Args:
        sample: The source :class:`WindowSample` (ground truth, numpy Sources).
        pred_batch: De-normalized model output for the whole batch.
        sample_idx: Column of this sample within the batch.

    Returns:
        A :class:`SamplePrediction` covering only this sample's target sources.
    """
    # Map (source_name, source_index) -> ground-truth Source using the same
    # chronological ordering as collate_window_samples (earliest snapshot = 0).
    sorted_items = sorted(sample.storm_data.sources.items(), key=lambda kv: kv[0][1])
    name_count: dict[str, int] = {}
    key_to_gt: dict[tuple[str, int], Source] = {}
    for (source_name, _time_utc), source in sorted_items:
        idx = name_count.get(source_name, 0)
        name_count[source_name] = idx + 1
        key_to_gt[(source_name, idx)] = source

    # Keep only the slots this sample marked as targets (the reconstructed ones).
    predicted: dict[tuple[str, int], Source] = {}
    target: dict[tuple[str, int], Source] = {}
    for key, is_target in sample.is_target.items():
        if not is_target:
            continue
        gt_source = key_to_gt[key]
        # Slice this sample's prediction column and detach to a numpy array.
        pred_values = (
            pred_batch.sources[key].values[sample_idx].detach().cpu().numpy().astype(np.float32)
        )
        # Predicted Source = ground truth with model values swapped in (coords,
        # mask, channels, kind, and time_utc are all preserved for evaluation).
        predicted[key] = dataclasses.replace(gt_source, values=pred_values)
        target[key] = gt_source

    return SamplePrediction(
        sample_id=sample.sample_id,
        sid=sample.sid,
        season=sample.season,
        basin=sample.basin,
        subbasin=sample.subbasin,
        window_ref_time_utc=sample.window_ref_time_utc,
        predicted=predicted,
        target=target,
    )


class PredictionWriter(BasePredictionWriter):
    """Persist each predicted batch into a :class:`PredictionRun` as it is produced.

    Hooks into ``trainer.predict`` so Lightning owns the loop and device placement.
    For every batch, the writer recovers the original
    :class:`~tcfuse.data.dataset.WindowSample` objects from the dataset (via the
    per-sample indices Lightning provides) so it can pair the model's output with
    the ground-truth Sources and snapshot times that the collated
    :class:`~tcfuse.data.collate.WindowBatch` no longer carries.

    Single-GPU assumption: inference runs on one device (``devices=1`` in
    ``scripts/inference/infer.py``).  The per-process index buffer in
    :class:`PredictionRun` and unconditional ``finalize`` are not DDP-safe — under
    multiple ranks each would write only its own shard's index and race on the
    output files.  Supporting DDP would require gathering index rows to rank 0,
    rank-guarding ``finalize``/metrics, and deduplicating padded tail samples.

    Args:
        run: Open :class:`PredictionRun` to append samples to.  The caller is
            responsible for calling :meth:`PredictionRun.finalize` afterwards.
        dataset: The dataset being predicted over; indexed to recover ground truth.
    """

    def __init__(self, run: PredictionRun, dataset: TCWindowDataset) -> None:
        # write_interval="batch": persist as each batch finishes (streaming).
        super().__init__(write_interval="batch")
        self._run = run
        self._dataset = dataset

    @override
    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: WindowBatch,
        batch_indices: Sequence[int] | None,
        batch: WindowBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write one batch of predictions, one SamplePrediction per window.

        Args:
            trainer: The Lightning trainer driving prediction (unused).
            pl_module: The module being predicted with (unused).
            prediction: De-normalized model output for this batch
                (the return value of ``predict_step``).
            batch_indices: Dataset indices of this batch's samples, in order.
            batch: The input :class:`WindowBatch` (unused; ground truth is read
                from the dataset for its numpy Sources and snapshot times).
            batch_idx: Index of this batch within the dataloader.
            dataloader_idx: Index of the dataloader (unused; single loader here).
        """
        # Recover the dataset positions of this batch's samples. With a single-GPU
        # sequential sampler these are the global dataset indices and are always
        # populated when a BasePredictionWriter is attached. Guard loudly rather
        # than guessing positions (a rank-local guess would silently mis-pair
        # predictions with ground truth — see the class docstring on DDP).
        indices = _flatten_indices(batch_indices)
        if not indices:
            raise RuntimeError(
                "PredictionWriter received no per-batch dataset indices. Inference "
                "assumes a single GPU with a sequential sampler; multi-rank DDP "
                "prediction is not supported."
            )

        # Reload each window's ground truth and pair it with the model output.
        for sample_idx, dataset_idx in enumerate(indices):
            sample = self._dataset[dataset_idx]
            sample_pred = _build_sample_prediction(sample, prediction, sample_idx)
            # Skip windows with no target slots (nothing to evaluate).
            if sample_pred.predicted:
                self._run.append(sample_pred)


def _flatten_indices(batch_indices: Sequence[Any] | None) -> list[int]:
    """Flatten Lightning's (possibly nested) per-batch indices into a flat int list."""
    if batch_indices is None:
        return []
    flat: list[int] = []
    for item in batch_indices:
        # Some samplers nest indices one level deep; flatten those.
        if isinstance(item, (list, tuple)):
            flat.extend(int(x) for x in item)
        else:
            flat.append(int(item))
    return flat
