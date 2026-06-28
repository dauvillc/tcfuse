"""PredictionRun: a serializable collection of per-window predictions + evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml

from tcfuse.data.predictions.sample import SamplePrediction
from tcfuse.data.sources.source import Source
from tcfuse.metrics.collection import build_source_metric_collection

# On-disk names inside a run directory.
_MANIFEST_FILENAME = "manifest.yaml"
_INDEX_FILENAME = "index.parquet"
_SAMPLES_DIR = "samples"

# Sample-level fields that may be used to group metrics.
_GROUPABLE_FIELDS: frozenset[str] = frozenset({"sid", "season", "basin", "subbasin"})


@dataclass
class PredictionRun:
    """A directory of per-window predictions plus a queryable index and manifest.

    Layout on disk::

        {run_dir}/
        ├── manifest.yaml          # run-level metadata (checkpoint, split, units, …)
        ├── index.parquet          # one row per (sample_id, source_name, source_index)
        └── samples/{sample_id}.h5 # one SamplePrediction per window

    Use :meth:`create` + :meth:`append` + :meth:`finalize` to write a run, and
    :meth:`open` to read one back.  :meth:`compute_metrics` evaluates the whole
    run from the saved files alone.

    Args:
        run_dir: Root directory of this prediction run.
        manifest: Run-level metadata dict (free-form; see :meth:`create`).
        index: Long-format catalog, one row per stored ``(sample_id,
            source_name, source_index)``.  Empty while a run is being written.
    """

    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    index: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Buffered index rows accumulated during writing; flushed by finalize().
    _pending_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, run_dir: Path, manifest: dict[str, Any]) -> PredictionRun:
        """Start a fresh run for writing under ``run_dir``.

        Args:
            run_dir: Destination directory (created if absent).
            manifest: Run-level metadata to persist, e.g. ``checkpoint_path``,
                ``windows_setup_name``, ``split``, ``units``, ``model_name``.

        Returns:
            An empty :class:`PredictionRun` ready for :meth:`append`.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / _SAMPLES_DIR).mkdir(exist_ok=True)
        return cls(run_dir=run_dir, manifest=dict(manifest))

    def append(self, sample: SamplePrediction) -> None:
        """Write one :class:`SamplePrediction` and record its index rows.

        Args:
            sample: The window prediction to persist.  Its ``sample_id`` names
                the on-disk file ``samples/{sample_id}.h5``.
        """
        # Persist the per-window HDF5 file.
        sample.write(self._sample_path(sample.sample_id))
        # Record one index row per predicted (source_name, source_index).
        for (source_name, source_index), source in sample.predicted.items():
            self._pending_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sid": sample.sid,
                    "season": sample.season,
                    "basin": sample.basin,
                    "subbasin": sample.subbasin,
                    "window_ref_time_utc": sample.window_ref_time_utc,
                    "source_name": source_name,
                    "source_index": source_index,
                    "kind": source.kind.name,
                    "time_utc": source.time_utc.isoformat(),
                    "n_channels": len(source.channels),
                }
            )

    def finalize(self) -> None:
        """Flush the index and manifest to disk, completing the run."""
        # Build the catalog DataFrame from buffered rows and persist it.
        self.index = pd.DataFrame(self._pending_rows)
        self.index.to_parquet(self.run_dir / _INDEX_FILENAME)
        # Record the sample count for convenience, then write the manifest.
        n_samples = int(cast(int, self.index["sample_id"].nunique())) if len(self.index) else 0
        self.manifest["num_samples"] = n_samples
        with open(self.run_dir / _MANIFEST_FILENAME, "w") as f:
            yaml.safe_dump(self.manifest, f, sort_keys=False)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, run_dir: Path) -> PredictionRun:
        """Open an existing run for reading.

        Args:
            run_dir: Root directory previously written by :meth:`finalize`.

        Returns:
            A :class:`PredictionRun` with its manifest and index loaded.
        """
        with open(run_dir / _MANIFEST_FILENAME) as f:
            manifest = yaml.safe_load(f)
        index = pd.read_parquet(run_dir / _INDEX_FILENAME)
        return cls(run_dir=run_dir, manifest=manifest, index=index)

    @property
    def sample_ids(self) -> list[str]:
        """Distinct window identifiers present in this run, in index order."""
        return self.index["sample_id"].drop_duplicates().tolist()

    def load_sample(self, sample_id: str) -> SamplePrediction:
        """Load a single :class:`SamplePrediction` by its window identifier."""
        return SamplePrediction.from_disk(self._sample_path(sample_id))

    def iter_samples(self) -> Iterator[SamplePrediction]:
        """Iterate over every stored :class:`SamplePrediction` lazily."""
        for sample_id in self.sample_ids:
            yield self.load_sample(sample_id)

    def _sample_path(self, sample_id: str) -> Path:
        """Return the HDF5 path for one window's predictions."""
        return self.run_dir / _SAMPLES_DIR / f"{sample_id}.h5"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def compute_metrics(self, group_by: list[str] | None = None) -> pd.DataFrame:
        """Compute per-source, per-channel metrics over the whole run.

        Reuses the same metric set as online validation
        (:func:`~tcfuse.metrics.collection.build_source_metric_collection`):
        bias, RMSE, MAE, and R2.  Values are already in physical units, and the
        validity mask is taken from each target Source's ``mask`` reduced to
        spatial positions where every channel is available — mirroring
        ``BaseLightningModule._update_val_metrics``.

        Args:
            group_by: Optional list of sample-level fields to group metrics by.
                Allowed values: ``"sid"``, ``"season"``, ``"basin"``,
                ``"subbasin"``.  ``None`` computes a single global result.

        Returns:
            Tidy DataFrame with columns ``source_name, channel, metric, value``
            (plus one column per ``group_by`` field).
        """
        group_fields = self._validate_group_by(group_by)

        # Per (group_key, source_name): a metric collection accumulating updates,
        # and the source's channel names for labelling the per-channel results.
        collections: dict[tuple[Any, ...], dict[str, torchmetrics.MetricCollection]] = {}
        channels_by_source: dict[str, list[str]] = {}

        # Single streaming pass over every stored window.
        for sample in self.iter_samples():
            # Resolve this sample's group key from its sample-level attributes.
            group_key = tuple(getattr(sample, f) for f in group_fields)
            for key, pred_source in sample.predicted.items():
                source_name, _source_index = key
                target_source = sample.target[key]
                # Flatten to (N, C) keeping only fully-available spatial positions.
                preds, targets = _flatten_valid(pred_source, target_source)
                if preds.shape[0] == 0:
                    continue
                channels_by_source.setdefault(source_name, pred_source.channels)
                # Lazily build a metric collection for this (group, source) pair.
                bucket = collections.setdefault(group_key, {})
                if source_name not in bucket:
                    bucket[source_name] = build_source_metric_collection(len(pred_source.channels))
                bucket[source_name].update(preds, targets)

        # Reduce all accumulated collections into tidy rows.
        rows: list[dict[str, Any]] = []
        for group_key, per_source in collections.items():
            for source_name, collection in per_source.items():
                channels = channels_by_source[source_name]
                rows.extend(
                    self._collection_to_rows(
                        collection, source_name, channels, group_fields, group_key
                    )
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _validate_group_by(group_by: list[str] | None) -> list[str]:
        """Return the validated grouping fields (empty list means global)."""
        if group_by is None:
            return []
        # Guard against grouping by a non-existent / non-sample field.
        unknown = set(group_by) - _GROUPABLE_FIELDS
        if unknown:
            raise ValueError(
                f"Cannot group predictions by {sorted(unknown)}; "
                f"allowed fields are {sorted(_GROUPABLE_FIELDS)}."
            )
        return list(group_by)

    @staticmethod
    def _collection_to_rows(
        collection: torchmetrics.MetricCollection,
        source_name: str,
        channels: list[str],
        group_fields: list[str],
        group_key: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        """Compute one metric collection into tidy per-channel rows."""
        rows: list[dict[str, Any]] = []
        # Compute each metric independently so a failing metric (e.g. R2 with a
        # single sample) does not suppress the others.
        for metric_name, metric in collection.items():
            try:
                values = torch.atleast_1d(metric.compute())
            except ValueError:
                # Not enough samples for this metric — skip it for this group/source.
                continue
            # Emit one row per channel.
            for channel, value in zip(channels, values):
                row: dict[str, Any] = dict(zip(group_fields, group_key))
                row["source_name"] = source_name
                row["channel"] = channel
                row["metric"] = metric_name
                row["value"] = float(value)
                rows.append(row)
        return rows


def _flatten_valid(
    pred_source: Source,
    target_source: Source,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a (pred, target) Source pair to ``(N, C)`` over valid positions.

    A spatial position is valid when every channel of the target is available
    (``target.mask`` True), matching the reduction used during training. SCALAR,
    PROFILE, and FIELD sources are all collapsed to a 2-D ``(N, C)`` layout.

    Args:
        pred_source: Predicted Source (physical units).
        target_source: Ground-truth Source (physical units), same shape.

    Returns:
        ``(preds, targets)`` float32 tensors of shape ``(N, C)`` over valid rows.
    """
    C = pred_source.values.shape[-1]
    # Collapse all leading (spatial) axes into one row axis: (..., C) -> (N, C).
    pred_2d = pred_source.values.reshape(-1, C)
    target_2d = target_source.values.reshape(-1, C)
    mask_2d = target_source.mask.reshape(-1, C)
    # Keep rows where every channel is available, so update() sees a consistent N.
    valid = mask_2d.all(axis=-1)
    preds = torch.from_numpy(np.asarray(pred_2d[valid], dtype=np.float32))
    targets = torch.from_numpy(np.asarray(target_2d[valid], dtype=np.float32))
    return preds, targets
