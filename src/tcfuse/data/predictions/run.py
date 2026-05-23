"""PredictionRun: run-level container for forecast artefacts.

A run produces three files plus a directory of per-window HDF5 files:

- ``manifest.yaml``    — run metadata (model, ckpt, split, sources predicted).
- ``index.parquet``    — one row per ``(storm_id, init_time)`` sample, the catalog.
- ``ibtracs.parquet``  — tidy-long IBTrACS preds+targets (loadable in one shot).
- ``samples/{sample_id}.h5`` — per-window predicted + target Source objects.

The writer is streaming: tidy-long IBTrACS rows are appended to a
``pyarrow.parquet.ParquetWriter`` as samples come in, the index is held in memory,
and both are finalised at :meth:`PredictionRun.close`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tcfuse.data.predictions.ibtracs import (
    IBTRACS_LONG_COLUMNS,
    empty_long_frame,
    ibtracs_long_schema,
)
from tcfuse.data.predictions.sample import SAMPLES_SUBDIR, SamplePrediction

_INDEX_FILE = "index.parquet"
_IBTRACS_FILE = "ibtracs.parquet"
_MANIFEST_FILE = "manifest.yaml"


_INDEX_COLUMNS: list[str] = [
    "sample_id",
    "storm_id",
    "season",
    "basin",
    "atcf_id",
    "init_time_utc",
    "window_start_time_utc",
    "window_end_time_utc",
    "n_predicted_sources",
    "predicted_source_names",
    "has_ibtracs_prediction",
    "sample_path",
]
"""Canonical column order for ``index.parquet``."""


def _now_iso_utc() -> str:
    """Return the current UTC time as a naive ISO timestamp."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


@dataclasses.dataclass
class PredictionRun:
    """Run-level container coordinating manifest, index, IBTrACS table, and per-sample files.

    Use :meth:`create` to start a new run for writing, or :meth:`from_disk` to read
    back an existing run. The writer is single-pass: call :meth:`add_sample` once per
    window, then :meth:`close` to finalise the on-disk artefacts.

    Args:
        run_root: Absolute path to the run directory (``{cfg.paths.predictions}/{run_id}``).
        manifest: Run-level metadata, written to ``manifest.yaml``.
    """

    run_root: Path
    manifest: dict[str, Any]
    _index_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list, repr=False)
    _ibtracs_writer: pq.ParquetWriter | None = dataclasses.field(default=None, repr=False)
    _ibtracs_schema: pa.Schema | None = dataclasses.field(default=None, repr=False)
    _closed: bool = dataclasses.field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, run_root: Path, manifest: dict[str, Any]) -> PredictionRun:
        """Create a fresh run directory ready for streaming writes.

        Initialises ``run_root/`` and ``run_root/samples/``, persists the manifest with
        a ``created_at_utc`` timestamp injected if absent, and returns an open writer
        instance. Existing files in the directory are not removed; callers are
        expected to choose unique ``run_id`` values.

        Args:
            run_root: Absolute path of the run directory.
            manifest: Run-level metadata dict.

        Returns:
            An open :class:`PredictionRun` writer instance.
        """
        run_root = Path(run_root)
        (run_root / SAMPLES_SUBDIR).mkdir(parents=True, exist_ok=True)

        # Inject created_at_utc if the caller did not provide one explicitly.
        manifest = dict(manifest)
        manifest.setdefault("created_at_utc", _now_iso_utc())

        instance = cls(run_root=run_root, manifest=manifest)
        instance._write_manifest()
        return instance

    # ------------------------------------------------------------------
    # Streaming writes
    # ------------------------------------------------------------------

    def add_sample(
        self,
        sample: SamplePrediction,
        ibtracs_long_rows: pd.DataFrame | None = None,
        *,
        window_start_time_utc: str | None = None,
        window_end_time_utc: str | None = None,
    ) -> None:
        """Persist one window: HDF5 file + index row + IBTrACS rows.

        Args:
            sample: Per-window container of predicted and target sources. Written to
                ``samples/{sample_id}.h5``.
            ibtracs_long_rows: Tidy-long IBTrACS rows for this sample; pass ``None`` or
                an empty frame when the model did not emit IBTrACS predictions for
                this window. The schema must match :func:`ibtracs_long_schema`.
            window_start_time_utc: Optional override for the window-start column in
                the catalog. Defaults to ``sample.init_time_utc``.
            window_end_time_utc: Optional override for the window-end column in
                the catalog. When omitted, the column is left empty for this row.
        """
        if self._closed:
            raise RuntimeError("PredictionRun is closed; cannot add more samples.")

        sample_path = sample.write(self.run_root)
        relative_sample_path = sample_path.relative_to(self.run_root).as_posix()

        # Detect whether the caller actually emitted IBTrACS rows for this sample.
        has_ibtracs = ibtracs_long_rows is not None and not ibtracs_long_rows.empty
        if has_ibtracs:
            assert ibtracs_long_rows is not None  # narrow for type checker
            self._append_ibtracs(ibtracs_long_rows)

        predicted_source_names = sample.predicted_source_names

        # Build the catalog row matching _INDEX_COLUMNS.
        index_row = {
            "sample_id": sample.sample_id,
            "storm_id": sample.storm_id,
            "season": int(sample.season),
            "basin": sample.basin,
            "atcf_id": sample.atcf_id,
            "init_time_utc": sample.init_time_utc,
            "window_start_time_utc": window_start_time_utc or sample.init_time_utc,
            "window_end_time_utc": window_end_time_utc,
            "n_predicted_sources": len(predicted_source_names),
            "predicted_source_names": predicted_source_names,
            "has_ibtracs_prediction": bool(has_ibtracs),
            "sample_path": relative_sample_path,
        }
        self._index_rows.append(index_row)

    def _append_ibtracs(self, frame: pd.DataFrame) -> None:
        """Append a tidy-long IBTrACS block to the streaming Parquet writer."""
        # Validate the schema before reindexing — reindex would silently add NaN
        # columns for missing keys, which would mask caller bugs.
        missing = [column for column in IBTRACS_LONG_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"ibtracs_long_rows is missing columns: {missing}")
        ordered = frame.reindex(columns=IBTRACS_LONG_COLUMNS)

        # Lazily open the writer; the schema is fixed for the lifetime of the run.
        if self._ibtracs_writer is None:
            self._ibtracs_schema = ibtracs_long_schema()
            self._ibtracs_writer = pq.ParquetWriter(
                self.run_root / _IBTRACS_FILE,
                self._ibtracs_schema,
            )

        table = pa.Table.from_pandas(
            ordered,
            schema=self._ibtracs_schema,
            preserve_index=False,
        )
        self._ibtracs_writer.write_table(table)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalise ``index.parquet``, ``ibtracs.parquet``, and ``manifest.yaml``.

        Idempotent — calling :meth:`close` more than once is a no-op.
        """
        if self._closed:
            return

        # Finalise the streaming IBTrACS writer; if no rows were ever appended, write
        # an empty parquet file so downstream code can rely on the file existing.
        if self._ibtracs_writer is not None:
            self._ibtracs_writer.close()
            self._ibtracs_writer = None
        else:
            empty_table = pa.Table.from_pandas(
                empty_long_frame(),
                schema=ibtracs_long_schema(),
                preserve_index=False,
            )
            pq.write_table(empty_table, self.run_root / _IBTRACS_FILE)

        # Materialise the catalog from the in-memory list.
        if self._index_rows:
            index_frame = pd.DataFrame(self._index_rows, columns=_INDEX_COLUMNS)
        else:
            index_frame = pd.DataFrame({column: [] for column in _INDEX_COLUMNS})
        index_frame.to_parquet(self.run_root / _INDEX_FILE, index=False)

        # Refresh the manifest with whatever derived stats are now known.
        self.manifest.setdefault("ibtracs_channels", [])
        self.manifest["n_samples"] = len(index_frame)
        self.manifest["predicted_sources"] = self._collect_predicted_source_names()
        self._write_manifest()

        self._closed = True

    def _collect_predicted_source_names(self) -> list[str]:
        """Return the union of predicted source names across all written samples."""
        seen: set[str] = set()
        for row in self._index_rows:
            for name in row["predicted_source_names"]:
                seen.add(name)
        return sorted(seen)

    def _write_manifest(self) -> None:
        """Serialise ``self.manifest`` to ``run_root/manifest.yaml``."""
        with (self.run_root / _MANIFEST_FILE).open("w") as f:
            yaml.safe_dump(self.manifest, f, sort_keys=False)

    def __enter__(self) -> PredictionRun:
        """Enter the context manager; returns ``self`` for ``with`` usage."""
        return self

    def __exit__(self, *_excinfo: object) -> None:
        """Exit the context manager, calling :meth:`close` to finalise the run."""
        self.close()

    # ------------------------------------------------------------------
    # Reader
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(cls, run_root: Path) -> PredictionRun:
        """Open an existing run for reading.

        Loads the manifest eagerly. ``index`` and ``ibtracs`` are exposed as cached
        properties so the parquet files are only read on first access.

        Args:
            run_root: Absolute path of the run directory.

        Returns:
            A read-only :class:`PredictionRun` instance with ``_closed=True``; calls
            to :meth:`add_sample` will raise.
        """
        run_root = Path(run_root)
        manifest_path = run_root / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        with manifest_path.open() as f:
            manifest = yaml.safe_load(f) or {}

        return cls(run_root=run_root, manifest=manifest, _closed=True)

    @property
    def index(self) -> pd.DataFrame:
        """Catalog DataFrame loaded from ``index.parquet``."""
        path = self.run_root / _INDEX_FILE
        if not path.exists():
            return pd.DataFrame({column: [] for column in _INDEX_COLUMNS})
        return pd.read_parquet(path)

    @property
    def ibtracs(self) -> pd.DataFrame:
        """Tidy-long IBTrACS DataFrame loaded from ``ibtracs.parquet``."""
        path = self.run_root / _IBTRACS_FILE
        if not path.exists():
            return empty_long_frame()
        return pd.read_parquet(path)

    def load_sample(self, sample_id: str) -> SamplePrediction:
        """Load a single per-window file by ``sample_id``."""
        return SamplePrediction.from_disk(self.run_root, sample_id)

    def iter_samples(self) -> Iterator[SamplePrediction]:
        """Iterate over per-window files in the order recorded in ``index.parquet``."""
        for sample_id in self.index["sample_id"].astype(str).tolist():
            yield SamplePrediction.from_disk(self.run_root, sample_id)
