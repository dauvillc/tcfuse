"""Utilities for loading and interpreting best-track window sample indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

SplitName = Literal["train", "val", "test"]

_SPLIT_FILENAMES: dict[SplitName, str] = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}

_REQUIRED_INDEX_COLUMNS = (
    "sample_id",
    "sid",
    "basin",
    "subbasin",
    "season",
    "init_time_utc",
    "window_start_time_utc",
    "window_end_time_utc",
)


def parse_utc_timestamp(value: Any) -> pd.Timestamp:
    """Parse a timestamp value as UTC for exact window matching.

    Args:
        value: ISO timestamp string, pandas Timestamp, or compatible scalar.

    Returns:
        Timezone-naive UTC :class:`~pandas.Timestamp`.
    """
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize("UTC").tz_localize(None))
    return cast(pd.Timestamp, timestamp.tz_convert("UTC").tz_localize(None))


def snapshot_in_window(
    time_utc: Any,
    window_start_utc: Any,
    window_end_utc: Any,
) -> bool:
    """Return True when a snapshot falls inside a closed assimilation window."""
    snapshot_ts = parse_utc_timestamp(time_utc)
    start_ts = parse_utc_timestamp(window_start_utc)
    end_ts = parse_utc_timestamp(window_end_utc)
    return start_ts <= snapshot_ts <= end_ts


def load_split_index(assembled_root: Path, split: SplitName) -> pd.DataFrame:
    """Load a train/val/test window index parquet produced by ``build_splits.py``.

    Args:
        assembled_root: Root directory for assembled data
            (``cfg.paths.preprocessed_data``).
        split: Which split parquet to load.

    Returns:
        Window-index DataFrame with one row per model sample.

    Raises:
        FileNotFoundError: When the split parquet is missing.
        ValueError: When required columns are absent.
    """
    index_path = assembled_root / _SPLIT_FILENAMES[split]
    if not index_path.exists():
        raise FileNotFoundError(
            f"Window index not found at {index_path}. Run scripts/preprocess/build_splits.py first."
        )

    index = pd.read_parquet(index_path)
    missing = [column for column in _REQUIRED_INDEX_COLUMNS if column not in index.columns]
    if missing:
        raise ValueError(f"Window index at {index_path} is missing required columns: {missing}")
    return index.reset_index(drop=True)
