"""Timestamp formatting utilities for on-disk keys and identifiers."""

from __future__ import annotations

import pandas as pd

_COMPACT_TIME_FMT = "%Y%m%dT%H%M%SZ"


def to_compact_time(
    time_utc: str | pd.Timestamp | float | int,
    *,
    unit: str | None = None,
) -> str:
    """Convert a UTC timestamp to a compact, HDF5-safe group name.

    Args:
        time_utc: ISO 8601 timestamp, pandas Timestamp, or numeric epoch
            value understood by :func:`pandas.Timestamp`.
        unit: Optional unit passed through to :func:`pandas.Timestamp` (e.g. ``"s"``).

    Returns:
        Compact string without separators or timezone offset,
        e.g. ``"20160912T010942Z"``.

    Raises:
        ValueError: If the value does not parse to a finite timestamp.
    """
    if unit is not None:
        ts = pd.Timestamp(time_utc, unit=unit)
    else:
        ts = pd.Timestamp(time_utc)
    if not isinstance(ts, pd.Timestamp):
        raise ValueError(f"Invalid timestamp: {time_utc!r}")
    return ts.strftime(_COMPACT_TIME_FMT)


def lead_hours_rounded(init_time_utc: str, time_utc: str) -> int:
    """Return integer lead hours between init time and snapshot time, rounded.

    Args:
        init_time_utc: Window anchor or init time as ISO 8601.
        time_utc: Snapshot valid time as ISO 8601.

    Returns:
        Rounded lead time in hours.
    """
    init_ts = pd.Timestamp(init_time_utc)
    snap_ts = pd.Timestamp(time_utc)
    delta_hours = (snap_ts - init_ts).total_seconds() / 3600.0
    return round(delta_hours)
