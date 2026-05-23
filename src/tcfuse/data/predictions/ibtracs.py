"""Tidy-long schema and helpers for IBTrACS prediction/target tables.

The schema stores one row per ``(sample_id, valid_time, channel)``:

- Keys: ``sample_id``, ``storm_id``, ``init_time_utc``, ``valid_time_utc``,
  ``lead_hour``, ``channel``.
- Carry: ``season``, ``basin`` (denormalized for cheap group-bys).
- Values: ``pred`` (float64), ``target`` (float64), ``mask`` (bool).

The ``mask`` column is ``True`` when both ``pred`` and ``target`` are finite,
matching the per-value availability convention used everywhere else in the
codebase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

IBTRACS_LONG_COLUMNS: list[str] = [
    "sample_id",
    "storm_id",
    "season",
    "basin",
    "init_time_utc",
    "valid_time_utc",
    "lead_hour",
    "channel",
    "pred",
    "target",
    "mask",
]
"""Canonical column order for the tidy-long IBTrACS prediction table."""


_IBTRACS_LONG_FIELDS: list[tuple[str, pa.DataType]] = [
    ("sample_id", pa.string()),
    ("storm_id", pa.string()),
    ("season", pa.int32()),
    ("basin", pa.dictionary(pa.int32(), pa.string())),
    ("init_time_utc", pa.string()),
    ("valid_time_utc", pa.string()),
    ("lead_hour", pa.int32()),
    ("channel", pa.dictionary(pa.int32(), pa.string())),
    ("pred", pa.float64()),
    ("target", pa.float64()),
    ("mask", pa.bool_()),
]


def ibtracs_long_schema() -> pa.Schema:
    """Return the canonical pyarrow schema for the tidy-long IBTrACS table."""
    return pa.schema(_IBTRACS_LONG_FIELDS)


def _isnan(value: object) -> bool:
    """Return True when value is missing or non-finite."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_float_or_nan(value: object) -> float:
    """Return ``value`` as a float, mapping missing/non-numeric inputs to NaN."""
    if _isnan(value):
        return float("nan")
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")


def build_long_rows(
    sample_id: str,
    storm_id: str,
    season: int,
    basin: str,
    init_time_utc: str,
    leads: Sequence[Mapping[str, Any]],
    channels: Sequence[str],
) -> pd.DataFrame:
    """Build a tidy-long DataFrame block for a single sample.

    Args:
        sample_id: Sample identifier ``f"{storm_id}_{anchor_time:%Y%m%dT%H%M%SZ}"``.
        storm_id: Storm identifier, e.g. ``"2016AL10"``.
        season: TC season year (carried for cheap group-bys).
        basin: Ocean basin code (carried for cheap group-bys).
        init_time_utc: Window anchor time in repository ISO format.
        leads: One mapping per lead time, each with keys
            ``"lead_hour"`` (int), ``"valid_time_utc"`` (str), ``"pred"`` (mapping
            channel -> float | NaN), and ``"target"`` (mapping channel -> float | NaN).
            ``pred`` or ``target`` may also be ``None`` to signal a fully missing block;
            in that case all entries are stored as NaN with ``mask=False``.
        channels: Ordered list of IBTrACS channel names to materialise. Channels
            absent from a given lead are stored as NaN with ``mask=False``.

    Returns:
        DataFrame with columns matching :data:`IBTRACS_LONG_COLUMNS`, one row per
        ``(lead_hour, channel)`` for this sample.
    """
    rows: list[dict[str, Any]] = []

    for lead in leads:
        lead_hour = int(lead["lead_hour"])
        valid_time_utc = str(lead["valid_time_utc"])
        pred_map = lead.get("pred") or {}
        target_map = lead.get("target") or {}

        # Iterate channels in caller-provided order so the long table is reproducible.
        for channel in channels:
            pred_value = pred_map.get(channel) if pred_map else np.nan
            target_value = target_map.get(channel) if target_map else np.nan

            pred_float = _coerce_float_or_nan(pred_value)
            target_float = _coerce_float_or_nan(target_value)
            mask = np.isfinite(pred_float) and np.isfinite(target_float)

            rows.append(
                {
                    "sample_id": sample_id,
                    "storm_id": storm_id,
                    "season": int(season),
                    "basin": basin,
                    "init_time_utc": init_time_utc,
                    "valid_time_utc": valid_time_utc,
                    "lead_hour": lead_hour,
                    "channel": channel,
                    "pred": pred_float,
                    "target": target_float,
                    "mask": bool(mask),
                }
            )

    if not rows:
        return empty_long_frame()

    frame = pd.DataFrame(rows, columns=IBTRACS_LONG_COLUMNS)
    return _coerce_long_dtypes(frame)


def empty_long_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical long-table columns and dtypes."""
    frame = pd.DataFrame({column: [] for column in IBTRACS_LONG_COLUMNS})
    return _coerce_long_dtypes(frame)


def _coerce_long_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the dtypes declared in :func:`ibtracs_long_schema`."""
    coerced = frame.copy()
    coerced["sample_id"] = coerced["sample_id"].astype("string")
    coerced["storm_id"] = coerced["storm_id"].astype("string")
    coerced["season"] = coerced["season"].astype("int32")
    coerced["basin"] = coerced["basin"].astype("string")
    coerced["init_time_utc"] = coerced["init_time_utc"].astype("string")
    coerced["valid_time_utc"] = coerced["valid_time_utc"].astype("string")
    coerced["lead_hour"] = coerced["lead_hour"].astype("int32")
    coerced["channel"] = coerced["channel"].astype("string")
    coerced["pred"] = coerced["pred"].astype("float64")
    coerced["target"] = coerced["target"].astype("float64")
    coerced["mask"] = coerced["mask"].astype("bool")
    return coerced


def long_to_pivot(
    long_frame: pd.DataFrame,
    *,
    columns: tuple[str, str] = ("lead_hour", "channel"),
    values: tuple[str, ...] = ("pred", "target"),
) -> pd.DataFrame:
    """Pivot the tidy-long IBTrACS table to a wide build_splits-style frame.

    Args:
        long_frame: A DataFrame with the tidy-long schema.
        columns: Pair of column names whose unique combinations become wide columns.
        values: Value columns to spread; produces ``f"lead_NNNh_{channel}_{value}"``-style
            keys for the default ``columns``.

    Returns:
        Wide DataFrame indexed by ``sample_id`` with one column per
        ``(value, lead_hour, channel)`` triple.
    """
    if long_frame.empty:
        return pd.DataFrame()

    pivoted = long_frame.pivot_table(
        index=["sample_id", "storm_id", "season", "basin", "init_time_utc"],
        columns=list(columns),
        values=list(values),
        aggfunc="first",
        dropna=False,
    )

    # Flatten the (value, lead_hour, channel) MultiIndex into lead_NNNh_<channel>_<value>.
    flat_columns: list[str] = []
    for col in pivoted.columns:
        value_name, lead_hour, channel = col
        flat_columns.append(f"lead_{int(lead_hour):03d}h_{channel}_{value_name}")
    pivoted.columns = flat_columns
    return pivoted.reset_index()
