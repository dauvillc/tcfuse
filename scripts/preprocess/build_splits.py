#!/usr/bin/env python3
"""Build best-track window sample indices and split them by season.

Reads ``{preprocessed_data}/index.parquet`` produced by ``assemble.py`` and
writes three sample-index parquet files — ``train.parquet``, ``val.parquet``,
``test.parquet`` — to the same directory.  The canonical ``index.parquet``
remains one row per source snapshot; the split files are one row per training
sample window anchored on ``ibtracs_best_track``.

Each sample spans the configured best-track lead hours from an anchor time.  By
default a sample covers ``t0`` through ``t0 + 30h`` and requires finite USA wind,
latitude, and longitude at ``+0h``, ``+6h``, and ``+30h``.  Intermediate leads
may be absent or NaN and are preserved in the output row.

Split seasons are read from ``cfg.splits`` (``conf/preproc.yaml``):
  - **val**:   seasons listed under ``splits.val``
  - **test**:  seasons listed under ``splits.test``
  - **train**: all remaining seasons

Run from the project root:
    python scripts/preprocess/build_splits.py [paths=jz]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from tcfuse.utils.time import to_compact_time


def _parse_time(value: Any) -> pd.Timestamp:
    """Parse a timestamp value as UTC for exact lead-time matching."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _isoformat_naive_utc(value: pd.Timestamp) -> str:
    """Return the repository's naive-UTC ISO timestamp representation."""
    return value.tz_convert("UTC").tz_localize(None).isoformat()


def _lead_prefix(lead_hour: int) -> str:
    """Return the fixed column prefix for a forecast lead."""
    return f"lead_{lead_hour:03d}h"


def _float_or_nan(value: Any) -> float:
    """Convert a scalar value to float while preserving missing values as NaN."""
    return np.nan if pd.isna(value) else float(value)


def _empty_window_index(leads_hours: list[int], required_columns: list[str]) -> pd.DataFrame:
    """Return an empty sample-index DataFrame with the expected schema."""
    columns = [
        "sample_id",
        "storm_id",
        "basin",
        "season",
        "atcf_id",
        "anchor_time_utc",
        "window_start_time_utc",
        "window_end_time_utc",
    ]
    for lead_hour in leads_hours:
        prefix = _lead_prefix(lead_hour)
        columns.extend([f"{prefix}_time_utc", f"{prefix}_available"])
        columns.extend(f"{prefix}_{column}" for column in required_columns)
    return pd.DataFrame(columns=columns)


def _is_finite(row: pd.Series | None, columns: list[str]) -> bool:
    """Return True when all requested columns are present and finite in a row."""
    if row is None:
        return False
    for column in columns:
        value = row[column] if column in row.index else np.nan
        if pd.isna(value) or not np.isfinite(float(value)):
            return False
    return True


def _first_rows_by_time(rows: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Index a storm's best-track rows by timestamp, keeping the first duplicate."""
    indexed: dict[pd.Timestamp, pd.Series] = {}
    sort_column = "_time" if "_time" in rows.columns else "snapshot_time_utc"
    for _, row in rows.sort_values(sort_column).iterrows():
        timestamp = _parse_time(row["snapshot_time_utc"])
        indexed.setdefault(timestamp, row)
    return indexed


def build_window_index(
    assembled_index: pd.DataFrame,
    source_name: str,
    leads_hours: list[int],
    required_leads_hours: list[int],
    required_columns: list[str],
) -> pd.DataFrame:
    """Build one sample row per valid best-track forecast window.

    Args:
        assembled_index: Canonical assembled index with one row per source snapshot.
        source_name: Best-track source used to anchor samples.
        leads_hours: Lead times, in hours, to materialise in each sample row.
        required_leads_hours: Lead times that must be present and finite.
        required_columns: Best-track columns required at each required lead.

    Returns:
        DataFrame with one row per valid sample window.
    """
    if not set(required_leads_hours).issubset(set(leads_hours)):
        raise ValueError("required_leads_hours must be a subset of leads_hours.")

    best_track = assembled_index[assembled_index["source_name"] == source_name].copy()
    if best_track.empty:
        return _empty_window_index(leads_hours, required_columns)

    # Parse timestamps once so sorting and lead matching use a consistent timezone.
    best_track["_time"] = best_track["snapshot_time_utc"].map(_parse_time)
    best_track = cast(Any, best_track).sort_values(["storm_id", "_time"]).reset_index(drop=True)

    sample_rows: list[dict[str, Any]] = []
    for storm_id_value, storm_rows in best_track.groupby("storm_id", sort=True):
        storm_id = str(storm_id_value)
        rows_by_time = _first_rows_by_time(storm_rows)
        for anchor_time in sorted(rows_by_time):
            lead_rows: dict[int, pd.Series | None] = {}
            for lead_hour in leads_hours:
                lead_time = anchor_time + pd.Timedelta(hours=lead_hour)
                lead_rows[lead_hour] = rows_by_time.get(lead_time)

            # Required leads must have finite USA wind and position metadata.
            if not all(
                _is_finite(lead_rows[lead_hour], required_columns)
                for lead_hour in required_leads_hours
            ):
                continue

            anchor_row = rows_by_time[anchor_time]
            sample_id = f"{storm_id}_{to_compact_time(anchor_time)}"
            sample: dict[str, Any] = {
                "sample_id": sample_id,
                "storm_id": storm_id,
                "basin": anchor_row.get("basin"),
                "season": int(anchor_row["season"]),
                "atcf_id": anchor_row.get("atcf_id"),
                "anchor_time_utc": _isoformat_naive_utc(anchor_time),
                "window_start_time_utc": _isoformat_naive_utc(anchor_time),
                "window_end_time_utc": _isoformat_naive_utc(
                    anchor_time + pd.Timedelta(hours=max(leads_hours))
                ),
            }

            # Fixed lead columns are easy to consume from Parquet in PyTorch datasets.
            for lead_hour in leads_hours:
                prefix = _lead_prefix(lead_hour)
                lead_time = anchor_time + pd.Timedelta(hours=lead_hour)
                row = lead_rows[lead_hour]
                sample[f"{prefix}_time_utc"] = _isoformat_naive_utc(lead_time)
                sample[f"{prefix}_available"] = row is not None
                for column in required_columns:
                    sample[f"{prefix}_{column}"] = (
                        _float_or_nan(row[column])
                        if row is not None and column in row.index
                        else np.nan
                    )

            sample_rows.append(sample)

    if not sample_rows:
        return _empty_window_index(leads_hours, required_columns)
    return (
        pd.DataFrame(sample_rows)
        .sort_values(["storm_id", "anchor_time_utc"])
        .reset_index(drop=True)
    )


def split_by_season(
    samples: pd.DataFrame,
    val_seasons: set[int],
    test_seasons: set[int],
) -> dict[str, pd.DataFrame]:
    """Split sample windows into train/val/test DataFrames by season."""
    overlap = val_seasons & test_seasons
    if overlap:
        raise ValueError(
            f"Val and test season sets overlap: {sorted(overlap)}. "
            "Fix cfg.splits in conf/preproc.yaml."
        )

    if samples.empty:
        return {
            "train": samples.copy(),
            "val": samples.copy(),
            "test": samples.copy(),
        }

    test_mask = samples["season"].isin(test_seasons)
    val_mask = samples["season"].isin(val_seasons) & ~test_mask
    train_mask = ~test_mask & ~val_mask
    return {
        "train": samples[train_mask].reset_index(drop=True),
        "val": samples[val_mask].reset_index(drop=True),
        "test": samples[test_mask].reset_index(drop=True),
    }


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Build season-based train/val/test best-track window index files."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    # Resolve paths from config.
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    index_path = assembled_root / "index.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Assembled index not found at {index_path}. Run scripts/preprocess/assemble.py first."
        )

    # Read season lists from config; convert to sets for O(1) lookup.
    val_seasons: set[int] = set(cfg["splits"]["val"])
    test_seasons: set[int] = set(cfg["splits"]["test"])

    # Load the global assembled index.
    print(f"Loading index from {index_path} …")
    index = pd.read_parquet(index_path)
    print(f"  {len(index)} rows, {index['storm_id'].nunique()} unique storms.")

    window_cfg = cast(dict[str, Any], cfg["window_index"])
    leads_hours = [int(h) for h in window_cfg["leads_hours"]]
    required_leads_hours = [int(h) for h in window_cfg["required_leads_hours"]]
    required_columns = [
        *[str(c) for c in window_cfg.get("required_channels", [])],
        *[str(c) for c in window_cfg.get("required_meta", [])],
    ]
    source_name = str(window_cfg["source_name"])

    print(
        f"Building {source_name} windows with leads {leads_hours} "
        f"(required: {required_leads_hours}) …"
    )
    samples = build_window_index(
        index,
        source_name=source_name,
        leads_hours=leads_hours,
        required_leads_hours=required_leads_hours,
        required_columns=required_columns,
    )
    n_storms = samples["storm_id"].nunique() if len(samples) else 0
    print(f"  {len(samples)} samples, {n_storms} storms.")

    # Slice sample windows into season-based splits.
    splits = split_by_season(samples, val_seasons, test_seasons)

    # Write each split and print a summary row.
    print(f"\n{'Split':<8}  {'Storms':>7}  {'Samples':>9}  {'Seasons'}")
    print("-" * 55)
    for split_name, df in splits.items():
        out_path = assembled_root / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        seasons_str = ", ".join(str(s) for s in sorted(df["season"].unique()))
        print(f"{split_name:<8}  {df['storm_id'].nunique():>7}  {len(df):>9}  {seasons_str}")

    # Sanity check: every sample appears in exactly one split.
    total = sum(len(df) for df in splits.values())
    assert total == len(samples), (
        f"Sample count mismatch after splitting: {total} != {len(samples)}. "
        "This is a bug in build_splits.py."
    )
    print(f"\nWrote train/val/test window-index parquet files to {assembled_root}")


if __name__ == "__main__":
    main()
