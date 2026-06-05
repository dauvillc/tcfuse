#!/usr/bin/env python3
"""Stage 3B — build window indexes from season-split source indexes.

Reads ``train.parquet``, ``val.parquet``, ``test.parquet`` produced by
``build_splits.py`` and writes one long-format window index per split to
``{preprocessed_data}/{windows_setup.name}/``.

Each row in a window index corresponds to one source snapshot that participates
in a window.  A window is anchored on a snapshot from one of the configured
``target_sources``; the window's reference time is that snapshot's ``time_utc``.
Input sources are all snapshots from the same storm whose ``time_utc`` falls
within ``[ref_time + start_time_offset, ref_time + end_time_offset]``.  The
target snapshot itself is always included with ``is_target = True``, even when
it falls outside the input time range (e.g. when ``end_time_offset`` is
negative).

Window configuration lives under ``conf/windows_setup/`` and is selected via
the ``windows_setup`` Hydra config group (default: ``ibtracs_forecast_24h``):

    python scripts/preprocess/build_windows.py [paths=jz] [windows_setup=<name>]

Output schema (one row per window × source snapshot):

    window_id | sid | basin | subbasin | season | usa_atcf_id |
    window_start_time_utc | window_end_time_utc | window_ref_time_utc |
    source_name | time_utc | is_target
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.preprocess.utils.runner import resolve_preproc_cfg
from tcfuse.utils.time import to_compact_time

# Ordered output columns; all columns from the split index row are appended after.
_WINDOW_COLS = [
    "window_id",
    "sid",
    "basin",
    "subbasin",
    "season",
    "usa_atcf_id",
    "window_start_time_utc",
    "window_end_time_utc",
    "window_ref_time_utc",
    "source_name",
    "time_utc",
    "is_target",
]


def _isoformat(ts: pd.Timestamp) -> str:
    """Return a naive-UTC ISO timestamp string."""
    return ts.tz_localize(None).isoformat() if ts.tzinfo is not None else ts.isoformat()


def build_window_index(
    split_df: pd.DataFrame,
    target_sources: set[str],
    start_time_offset: pd.Timedelta,
    end_time_offset: pd.Timedelta,
    desc: str = "windows",
) -> pd.DataFrame:
    """Build a long-format window index from a season-split source index.

    Args:
        split_df: One of train/val/test.parquet — one row per source snapshot.
        target_sources: Source names whose snapshots anchor windows.
        start_time_offset: Window start relative to reference time (typically
            negative, e.g. ``pd.Timedelta('-24H')``).
        end_time_offset: Window end relative to reference time (may be negative
            for pure forecasting or positive for assimilation).
        desc: Label for the per-storm tqdm progress bar.

    Returns:
        Long-format DataFrame with one row per (window, source snapshot).
    """
    output_rows: list[dict[str, Any]] = []

    # One progress step per storm (SID).
    n_storms = cast(int, split_df["sid"].nunique())
    storm_groups = split_df.groupby("sid", sort=True)
    for sid_value, storm_rows in tqdm(storm_groups, total=n_storms, desc=desc, leave=False):
        sid = str(sid_value)

        # Parse timestamps once per storm for fast vectorised comparison.
        storm_times = pd.to_datetime(storm_rows["time_utc"], utc=False)

        target_mask = storm_rows["source_name"].isin(list(target_sources))
        target_rows = storm_rows[target_mask]

        for target_idx, target_row in target_rows.iterrows():
            ref_time = cast(pd.Timestamp, storm_times.at[target_idx])
            window_start = ref_time + start_time_offset
            window_end = ref_time + end_time_offset

            in_window = (storm_times >= window_start) & (storm_times <= window_end)
            window_rows = storm_rows[in_window]

            target_time_utc = str(target_row["time_utc"])
            target_source_name = str(target_row["source_name"])
            window_id = f"{sid}_{target_source_name}_{to_compact_time(ref_time)}"

            window_start_str = _isoformat(window_start)
            window_end_str = _isoformat(window_end)
            window_ref_str = _isoformat(ref_time)

            target_in_window = False
            for _, src_row in window_rows.iterrows():
                is_target = (
                    str(src_row["time_utc"]) == target_time_utc
                    and str(src_row["source_name"]) == target_source_name
                )
                if is_target:
                    target_in_window = True
                output_rows.append(
                    {
                        "window_id": window_id,
                        "sid": src_row["sid"],
                        "basin": src_row.get("basin"),
                        "subbasin": src_row.get("subbasin"),
                        "season": src_row.get("season"),
                        "usa_atcf_id": src_row.get("usa_atcf_id"),
                        "window_start_time_utc": window_start_str,
                        "window_end_time_utc": window_end_str,
                        "window_ref_time_utc": window_ref_str,
                        "source_name": src_row["source_name"],
                        "time_utc": src_row["time_utc"],
                        "is_target": is_target,
                    }
                )

            # Always include the target, even when outside the input window.
            if not target_in_window:
                output_rows.append(
                    {
                        "window_id": window_id,
                        "sid": target_row["sid"],
                        "basin": target_row.get("basin"),
                        "subbasin": target_row.get("subbasin"),
                        "season": target_row.get("season"),
                        "usa_atcf_id": target_row.get("usa_atcf_id"),
                        "window_start_time_utc": window_start_str,
                        "window_end_time_utc": window_end_str,
                        "window_ref_time_utc": window_ref_str,
                        "source_name": target_row["source_name"],
                        "time_utc": target_row["time_utc"],
                        "is_target": True,
                    }
                )

    if not output_rows:
        return pd.DataFrame(columns=_WINDOW_COLS)
    return cast(
        pd.DataFrame,
        pd.DataFrame(output_rows, columns=_WINDOW_COLS).sort_values(
            ["sid", "window_ref_time_utc", "time_utc"]
        ).reset_index(drop=True),
    )


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Build window indexes for all three splits using the active windows_setup config."""
    cfg = resolve_preproc_cfg(raw_cfg)

    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    windows_cfg = cast(dict[str, Any], cfg["windows_setup"])

    windows_name = str(windows_cfg["name"])
    target_sources: set[str] = set(str(s) for s in windows_cfg["target_sources"])
    start_time_offset = cast(
        pd.Timedelta, pd.Timedelta(str(windows_cfg["start_time_offset"]))
    )
    end_time_offset = cast(
        pd.Timedelta, pd.Timedelta(str(windows_cfg["end_time_offset"]))
    )

    windows_root = assembled_root / windows_name
    windows_root.mkdir(parents=True, exist_ok=True)

    print(f"Windows config: {windows_name}")
    print(f"  target_sources : {sorted(target_sources)}")
    print(f"  start_time_offset: {start_time_offset}")
    print(f"  end_time_offset  : {end_time_offset}")
    print(f"  output dir       : {windows_root}\n")

    print(f"{'Split':<8}  {'Windows':>9}  {'Rows':>9}")
    print("-" * 32)
    for split_name in ("train", "val", "test"):
        split_path = assembled_root / f"{split_name}.parquet"
        if not split_path.exists():
            print(f"{split_name:<8}  (missing {split_path.name})")
            continue

        split_df = pd.read_parquet(split_path)
        windows_df = build_window_index(
            split_df,
            target_sources,
            start_time_offset,
            end_time_offset,
            desc=f"{split_name} windows",
        )

        out_path = windows_root / f"{split_name}_windows.parquet"
        windows_df.to_parquet(out_path, index=False)

        n_windows = windows_df["window_id"].nunique() if len(windows_df) else 0
        print(f"{split_name:<8}  {n_windows:>9}  {len(windows_df):>9}")

    print(f"\nWrote window index files to {windows_root}")


if __name__ == "__main__":
    main()
