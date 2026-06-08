#!/usr/bin/env python3
"""Stage 3B — build window indexes from season-split source indexes.

Reads ``train.parquet``, ``val.parquet``, ``test.parquet`` produced by
``build_splits.py`` and writes one long-format window index per split to
``{preprocessed_data}/{windows_setup.name}/``.

Each row in a window index corresponds to one source snapshot that participates
in a window.  A window is anchored on a snapshot from one of the configured
``target_sources``; the window's reference time is that snapshot's ``time_utc``.

Input sources are governed by the ``input_sources`` specification: a mapping
from source *type* to a list of ``(start_offset, end_offset, min_required)``
periods (``pd.Timedelta`` strings).  A snapshot from the same storm is emitted
into a window only when its source type is listed AND its ``time_utc`` falls
within one of that type's periods.  A window is discarded entirely if any listed
period contains fewer than ``min_required`` matching snapshots (use ``0`` to
include a source without requiring it).  Type keys match ``source_name`` by
prefix: ``"era5"`` matches ``era5_surface`` and ``"pmw"`` matches ``pmw_gmi`` /
``pmw_tmi`` (counts are summed across matching sources).  The window's overall
span is the union of all periods: ``[ref + min(start), ref + max(end)]``.  The
target snapshot itself is always included with ``is_target = True``, even when
it falls outside that span.

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


# One input period: snapshots of a source type in [ref + start, ref + end],
# with at least ``min_required`` of them needed for the window to survive.
InputPeriod = tuple[pd.Timedelta, pd.Timedelta, int]


def _parse_input_sources(
    input_sources: dict[str, list[tuple[str, str, int]]],
) -> dict[str, list[InputPeriod]]:
    """Parse the raw ``input_sources`` config into typed timedelta periods."""
    parsed: dict[str, list[InputPeriod]] = {}
    for source_type, periods in input_sources.items():
        parsed[str(source_type)] = [
            (
                cast(pd.Timedelta, pd.Timedelta(str(start))),
                cast(pd.Timedelta, pd.Timedelta(str(end))),
                int(min_required),
            )
            for start, end, min_required in periods
        ]
    return parsed


def _matches_source_type(source_names: pd.Series, source_type: str) -> pd.Series:
    """Prefix match: ``"era5"`` matches ``era5_surface``; ``"pmw"`` matches ``pmw_gmi``."""
    return (source_names == source_type) | source_names.str.startswith(f"{source_type}_")


def build_window_index(
    split_df: pd.DataFrame,
    target_sources: set[str],
    input_sources: dict[str, list[tuple[str, str, int]]],
    desc: str = "windows",
) -> pd.DataFrame:
    """Build a long-format window index from a season-split source index.

    Args:
        split_df: One of train/val/test.parquet — one row per source snapshot.
        target_sources: Source names whose snapshots anchor windows.
        input_sources: Mapping from source type to a list of
            ``(start_offset, end_offset, min_required)`` periods. A snapshot is
            emitted only when its source type is listed and it falls within one
            of that type's periods; a window is discarded when any period has
            fewer than ``min_required`` matching snapshots. Keys match
            ``source_name`` by prefix.
        desc: Label for the per-storm tqdm progress bar.

    Returns:
        Long-format DataFrame with one row per (window, source snapshot).
    """
    output_rows: list[dict[str, Any]] = []

    # Parse periods once and derive the union span used for window metadata.
    parsed_sources = _parse_input_sources(input_sources)
    all_periods = [period for periods in parsed_sources.values() for period in periods]
    span_start = min(start for start, _, _ in all_periods)
    span_end = max(end for _, end, _ in all_periods)

    # One progress step per storm (SID).
    n_storms = cast(int, split_df["sid"].nunique())
    storm_groups = split_df.groupby("sid", sort=True)
    for sid_value, storm_rows in tqdm(storm_groups, total=n_storms, desc=desc, leave=False):
        sid = str(sid_value)

        # Parse timestamps once per storm for fast vectorised comparison.
        storm_times = pd.to_datetime(storm_rows["time_utc"], utc=False)

        # Precompute the type mask for each source type once per storm.
        storm_source_names = cast(pd.Series, storm_rows["source_name"].astype(str))
        type_masks = {
            source_type: _matches_source_type(storm_source_names, source_type)
            for source_type in parsed_sources
        }

        target_mask = storm_rows["source_name"].isin(list(target_sources))
        target_rows = storm_rows[target_mask]

        for target_idx, target_row in target_rows.iterrows():
            ref_time = cast(pd.Timestamp, storm_times.at[target_idx])
            window_start = ref_time + span_start
            window_end = ref_time + span_end

            # Select which snapshots to emit, validating availability per period.
            emit_mask = pd.Series(False, index=storm_rows.index)
            window_ok = True
            for source_type, periods in parsed_sources.items():
                for start_off, end_off, min_required in periods:
                    in_period = (storm_times >= ref_time + start_off) & (
                        storm_times <= ref_time + end_off
                    )
                    period_mask = type_masks[source_type] & in_period
                    # Discard the whole window if this period is under-populated.
                    if int(period_mask.sum()) < min_required:
                        window_ok = False
                        break
                    emit_mask |= period_mask
                if not window_ok:
                    break

            # Skip windows that fail any availability constraint entirely.
            if not window_ok:
                continue

            window_rows = storm_rows[emit_mask]

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

    # input_sources is the canonical input specification (no top-level offsets).
    input_sources = cast(
        dict[str, list[tuple[str, str, int]]], windows_cfg.get("input_sources") or {}
    )
    if not input_sources:
        raise ValueError(
            f"windows_setup '{windows_name}' has no 'input_sources'. Declare at least one "
            "source type with (start_offset, end_offset, min_required) periods."
        )

    windows_root = assembled_root / windows_name
    windows_root.mkdir(parents=True, exist_ok=True)

    print(f"Windows config: {windows_name}")
    print(f"  target_sources : {sorted(target_sources)}")
    print("  input_sources  :")
    for source_type, periods in input_sources.items():
        print(f"    {source_type}: {[tuple(p) for p in periods]}")
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
            input_sources,
            desc=f"{split_name} windows",
        )

        out_path = windows_root / f"{split_name}_windows.parquet"
        windows_df.to_parquet(out_path, index=False)

        n_windows = windows_df["window_id"].nunique() if len(windows_df) else 0
        print(f"{split_name:<8}  {n_windows:>9}  {len(windows_df):>9}")

    print(f"\nWrote window index files to {windows_root}")


if __name__ == "__main__":
    main()
