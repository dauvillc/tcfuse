"""Tests for build_window_index input-source specification."""

from __future__ import annotations

from typing import cast

import pandas as pd
from scripts.preprocess.build_windows import build_window_index

_SID = "2016292N14270"
_REF = cast(pd.Timestamp, pd.Timestamp("2016-10-18T12:00:00"))
_TARGET_SOURCES = {"ibtracs_best_track"}


def _before(offset: str) -> pd.Timestamp:
    """Reference time minus a pd.Timedelta offset, e.g. ``_before("6H")``."""
    return cast(pd.Timestamp, _REF - pd.Timedelta(offset))


def _row(source_name: str, time: pd.Timestamp) -> dict[str, object]:
    """One synthetic source-index row for a single storm."""
    return {
        "sid": _SID,
        "source_name": source_name,
        "time_utc": time.isoformat(),
        "basin": "AL",
        "subbasin": "GM",
        "season": 2016,
        "usa_atcf_id": "AL102016",
    }


def _split_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a split_df from rows, always anchored on one IBTrACS target at _REF."""
    return pd.DataFrame([_row("ibtracs_best_track", _REF), *rows])


# Two era5 periods, both requiring one snapshot: ~-12h and ~-6h before target.
_ERA5_TWO_PERIODS = {"era5": [("-13H", "-11H", 1), ("-7H", "-5H", 1)]}


def test_window_kept_when_required_snapshots_present() -> None:
    """A window with one era5 snapshot in each period is kept, emitting both + target."""
    split_df = _split_df(
        [
            _row("era5_surface", _before("12H")),
            _row("era5_surface", _before("6H")),
        ]
    )

    windows = build_window_index(split_df, _TARGET_SOURCES, _ERA5_TWO_PERIODS)

    # One window: two era5 input rows plus the target row.
    assert windows["window_id"].nunique() == 1
    assert (windows["source_name"] == "era5_surface").sum() == 2
    assert (windows["is_target"]).sum() == 1
    # Window span is the union of periods: [-13H, -5H].
    assert windows["window_start_time_utc"].iloc[0] == (_REF - pd.Timedelta("13H")).isoformat()
    assert windows["window_end_time_utc"].iloc[0] == (_REF - pd.Timedelta("5H")).isoformat()


def test_window_discarded_when_a_period_is_empty() -> None:
    """Missing the -6h era5 snapshot discards the whole window."""
    split_df = _split_df([_row("era5_surface", _before("12H"))])

    windows = build_window_index(split_df, _TARGET_SOURCES, _ERA5_TWO_PERIODS)

    assert windows.empty


def test_prefix_match_counts_across_sources() -> None:
    """Key "pmw" matches both pmw_gmi and pmw_tmi when counting min_required."""
    constraints = {"pmw": [("-7H", "-5H", 2)]}
    split_df = _split_df(
        [
            _row("pmw_gmi", _before("6H")),
            _row("pmw_tmi", _before("6H")),
        ]
    )

    windows = build_window_index(split_df, _TARGET_SOURCES, constraints)

    # Both sub-sources count toward the requirement of 2, so the window survives.
    assert windows["window_id"].nunique() == 1
    assert set(windows.loc[windows["source_name"] != "ibtracs_best_track", "source_name"]) == {
        "pmw_gmi",
        "pmw_tmi",
    }


def test_min_required_zero_includes_without_discarding() -> None:
    """min_required=0 emits the source when present but never discards the window."""
    constraints = {"era5": [("-7H", "-5H", 0)]}

    # Present: emitted.
    with_era5 = build_window_index(
        _split_df([_row("era5_surface", _before("6H"))]),
        _TARGET_SOURCES,
        constraints,
    )
    assert (with_era5["source_name"] == "era5_surface").sum() == 1

    # Absent: window still kept (target only), not discarded.
    without_era5 = build_window_index(_split_df([]), _TARGET_SOURCES, constraints)
    assert without_era5["window_id"].nunique() == 1
    assert (without_era5["source_name"] == "era5_surface").sum() == 0
