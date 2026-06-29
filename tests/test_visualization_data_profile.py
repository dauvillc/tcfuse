"""Unit tests for windows-setup profiling figures (synthetic index only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tcfuse.data.visualization.data_profile import (
    compute_split_summary,
    plot_basin_distribution,
    plot_sample_timeline,
    plot_samples_per_season,
    plot_source_availability,
    plot_sources_per_window_hist,
    plot_target_distribution,
    plot_windows_per_storm,
)


def _synthetic_index(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Build a tiny long-format windows index spanning a couple of storms.

    Each window has two input snapshots (pmw_gmi, ir) plus one target row; the
    layout mimics the parquet produced by ``build_windows.py``.
    """
    rows: list[dict[str, object]] = []
    # Two storms per season, two windows per storm.
    for season in seasons:
        for storm in range(2):
            sid = f"{season}STORM{storm}"
            for window in range(2):
                window_id = f"{sid}_w{window}"
                ref_time = f"{season}-08-{10 + window:02d}T00:00:00"
                # Common window-level attributes shared by every snapshot row.
                base = {
                    "window_id": window_id,
                    "sid": sid,
                    "basin": "NA" if storm == 0 else "WP",
                    "subbasin": "MM",
                    "season": season,
                    "usa_atcf_id": None,
                    "window_start_time_utc": ref_time,
                    "window_end_time_utc": ref_time,
                    "window_ref_time_utc": ref_time,
                }
                # Two input sources and one target snapshot per window.
                for name, is_target in (("pmw_gmi", False), ("ir", False), ("best_track", True)):
                    rows.append(
                        {**base, "source_name": name, "time_utc": ref_time, "is_target": is_target}
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def split_frames() -> dict[str, pd.DataFrame]:
    """Two splits with disjoint seasons, mirroring a season-based split."""
    return {
        "train": _synthetic_index((2018, 2019)),
        "val": _synthetic_index((2020,)),
    }


def test_compute_split_summary_counts(split_frames: dict[str, pd.DataFrame]) -> None:
    """Summary reports correct sample / storm / snapshot counts per split."""
    summary = compute_split_summary(split_frames)

    # train: 2 seasons x 2 storms x 2 windows = 8 samples, 4 storms, 24 snapshots.
    assert int(summary.loc["train", "n_samples"]) == 8
    assert int(summary.loc["train", "n_storms"]) == 4
    assert int(summary.loc["train", "n_snapshots"]) == 24
    assert float(summary.loc["train", "avg_sources_per_window"]) == 3.0
    # val: 1 season x 2 storms x 2 windows = 4 samples.
    assert int(summary.loc["val", "n_samples"]) == 4


@pytest.mark.parametrize(
    "plot_fn",
    [
        plot_sample_timeline,
        plot_samples_per_season,
        plot_source_availability,
        plot_target_distribution,
        plot_sources_per_window_hist,
        plot_basin_distribution,
        plot_windows_per_storm,
    ],
)
def test_plot_functions_return_fig_ax(plot_fn, split_frames: dict[str, pd.DataFrame]) -> None:
    """Each plotting function returns a (Figure, Axes) without error."""
    fig, ax = plot_fn(split_frames)

    assert isinstance(fig, plt.Figure)
    assert ax is not None
    plt.close(fig)


def test_plot_saves_svg(tmp_path, split_frames: dict[str, pd.DataFrame]) -> None:
    """A save_path writes an SVG next to the requested path."""
    save_path = tmp_path / "timeline"
    fig, _ax = plot_sample_timeline(split_frames, save_path=save_path)

    assert save_path.with_suffix(".svg").exists()
    plt.close(fig)
