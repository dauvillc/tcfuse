"""Windows-setup profiling figures computed from the long-format window index.

These functions characterize the train/val/test samples produced by a given
windows setup *without* loading any HDF5 or iterating the dataset: everything is
derived from the windows-index parquet (one row per ``window_id x source_name x
time_utc``). Each function takes a ``{split: DataFrame}`` mapping of those long
frames, aggregates internally, and follows the project visualization
conventions (``setup_style()`` at import, ``COL1/COL2`` sizing, ``save_fig()``
SVG output, ``plot_<thing>(...) -> (fig, ax)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcfuse.data.visualization.style import (
    AR_GOLDEN,
    COL1,
    COL2,
    save_fig,
    setup_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Apply publication-quality rcParams at import time (project-wide convention).
setup_style()

# Stable color per split so every figure reads the same way.
SPLIT_COLORS: dict[str, str] = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}

# Default number of horizontal bins for the sample timeline.
_TIMELINE_BINS = 80


def _to_int(value: object) -> int:
    """Cast a pandas/numpy scalar to a plain int (isolates stub-type noise)."""
    return int(cast(SupportsInt, value))


def _split_color(split: str, fallback_cycle: cycle[str]) -> str:
    """Return a stable color for a split name, falling back to the prop cycle."""
    # Known splits get a fixed color; unknown names draw from the prop cycle.
    return SPLIT_COLORS.get(split, next(fallback_cycle))


def _window_level(index_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long index to one row per ``window_id`` (sample level)."""
    # One snapshot row per window is enough for window-level attributes.
    return index_df.drop_duplicates("window_id")


def _grouped_bars(
    ax: Axes,
    categories: list[str],
    counts_by_split: dict[str, list[int]],
    *,
    horizontal: bool = False,
) -> None:
    """Draw side-by-side bars: one bar group per category, one bar per split.

    Args:
        ax:              Target axes.
        categories:      Ordered category labels along the category axis.
        counts_by_split: Maps split name -> per-category counts (same length as
                         ``categories``).
        horizontal:      Draw horizontal bars when True, vertical otherwise.
    """
    # Integer positions for each category group.
    positions = np.arange(len(categories))
    n_splits = len(counts_by_split)
    # Total group width 0.8 shared evenly across the splits.
    bar_width = 0.8 / max(n_splits, 1)
    fallback: cycle[str] = cycle([p["color"] for p in plt.rcParams["axes.prop_cycle"]])

    # One offset bar series per split.
    for series_idx, (split, counts) in enumerate(counts_by_split.items()):
        # Center the group of bars around each category position.
        offset = (series_idx - (n_splits - 1) / 2) * bar_width
        color = _split_color(split, fallback)
        if horizontal:
            ax.barh(positions + offset, counts, height=bar_width, label=split, color=color)
        else:
            ax.bar(positions + offset, counts, width=bar_width, label=split, color=color)

    # Label the category axis with the category names.
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45, ha="right")


def compute_split_summary(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute one summary row per split from the long window indices.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.

    Returns:
        DataFrame indexed by split with columns ``n_samples``, ``n_storms``,
        ``n_seasons``, ``first_time``, ``last_time``, ``n_snapshots``,
        ``avg_sources_per_window``.
    """
    rows: dict[str, dict[str, object]] = {}
    # Aggregate each split independently.
    for split, index_df in split_frames.items():
        # Window-level frame for per-sample counts.
        windows = _window_level(index_df)
        n_samples = _to_int(windows["window_id"].nunique())
        n_snapshots = len(index_df)
        # Reference times bound the temporal coverage of the split.
        ref_times = pd.to_datetime(windows["window_ref_time_utc"], utc=True)
        rows[split] = {
            "n_samples": n_samples,
            "n_storms": _to_int(windows["sid"].nunique()),
            "n_seasons": _to_int(windows["season"].nunique()),
            "first_time": ref_times.min(),
            "last_time": ref_times.max(),
            "n_snapshots": n_snapshots,
            # Average snapshots (input + target) bundled into each window.
            "avg_sources_per_window": n_snapshots / n_samples if n_samples else 0.0,
        }
    # Stack the per-split dicts into a split-indexed DataFrame.
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_sample_timeline(
    split_frames: dict[str, pd.DataFrame],
    *,
    n_bins: int = _TIMELINE_BINS,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of sample reference times over the full record, one series per split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        n_bins:       Number of shared time bins along the x-axis.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Collect window-level reference times (matplotlib date floats) per split.
    times_by_split: dict[str, np.ndarray] = {}
    for split, index_df in split_frames.items():
        windows = _window_level(index_df)
        ref_times = pd.to_datetime(windows["window_ref_time_utc"], utc=True)
        times_by_split[split] = mdates.date2num(ref_times.dt.tz_convert(None))

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))

    # Shared bin edges across all splits so the series are directly comparable.
    nonempty = [t for t in times_by_split.values() if t.size]
    all_times = np.concatenate(nonempty) if nonempty else np.array([])
    if all_times.size:
        bin_edges = np.linspace(all_times.min(), all_times.max(), n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Overlay a translucent histogram per split.
    fallback: cycle[str] = cycle([p["color"] for p in plt.rcParams["axes.prop_cycle"]])
    bins = cast(Sequence[float], bin_edges)
    for split, times in times_by_split.items():
        ax.hist(times, bins=bins, alpha=0.6, label=split, color=_split_color(split, fallback))

    # Format the x-axis as calendar dates.
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.set_xlabel("Sample reference time (UTC)")
    ax.set_ylabel("Number of samples")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_samples_per_season(
    split_frames: dict[str, pd.DataFrame],
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar chart of sample count per season, grouped by split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Per-split sample count by season (window-level).
    counts_by_split_season: dict[str, dict[int, int]] = {}
    seasons: set[int] = set()
    for split, index_df in split_frames.items():
        windows = _window_level(index_df)
        per_season = windows.groupby("season")["window_id"].nunique()
        counts_by_split_season[split] = {_to_int(k): _to_int(v) for k, v in per_season.items()}
        seasons.update(counts_by_split_season[split].keys())

    # Ordered season axis shared across splits.
    season_axis = sorted(seasons)
    categories = [str(s) for s in season_axis]
    counts_by_split = {
        split: [counts_by_split_season[split].get(s, 0) for s in season_axis]
        for split in counts_by_split_season
    }

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))
    _grouped_bars(ax, categories, counts_by_split)
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of samples")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_source_availability(
    split_frames: dict[str, pd.DataFrame],
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Fraction of windows containing each source, grouped by split.

    A source counts as present in a window when it has at least one snapshot row
    in that window (input or target).

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Per-split availability fraction by source name.
    frac_by_split_source: dict[str, dict[str, float]] = {}
    sources: set[str] = set()
    for split, index_df in split_frames.items():
        total_windows = _to_int(index_df["window_id"].nunique())
        # Windows containing each source name (any snapshot of it).
        per_source = index_df.groupby("source_name")["window_id"].nunique()
        frac_by_split_source[split] = {
            str(name): (_to_int(count) / total_windows if total_windows else 0.0)
            for name, count in per_source.items()
        }
        sources.update(frac_by_split_source[split].keys())

    # Mean availability across splits drives a readable source ordering.
    def _mean_frac(source: str) -> float:
        per_split = [fracs.get(source, 0.0) for fracs in frac_by_split_source.values()]
        return float(np.mean(per_split))

    source_axis = sorted(sources, key=_mean_frac)
    # Reuse the grouped-bar helper with float "counts" (fractions).
    counts_by_split = {
        split: [frac_by_split_source[split].get(s, 0.0) for s in source_axis]  # type: ignore[misc]
        for split in frac_by_split_source
    }

    # Height grows with the number of source rows.
    fig_height = max(COL1 * AR_GOLDEN, 0.4 * len(source_axis) + 1.0)
    fig, ax = plt.subplots(figsize=(COL2, fig_height))
    _grouped_bars(ax, source_axis, cast("dict[str, list[int]]", counts_by_split), horizontal=True)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction of samples containing source")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_target_distribution(
    split_frames: dict[str, pd.DataFrame],
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Count of windows whose target is each source, grouped by split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Per-split window count by target source name.
    counts_by_split_source: dict[str, dict[str, int]] = {}
    sources: set[str] = set()
    for split, index_df in split_frames.items():
        # Keep only target snapshot rows, then count windows per source name.
        targets = index_df[index_df["is_target"]]
        per_source = targets.groupby("source_name")["window_id"].nunique()
        counts_by_split_source[split] = {str(k): _to_int(v) for k, v in per_source.items()}
        sources.update(counts_by_split_source[split].keys())

    # Ordered source axis shared across splits.
    source_axis = sorted(sources)
    counts_by_split = {
        split: [counts_by_split_source[split].get(s, 0) for s in source_axis]
        for split in counts_by_split_source
    }

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))
    _grouped_bars(ax, source_axis, counts_by_split)
    ax.set_xlabel("Target source")
    ax.set_ylabel("Number of samples")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_sources_per_window_hist(
    split_frames: dict[str, pd.DataFrame],
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of the number of snapshot rows per window, one series per split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Snapshots-per-window counts per split.
    counts_by_split: dict[str, np.ndarray] = {}
    max_count = 1
    for split, index_df in split_frames.items():
        per_window = index_df.groupby("window_id").size().to_numpy()
        counts_by_split[split] = per_window
        if per_window.size:
            max_count = max(max_count, _to_int(per_window.max()))

    # Integer bin edges spanning 1..max_count so each count gets its own bin.
    bins = cast(Sequence[float], np.arange(0.5, max_count + 1.5, 1.0))

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))
    fallback: cycle[str] = cycle([p["color"] for p in plt.rcParams["axes.prop_cycle"]])
    # Overlay a translucent histogram per split.
    for split, counts in counts_by_split.items():
        ax.hist(counts, bins=bins, alpha=0.6, label=split, color=_split_color(split, fallback))

    ax.set_xlabel("Sources (snapshots) per sample")
    ax.set_ylabel("Number of samples")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_basin_distribution(
    split_frames: dict[str, pd.DataFrame],
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar chart of sample count per ocean basin, grouped by split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Per-split sample count by basin (window-level).
    counts_by_split_basin: dict[str, dict[str, int]] = {}
    basins: set[str] = set()
    for split, index_df in split_frames.items():
        windows = _window_level(index_df)
        per_basin = windows.groupby("basin")["window_id"].nunique()
        counts_by_split_basin[split] = {str(k): _to_int(v) for k, v in per_basin.items()}
        basins.update(counts_by_split_basin[split].keys())

    # Ordered basin axis shared across splits.
    basin_axis = sorted(basins)
    counts_by_split = {
        split: [counts_by_split_basin[split].get(b, 0) for b in basin_axis]
        for split in counts_by_split_basin
    }

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))
    _grouped_bars(ax, basin_axis, counts_by_split)
    ax.set_xlabel("Basin")
    ax.set_ylabel("Number of samples")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_windows_per_storm(
    split_frames: dict[str, pd.DataFrame],
    *,
    n_bins: int = 30,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of the number of samples (windows) per storm, one series per split.

    Args:
        split_frames: Maps split name -> long-format windows-index DataFrame.
        n_bins:       Number of shared histogram bins.
        title:        Optional figure title.
        save_path:    If provided, save the figure here (SVG).

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Windows-per-storm counts per split (window-level group by sid).
    counts_by_split: dict[str, np.ndarray] = {}
    max_count = 1
    for split, index_df in split_frames.items():
        windows = _window_level(index_df)
        per_storm = windows.groupby("sid")["window_id"].nunique().to_numpy()
        counts_by_split[split] = per_storm
        if per_storm.size:
            max_count = max(max_count, _to_int(per_storm.max()))

    # Shared integer-aligned bin edges across all splits.
    bins = cast(Sequence[float], np.linspace(0.5, max_count + 0.5, min(n_bins, max_count) + 1))

    fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))
    fallback: cycle[str] = cycle([p["color"] for p in plt.rcParams["axes.prop_cycle"]])
    # Overlay a translucent histogram per split.
    for split, counts in counts_by_split.items():
        ax.hist(counts, bins=bins, alpha=0.6, label=split, color=_split_color(split, fallback))

    ax.set_xlabel("Samples (windows) per storm")
    ax.set_ylabel("Number of storms")
    ax.legend(title="split")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax
