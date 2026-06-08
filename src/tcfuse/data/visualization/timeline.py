"""Source availability timeline: one horizontal row per observation source.

Shows when each source in the assembled dataset index has observations.  Snapshot
times are aggregated into time bins and drawn as a rasterized strip so SVG/PDF
exports stay small while axis labels remain vector text.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcfuse.data.visualization.style import (
    AR_GOLDEN,
    COL1,
    COL2,
    SOURCE_COLORS,
    save_fig,
    setup_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Apply publication-quality rcParams at import time (project-wide convention).
setup_style()

# Minimum figure height (inches) regardless of source count.
_MIN_HEIGHT_IN = COL1 * AR_GOLDEN
# Height allocated per source row (inches).
_HEIGHT_PER_SOURCE_IN = 0.30
# Default cap on horizontal time bins — keeps raster layers small in vector exports.
_MAX_TIME_BINS = 2000
# Floor on horizontal time bins so short spans still resolve individual days.
_MIN_TIME_BINS = 50
# Alpha applied to occupied time bins in the raster layer.
_BIN_ALPHA = 0.75


def _get_source_color(
    source_name: str,
    color_cache: dict[str, str],
    fallback_cycle: cycle[str],
) -> str:
    """Return a stable color for a source name, cached across calls.

    Args:
        source_name:    Source identifier, e.g. ``"pmw_amsr2_gcomw1"``.
        color_cache:    Mutable dict accumulating already-assigned colors.
        fallback_cycle: Infinite iterator over fallback prop_cycle colors.

    Returns:
        Hex color string.
    """
    # Return cached assignment if already seen.
    if source_name in color_cache:
        return color_cache[source_name]

    # Substring-match against known SOURCE_COLORS keys (longest key first for determinism).
    matched = next(
        (v for k, v in sorted(SOURCE_COLORS.items(), key=lambda x: -len(x[0])) if k in source_name),
        None,
    )
    color = matched if matched is not None else next(fallback_cycle)
    color_cache[source_name] = color
    return color


def _resolve_n_time_bins(span_days: float, n_time_bins: int | None) -> int:
    """Return the number of UTC time bins for the raster timeline strip.

    Args:
        span_days:   Full time span in matplotlib date units (days).
        n_time_bins: Caller override; ``None`` picks an automatic value.

    Returns:
        Bin count clamped to ``[_MIN_TIME_BINS, _MAX_TIME_BINS]``.
    """
    # Honour an explicit bin count when the caller provides one.
    if n_time_bins is not None:
        return int(np.clip(n_time_bins, _MIN_TIME_BINS, _MAX_TIME_BINS))

    # Otherwise use ~one bin per day, capped for very long archives.
    auto_bins = max(_MIN_TIME_BINS, int(np.ceil(span_days)))
    return int(np.clip(auto_bins, _MIN_TIME_BINS, _MAX_TIME_BINS))


def _build_timeline_rgba(
    positions: list[np.ndarray],
    colors: list[str],
    *,
    t_lo: float,
    t_hi: float,
    n_bins: int,
) -> np.ndarray:
    """Build an ``(n_sources, n_bins, 4)`` RGBA image for the availability strip.

    Args:
        positions: Per-source matplotlib date floats, one 1-D array each.
        colors:    Hex color per source row.
        t_lo:      Left x-limit in matplotlib date units.
        t_hi:      Right x-limit in matplotlib date units.
        n_bins:    Number of uniform UTC bins between ``t_lo`` and ``t_hi``.

    Returns:
        RGBA array suitable for ``Axes.imshow``, shape ``(n_sources, n_bins, 4)``.
    """
    n_sources = len(positions)
    rgba = np.zeros((n_sources, n_bins, 4), dtype=np.float32)
    # Uniform bin edges spanning the padded x-axis limits.
    bin_edges = np.linspace(t_lo, t_hi, n_bins + 1)

    for row_idx, (dates, color) in enumerate(zip(positions, colors, strict=True)):
        # Convert the named source color to RGB channels.
        rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
        # Count snapshots falling in each UTC bin for this source.
        counts, _ = np.histogram(dates, bins=bin_edges)
        occupied = counts > 0
        # Paint occupied bins with the source color and shared alpha.
        rgba[row_idx, occupied, :3] = rgb
        rgba[row_idx, occupied, 3] = _BIN_ALPHA

    return rgba


def plot_source_timeline(
    index_df: pd.DataFrame,
    *,
    title: str = "",
    n_time_bins: int | None = None,
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a horizontal timeline showing when each source has observations.

    Each source occupies one row on the y-axis.  Snapshot times are histogrammed
    into UTC bins and rendered as a rasterized availability strip so large
    indexes do not produce millions of SVG path elements.  Titles, tick labels,
    and axis annotations remain vector text.  Sources are ordered top-to-bottom
    by their first observation time.

    Args:
        index_df:     Assembled dataset index DataFrame.  Must contain columns
                      ``source_name`` and ``time_utc``.
        title:        Optional figure title string.
        n_time_bins:  Number of uniform UTC bins along the x-axis.  ``None``
                      selects ~one bin per day, capped at ``_MAX_TIME_BINS``.
        save_path:    If provided, save the figure as SVG at this path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    # --- Parse timestamps to UTC-aware datetimes --------------------------------
    times = pd.to_datetime(index_df["time_utc"], utc=True)
    df = index_df[["source_name"]].copy()
    df["time"] = times

    # --- Determine source order (earliest first appearance → topmost row) -------
    first_seen = df.groupby("source_name")["time"].min().sort_values()
    sources: list[str] = first_seen.index.tolist()
    n_sources = len(sources)

    # --- Build per-source lists of matplotlib date floats -----------------------
    positions: list[np.ndarray] = []
    for name in sources:
        mask = df["source_name"] == name
        # Convert to matplotlib internal float dates for eventplot compatibility.
        date_floats = mdates.date2num(df.loc[mask, "time"].dt.to_pydatetime())
        positions.append(np.sort(date_floats))

    # --- Create figure and axes -------------------------------------------------
    fig_height = max(_MIN_HEIGHT_IN, _HEIGHT_PER_SOURCE_IN * n_sources)
    fig, ax = plt.subplots(figsize=(COL2, fig_height))

    # --- Assign stable colors via SOURCE_COLORS prefix-matching -----------------
    prop_colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    fallback: cycle[str] = cycle(prop_colors)
    color_cache: dict[str, str] = {}
    colors = [_get_source_color(name, color_cache, fallback) for name in sources]

    # --- Determine padded UTC limits shared by bins and x-axis ------------------
    all_dates = np.concatenate(positions) if positions else np.array([], dtype=float)
    if all_dates.size:
        # 1% padding on each side of the full time span.
        span = all_dates.max() - all_dates.min()
        pad = max(span * 0.01, 1.0)  # at least 1 day of padding
        t_lo = float(all_dates.min() - pad)
        t_hi = float(all_dates.max() + pad)
    else:
        t_lo, t_hi, span = 0.0, 1.0, 1.0

    # --- Build raster availability strip (one row per source) -------------------
    n_bins = _resolve_n_time_bins(span, n_time_bins)
    rgba = _build_timeline_rgba(
        positions,
        colors,
        t_lo=t_lo,
        t_hi=t_hi,
        n_bins=n_bins,
    )
    # Row 0 is the earliest source; align imshow rows with inverted y tick order.
    image = ax.imshow(
        rgba,
        aspect="auto",
        extent=(t_lo, t_hi, n_sources - 0.5, -0.5),
        interpolation="nearest",
        zorder=1,
    )
    # Embed the availability layer as a bitmap inside SVG/PDF exports.
    image.set_rasterized(True)

    # --- Format x-axis as dates -------------------------------------------------
    ax.set_xlim(t_lo, t_hi)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.set_xlabel("Date (UTC)")

    # --- Format y-axis with source name labels ----------------------------------
    ax.set_yticks(range(n_sources))
    ax.set_yticklabels(sources)
    # Invert so the source with the earliest data appears at the top.
    ax.invert_yaxis()

    # --- Subtle vertical grid lines on x-axis only ------------------------------
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.25, zorder=0)
    ax.yaxis.grid(False)

    # --- Title and tight layout -------------------------------------------------
    if title:
        ax.set_title(title)
    fig.tight_layout()

    # --- Optional save ----------------------------------------------------------
    if save_path is not None:
        save_fig(fig, save_path)

    return fig, ax
