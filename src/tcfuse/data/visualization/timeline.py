"""Source availability timeline: one horizontal event row per observation source.

Shows when each source in the assembled dataset index has observations, with
snapshot times rendered as thin tick marks on a common UTC time axis.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

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
# Tick mark height in axes-fraction units, passed to eventplot as linelengths.
_TICK_LENGTH = 0.6


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


def plot_source_timeline(
    index_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a horizontal event timeline showing when each source has observations.

    Each source occupies one row on the y-axis.  Each snapshot in the index is
    drawn as a thin vertical tick mark at the corresponding UTC time.  Sources
    are ordered top-to-bottom by their first observation time.

    Args:
        index_df:  Assembled dataset index DataFrame.  Must contain columns
                   ``source_name`` and ``time_utc``.
        title:     Optional figure title string.
        save_path: If provided, save the figure as SVG at this path.

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

    # --- Draw event ticks with eventplot ----------------------------------------
    # y-positions are integers 0..n_sources-1; reversed so oldest source sits at top.
    ax.eventplot(
        positions,
        orientation="horizontal",
        lineoffsets=list(range(n_sources)),
        linelengths=_TICK_LENGTH,
        linewidths=0.6,
        colors=colors,
        alpha=0.7,
    )

    # --- Format x-axis as dates -------------------------------------------------
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

    # --- Axis limits ------------------------------------------------------------
    all_dates = np.concatenate(positions)
    if all_dates.size:
        # 1% padding on each side of the full time span.
        span = all_dates.max() - all_dates.min()
        pad = max(span * 0.01, 1.0)  # at least 1 day of padding
        ax.set_xlim(all_dates.min() - pad, all_dates.max() + pad)

    # --- Title and tight layout -------------------------------------------------
    if title:
        ax.set_title(title)
    fig.tight_layout()

    # --- Optional save ----------------------------------------------------------
    if save_path is not None:
        save_fig(fig, save_path)

    return fig, ax
