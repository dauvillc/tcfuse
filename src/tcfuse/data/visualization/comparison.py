"""Model-comparison figures for the offline evaluation suite.

These plots compare several models side by side over the metrics computed by the
evaluation plugins (see :mod:`tcfuse.evaluation`).  The flagship figure is a
grouped bar chart: one bar group per channel, one coloured bar per model, for a
single ``(metric, source)`` pair — the kind of comparison that goes straight into
a paper.  Keeping all matplotlib here (and not in the plugins) lets future
comparison plugins reuse the same drawing code.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tcfuse.data.visualization.style import AR_GOLDEN, COL1, COL2, save_fig, setup_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

# Apply the shared publication style once at import time.
setup_style()

# Above this many channels the single-column width gets too cramped for grouped
# bars, so we switch to the double-column width.
_WIDE_CHANNEL_THRESHOLD = 4


def plot_metric_comparison(
    df: pd.DataFrame,
    *,
    metric: str,
    source_name: str,
    ax: Axes | None = None,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar chart comparing models on one metric for one source.

    Args:
        df: Tidy rows for a single ``(metric, source_name)`` pair, with columns
            ``model``, ``channel`` and ``value``. Model order is taken from the
            order models first appear in the ``model`` column (i.e. config order).
        metric: Metric name, used for the y-axis label.
        source_name: Source name, used for the default title.
        ax: Axes to draw into; a new figure is created when None.
        title: Optional figure title (defaults to ``"<source_name> — <metric>"``).
        save_path: If provided, save the figure here (SVG).

    Returns:
        (fig, ax) tuple.
    """
    # Preserve the model and channel order as first encountered so the legend and
    # x-axis match the config declaration order rather than an alphabetical sort.
    models = list(dict.fromkeys(df["model"]))
    channels = list(dict.fromkeys(df["channel"]))

    # Pivot to a (channel, model) value lookup for easy per-bar indexing; missing
    # (channel, model) combinations become NaN and are simply not drawn.
    table = df.pivot_table(index="channel", columns="model", values="value")

    # Pick a width that stays legible as the number of channels grows.
    width = COL2 if len(channels) > _WIDE_CHANNEL_THRESHOLD else COL1
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, width * AR_GOLDEN))
    else:
        fig = cast(Figure, ax.get_figure())

    # Lay out one group of bars per channel; each model gets an evenly spaced slot
    # within the group, colours coming from the default categorical cycle.
    group_centers = np.arange(len(channels))
    bar_width = 0.8 / max(len(models), 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model_index, model in enumerate(models):
        # Offset this model's bars from the group center so they sit side by side.
        offset = (model_index - (len(models) - 1) / 2) * bar_width
        values = [
            table.at[channel, model] if model in table.columns else np.nan for channel in channels
        ]
        ax.bar(
            group_centers + offset,
            values,
            width=bar_width,
            label=model,
            color=colors[model_index % len(colors)],
        )

    # Label the channel groups and the metric axis.
    ax.set_xticks(group_centers)
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{source_name} — {metric}")
    ax.legend(title="model", frameon=False)

    # Optionally persist the figure as SVG.
    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax
