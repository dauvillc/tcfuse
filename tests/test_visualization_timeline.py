"""Unit tests for source availability timeline plots (synthetic index only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage

from tcfuse.data.visualization.timeline import plot_source_timeline


def _make_index(n_per_source: int) -> pd.DataFrame:
    """Build a tiny assembled-index-like DataFrame for timeline plotting."""
    rows: list[dict[str, object]] = []
    for source_name, start in [("pmw_a", "2020-01-01"), ("ir_b", "2021-06-01")]:
        times = pd.date_range(start, periods=n_per_source, freq="6h", tz="UTC")
        for time_utc in times:
            rows.append({"source_name": source_name, "time_utc": time_utc})
    return pd.DataFrame(rows)


def test_plot_source_timeline_rasterizes_availability_layer() -> None:
    """Large synthetic indexes use a rasterized imshow strip instead of event ticks."""
    index_df = _make_index(n_per_source=500)

    fig, ax = plot_source_timeline(index_df, title="test timeline")

    images = [artist for artist in ax.get_children() if isinstance(artist, AxesImage)]
    assert len(images) == 1
    assert images[0].get_rasterized() is True
    plt.close(fig)


def test_plot_source_timeline_respects_explicit_bin_count() -> None:
    """Caller-provided ``n_time_bins`` controls the raster strip width."""
    index_df = _make_index(n_per_source=20)

    fig, ax = plot_source_timeline(index_df, n_time_bins=120)

    image = next(artist for artist in ax.get_children() if isinstance(artist, AxesImage))
    rgba = np.asarray(image.get_array())
    assert rgba.shape == (2, 120, 4)
    plt.close(fig)
