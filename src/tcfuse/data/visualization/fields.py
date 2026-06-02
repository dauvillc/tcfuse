"""2D satellite / model field visualization for tcfuse.

Provides generic and source-specific functions for displaying gridded
field data (PMW, IR, ERA5, SAR, etc.) over geographic maps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from tcfuse.data.visualization.style import (
    AR_GOLDEN,
    COL1,
    COL2,
    UNIT_M_S,
    format_text_for_renderer,
    get_cmap,
    save_fig,
    setup_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tcfuse.data.sources import Source

setup_style()


class ChannelPlotSpec(NamedTuple):
    """Per-channel colormap and labeling for multi-panel field plots."""

    cmap_key: str
    unit: str
    title: str | None = None


def _panel_grid(n_channels: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a channel-count-appropriate subplot layout."""
    if n_channels <= 1:
        return 1, 1
    if n_channels == 3:
        return 1, 3
    ncols = 2
    nrows = math.ceil(n_channels / ncols)
    return nrows, ncols


def _field_coords_arrays(source: Source) -> tuple[np.ndarray, np.ndarray]:
    """Extract 2D lat/lon grids from a FIELD source, shape (H, W) each."""
    # Visualization utilities operate on single snapshots only.
    if source.batched:
        raise ValueError(
            f"Visualization expects non-batched FIELD sources, got batched=True for {source.source_name}."
        )
    coords = source.coords.detach().cpu().numpy()
    lats = coords[..., 1]
    lons = coords[..., 2]
    return lats, lons


def _field_channel_values(source: Source, channel_idx: int) -> np.ndarray:
    """Extract one 2D channel from a FIELD source with mask applied, shape (H, W)."""
    # Visualization utilities operate on single snapshots only.
    if source.batched:
        raise ValueError(
            f"Visualization expects non-batched FIELD sources, got batched=True for {source.source_name}."
        )
    values = source.values.detach().cpu().numpy()[..., channel_idx]
    if source.mask is not None:
        mask_np = source.mask.detach().cpu().numpy()
        if mask_np.ndim == 3:
            valid = mask_np[..., channel_idx]
        else:
            valid = mask_np
        values = np.where(valid, values, np.nan)
    return values


def _storm_center_from_coords(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Return lat/lon at the grid center pixel (storm-centered grids)."""
    h, w = lats.shape[:2]
    return float(lats[h // 2, w // 2]), float(lons[h // 2, w // 2])


def plot_field(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    ax: Axes | None = None,
    *,
    channel: str = "wind",
    unit: str = "",
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single 2D field channel over a geographic domain.

    Args:
        values:    2D array of field values, shape (H, W). NaN where invalid.
        lats:      2D array of pixel latitudes, shape (H, W).
        lons:      2D array of pixel longitudes, shape (H, W).
        ax:        Existing GeoAxes to draw into; a new figure is created when None.
        channel:   Colormap key passed to get_cmap() (e.g. "sar_wind", "tb", "wind").
        unit:      Physical unit string for the colorbar label.
        storm_lat: Optional storm-center latitude for a crosshair marker.
        storm_lon: Optional storm-center longitude for a crosshair marker.
        title:     Optional figure title.
        save_path: If provided, save the figure here (SVG/PDF). The field layer is
                   rasterized on export so vector output stays small while text
                   and axes remain vector.

    Returns:
        (fig, ax) tuple.
    """
    # Create figure and geo-axes when no axes are provided
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(COL1, COL1),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = ax.get_figure()

    # Draw the field with the physical-quantity colormap
    im = ax.pcolormesh(
        lons,
        lats,
        values,
        cmap=get_cmap(channel),
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    # Embed the colormap as a bitmap inside SVG/PDF; keep titles and axes as vectors.
    im.set_rasterized(True)

    # Add coastlines for geographic context
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)

    # Colorbar below the axes with physical unit label
    channel_label = format_text_for_renderer(channel)
    label = f"{channel_label} ({unit})" if unit else channel_label
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label(label)

    # Storm-center crosshair (optional)
    if storm_lat is not None and storm_lon is not None:
        ax.plot(
            storm_lon,
            storm_lat,
            "+",
            color="white",
            markersize=6,
            markeredgewidth=1.0,
            transform=ccrs.PlateCarree(),
        )

    # Set domain from the field bounding box
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    if title:
        ax.set_title(format_text_for_renderer(title))
    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_field_from_source(
    source: Source,
    channel_idx: int,
    ax: Axes | None = None,
    *,
    cmap_key: str = "wind",
    unit: str = "",
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one channel of a FIELD Source over a geographic map.

    Args:
        source:      FIELD Source; values (H, W, C), coords (H, W, 3).
        channel_idx: Index into the channel axis of ``source.values``.
        ax:          Existing GeoAxes; a new figure is created when None.
        cmap_key:    Colormap key for :func:`get_cmap`.
        unit:        Physical unit for the colorbar.
        storm_lat:   Optional storm-center latitude for a crosshair.
        storm_lon:   Optional storm-center longitude for a crosshair.
        title:       Panel title; defaults to the channel name when empty.
        save_path:   If provided, save the figure here (SVG).

    Returns:
        (fig, ax) tuple.
    """
    lats, lons = _field_coords_arrays(source)
    values = _field_channel_values(source, channel_idx)

    channel_name = source.channels[channel_idx]
    plot_title = title if title else format_text_for_renderer(channel_name)

    return plot_field(
        values,
        lats,
        lons,
        ax,
        channel=cmap_key,
        unit=unit,
        storm_lat=storm_lat,
        storm_lon=storm_lon,
        title=plot_title,
        save_path=save_path,
    )


def plot_field_source_channels(
    source: Source,
    channel_specs: list[ChannelPlotSpec],
    *,
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    suptitle: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot every channel of a FIELD Source in a multi-panel geographic layout.

    Args:
        source:         FIELD Source with ``len(channel_specs)`` channels.
        channel_specs:  One spec per channel (cmap, unit, optional title).
        storm_lat:      Storm-center latitude; when None and ``storm_lon`` is None,
                        uses the grid-center pixel from ``source.coords``.
        storm_lon:      Storm-center longitude (paired with ``storm_lat``).
        suptitle:       Optional figure suptitle (source name and time).
        save_path:      If provided, save the combined figure here (SVG).

    Returns:
        (fig, list of GeoAxes) tuple.
    """
    n_channels = len(channel_specs)
    if n_channels != len(source.channels):
        raise ValueError(
            f"channel_specs length {n_channels} != source channels {len(source.channels)}"
        )

    lats, lons = _field_coords_arrays(source)
    if storm_lat is None and storm_lon is None:
        storm_lat, storm_lon = _storm_center_from_coords(lats, lons)

    nrows, ncols = _panel_grid(n_channels)
    fig_width = COL2 if n_channels > 1 else COL1
    fig_height = fig_width * AR_GOLDEN * nrows / max(ncols, 1)

    fig, axes_arr = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    axes_flat: list[Axes] = list(axes_arr.ravel())

    # Draw each channel into its subplot
    for idx, spec in enumerate(channel_specs):
        raw_title = spec.title if spec.title is not None else source.channels[idx]
        panel_title = format_text_for_renderer(raw_title)
        plot_field_from_source(
            source,
            idx,
            axes_flat[idx],
            cmap_key=spec.cmap_key,
            unit=spec.unit,
            storm_lat=storm_lat,
            storm_lon=storm_lon,
            title=panel_title,
        )

    # Hide unused subplot slots when the grid is larger than n_channels
    for ax in axes_flat[n_channels:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(format_text_for_renderer(suptitle))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, axes_flat[:n_channels]


def plot_sar_wind(
    source: Source,
    ax: Axes | None = None,
    *,
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    title: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a SAR C-band wind speed field from a Source object.

    Convenience wrapper around :func:`plot_field_from_source` for single-channel
    SAR wind snapshots.

    Args:
        source:    Source object with kind=FIELD, channels=["wind_speed"].
                   values shape: (H, W, 1); coords shape: (H, W, 3) — [time, lat, lon].
        ax:        Existing GeoAxes to draw into; a new figure is created when None.
        storm_lat: Optional storm-center latitude for a crosshair marker.
        storm_lon: Optional storm-center longitude for a crosshair marker.
        title:     Optional figure title; defaults to source_name when empty.
        save_path: If provided, save the figure here (SVG).

    Returns:
        (fig, ax) tuple.
    """
    plot_title = title if title else format_text_for_renderer(source.source_name)
    return plot_field_from_source(
        source,
        0,
        ax,
        cmap_key="sar_wind",
        unit=UNIT_M_S,
        storm_lat=storm_lat,
        storm_lon=storm_lon,
        title=plot_title,
        save_path=save_path,
    )
