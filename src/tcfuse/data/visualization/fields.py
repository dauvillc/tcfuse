"""2D satellite / model field visualization for tcfuse.

Provides generic and source-specific functions for displaying gridded
field data (PMW, IR, ERA5, SAR, etc.) over geographic maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from tcfuse.data.visualization.style import COL1, get_cmap, save_fig, setup_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

setup_style()


def plot_field(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    ax: "Axes | None" = None,
    *,
    channel: str = "wind",
    unit: str = "",
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    title: str = "",
    save_path: "Path | str | None" = None,
) -> tuple["Figure", "Axes"]:
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
        save_path: If provided, save the figure here (SVG).

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

    # Add coastlines for geographic context
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)

    # Colorbar below the axes with physical unit label
    label = f"{channel} ({unit})" if unit else channel
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
        ax.set_title(title)
    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_sar_wind(
    source: "Source",  # type: ignore[name-defined]  # noqa: F821
    ax: "Axes | None" = None,
    *,
    storm_lat: float | None = None,
    storm_lon: float | None = None,
    title: str = "",
    save_path: "Path | str | None" = None,
) -> tuple["Figure", "Axes"]:
    """Plot a SAR C-band wind speed field from a Source object.

    Convenience wrapper around plot_field() that extracts arrays from the
    Source dataclass and applies the validity mask before plotting.

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
    # Extract arrays from Source tensors; move to CPU if needed
    # values: (H, W, 1) → (H, W)
    values = source.values.detach().cpu().numpy()[..., 0]
    # coords: (H, W, 3) — channels are [time, lat, lon]
    coords = source.coords.detach().cpu().numpy()
    lats = coords[..., 1]  # (H, W)
    lons = coords[..., 2]  # (H, W)

    # Apply validity mask: invalid pixels → NaN so pcolormesh skips them
    if source.mask is not None:
        mask = source.mask.detach().cpu().numpy()  # (H, W), True = valid
        values = np.where(mask, values, np.nan)

    # Use source_name as default title
    plot_title = title if title else source.source_name

    return plot_field(
        values,
        lats,
        lons,
        ax,
        channel="sar_wind",
        unit="m s⁻¹",
        storm_lat=storm_lat,
        storm_lon=storm_lon,
        title=plot_title,
        save_path=save_path,
    )
