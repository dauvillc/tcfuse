"""Model-diagnostic visualizations for tcfuse training and validation.

Currently provides a Target | Prediction | Error comparison for reconstructed
FIELD sources, used to spot-check a few validation samples each epoch. Inputs
are plain numpy arrays so the function stays dataset-agnostic and unit-testable
with synthetic data, and can be reused from an offline inference viz script.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap

from tcfuse.data.visualization.style import (
    AR_GOLDEN,
    COL2,
    UNIT_K,
    UNIT_M_S,
    get_cmap,
    save_fig,
    setup_style,
)

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

setup_style()

# Number of columns in the comparison grid: target, prediction, error.
_N_COLS = 3


def _masked_channel(values: np.ndarray, mask: np.ndarray | None, channel_idx: int) -> np.ndarray:
    """Return one 2D channel with invalid pixels set to NaN, shape (H, W)."""
    # Slice the requested channel from the (H, W, C) array.
    channel = values[..., channel_idx]
    # Without a mask, every pixel is treated as valid.
    if mask is None:
        return channel
    # mask is (H, W, C); NaN-out pixels flagged unavailable for this channel.
    valid = mask[..., channel_idx]
    return np.where(valid, channel, np.nan)


def _draw_mesh(
    fig: Figure,
    ax: GeoAxes,
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    cmap: Colormap | str,
    vmin: float | None,
    vmax: float | None,
    label: str,
    title: str,
) -> None:
    """Draw a single rasterized pcolormesh panel with coastlines and a colorbar."""
    # Plot the field; explicit vmin/vmax lets target and prediction share a scale.
    im = ax.pcolormesh(
        lons,
        lats,
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    # Rasterize the mesh layer so SVG/PDF output stays small; text stays vector.
    im.set_rasterized(True)
    # Light coastlines for geographic context.
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    # Horizontal colorbar below the panel with the physical-quantity label.
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label(label)
    # Frame the domain to the field bounding box.
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    ax.set_title(title)


def plot_field_reconstruction(
    target: np.ndarray,
    prediction: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    channels: list[str],
    cmap_key: str = "tb",
    unit: str = "",
    mask: np.ndarray | None = None,
    suptitle: str = "",
    save_path: Path | str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot Target | Prediction | Error panels for every channel of a FIELD source.

    Each channel becomes one row of three geographic panels. The target and
    prediction columns share a per-channel color scale (so differences are
    visible); the error column (prediction minus target) uses a symmetric
    diverging scale centered on zero.

    Args:
        target:     Ground-truth field values, shape (H, W, C), physical units.
        prediction: Reconstructed field values, shape (H, W, C), physical units.
        lats:       Pixel latitudes, shape (H, W).
        lons:       Pixel longitudes, shape (H, W).
        channels:   Channel names, length C (used for row titles).
        cmap_key:   Colormap key for target/prediction passed to get_cmap().
        unit:       Physical unit string for the colorbar labels.
        mask:       Optional availability mask, shape (H, W, C); invalid pixels
                    are NaN-ed out before plotting.
        suptitle:   Optional figure suptitle (e.g. source name + sample id).
        save_path:  If provided, save the figure here (SVG).

    Returns:
        (fig, list of Axes) tuple; the axes list is row-major (channel, column).
    """
    # One row per channel; three columns (target, prediction, error).
    n_channels = target.shape[-1]
    # Height scales with the channel count; width is the full double-column figure.
    fig_height = COL2 * AR_GOLDEN * n_channels / _N_COLS

    fig, axes_arr = plt.subplots(
        n_channels,
        _N_COLS,
        figsize=(COL2, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    # Diverging colormap for the signed error column.
    error_cmap = get_cmap("anomaly")
    field_cmap = get_cmap(cmap_key)
    axes_flat: list[Axes] = []

    # Draw one row of three panels per channel.
    for ch_idx in range(n_channels):
        # Pull masked target / prediction channels (NaN where unavailable).
        tgt = _masked_channel(target, mask, ch_idx)
        pred = _masked_channel(prediction, mask, ch_idx)
        # Shared color scale for target and prediction from finite target pixels.
        finite = np.isfinite(tgt)
        if finite.any():
            vmin = float(np.nanmin(tgt))
            vmax = float(np.nanmax(tgt))
        else:
            vmin, vmax = None, None
        # Signed error and a symmetric range so zero maps to the colormap center.
        error = pred - tgt
        err_abs = float(np.nanmax(np.abs(error))) if np.isfinite(error).any() else 0.0
        err_lim = err_abs if err_abs > 0 else None

        ch_name = channels[ch_idx]
        # Target panel.
        ax_tgt = axes_arr[ch_idx, 0]
        _draw_mesh(
            fig,
            ax_tgt,
            tgt,
            lats,
            lons,
            cmap=field_cmap,
            vmin=vmin,
            vmax=vmax,
            label=f"{ch_name} ({unit})" if unit else ch_name,
            title=f"{ch_name} — target",
        )
        # Prediction panel (same scale as target).
        ax_pred = axes_arr[ch_idx, 1]
        _draw_mesh(
            fig,
            ax_pred,
            pred,
            lats,
            lons,
            cmap=field_cmap,
            vmin=vmin,
            vmax=vmax,
            label=f"{ch_name} ({unit})" if unit else ch_name,
            title=f"{ch_name} — prediction",
        )
        # Error panel (symmetric diverging scale).
        ax_err = axes_arr[ch_idx, 2]
        _draw_mesh(
            fig,
            ax_err,
            error,
            lats,
            lons,
            cmap=error_cmap,
            vmin=-err_lim if err_lim is not None else None,
            vmax=err_lim,
            label=f"error ({unit})" if unit else "error",
            title=f"{ch_name} — error",
        )
        axes_flat.extend([ax_tgt, ax_pred, ax_err])

    # Title and layout; leave headroom for the suptitle when present.
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, axes_flat


def _field_display(source_name: str) -> tuple[str, str]:
    """Pick a colormap key and physical unit from a FIELD source's name.

    A small heuristic so reconstruction figures use a sensible colormap/unit
    without per-source config; defaults to brightness-temperature styling.

    Args:
        source_name: Source identifier, e.g. ``"pmw_gmi"`` or ``"sar"``.

    Returns:
        ``(cmap_key, unit)`` for :func:`plot_field_reconstruction`.
    """
    name = source_name.lower()
    # Surface-wind sources (SAR, ERA5 wind) read best with the speed colormap.
    if "sar" in name or "wind" in name:
        return "wind", UNIT_M_S
    # Default: PMW / IR brightness temperature.
    return "tb", UNIT_K


def _figure_to_rgb(fig: Figure) -> np.ndarray:
    """Rasterize a figure to a fixed-size RGB array, shape (H, W, 3).

    Logging several images under one W&B key requires identical pixel sizes.
    Saving a figure with the project's ``savefig.bbox="tight"`` rcParam crops to
    each figure's content and yields per-sample sizes (the suptitle text length
    varies). Drawing the Agg canvas instead gives a deterministic
    ``figsize * figure.dpi`` raster.

    Args:
        fig: The figure to rasterize (already fully drawn/laid out).

    Returns:
        Contiguous uint8 RGB array sized ``figsize * figure.dpi`` (constant for a
        given channel count).
    """
    # Draw on an explicit Agg canvas so the raster size is figsize * figure.dpi,
    # independent of content (no tight-bbox cropping).
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # buffer_rgba gives an (H, W, 4) uint8 view; drop alpha and copy out.
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[..., :3].copy()


def render_field_reconstruction(
    target: np.ndarray,
    prediction: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    channels: list[str],
    source_name: str,
    save_path: Path | str,
    mask: np.ndarray | None = None,
    suptitle: str = "",
) -> np.ndarray:
    """Build a reconstruction comparison figure, save it, and return a raster.

    Convenience wrapper for callers (e.g. the validation loop) that need both an
    SVG on disk and a fixed-size pixel array to hand to an image logger, without
    importing matplotlib themselves. Colormap and unit are derived from
    ``source_name`` via :func:`_field_display`.

    Args:
        target:      Ground-truth field values, shape (H, W, C), physical units.
        prediction:  Reconstructed field values, shape (H, W, C), physical units.
        lats:        Pixel latitudes, shape (H, W).
        lons:        Pixel longitudes, shape (H, W).
        channels:    Channel names, length C.
        source_name: FIELD source identifier (drives colormap/unit).
        save_path:   Destination for the SVG (extension forced to .svg).
        mask:        Optional availability mask, shape (H, W, C).
        suptitle:    Optional figure suptitle.

    Returns:
        Fixed-size uint8 RGB array of the rendered figure, shape (H, W, 3).
    """
    # Choose a colormap/unit appropriate for this source.
    cmap_key, unit = _field_display(source_name)
    # Build the Target | Prediction | Error figure and write the SVG.
    fig, _axes = plot_field_reconstruction(
        target,
        prediction,
        lats,
        lons,
        channels=channels,
        cmap_key=cmap_key,
        unit=unit,
        mask=mask,
        suptitle=suptitle,
        save_path=save_path,
    )
    # Rasterize at a deterministic size, then release the figure.
    rgb = _figure_to_rgb(fig)
    plt.close(fig)
    return rgb
