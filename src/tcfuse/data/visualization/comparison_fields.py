"""Multi-model Target | Prediction | Diff comparison for reconstructed FIELD sources.

Generalizes ``training.plot_field_reconstruction`` (a single-model Target |
Prediction | Error triptych) to an arbitrary number of models, for use by the
offline evaluation suite (``tcfuse.evaluation``) when comparing several
models' saved predictions against the same ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from tcfuse.data.visualization.style import AR_GOLDEN, COL2, get_cmap, save_fig, setup_style
from tcfuse.data.visualization.training import draw_mesh_panel, masked_channel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

setup_style()

# Reference panel width/height (inches), matching training.plot_field_reconstruction's
# per-panel proportions (a COL2-wide, 3-column figure) so single- and multi-model
# figures look consistent.
_PANEL_WIDTH = COL2 / 3
_PANEL_HEIGHT = COL2 * AR_GOLDEN / 3


def plot_field_prediction_comparison(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
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
    """Plot Target | Pred | Diff panels per model, for every channel of a FIELD source.

    Each channel becomes one row of ``1 + 2 * len(predictions)`` geographic
    panels: the target, then one (prediction, difference) pair per model, in
    ``predictions`` order. Target and every prediction panel share one
    per-channel color scale; every difference panel (prediction minus target)
    shares one symmetric diverging scale so error magnitudes are directly
    comparable across models.

    Args:
        target:      Ground-truth field values, shape (H, W, C), physical units.
        predictions: Model name -> reconstructed field values, shape (H, W, C),
                     physical units. Column order follows dict order.
        lats:        Pixel latitudes, shape (H, W).
        lons:        Pixel longitudes, shape (H, W).
        channels:    Channel names, length C (used for row titles).
        cmap_key:    Colormap key for target/prediction panels, via get_cmap().
        unit:        Physical unit string for the colorbar labels.
        mask:        Optional availability mask, shape (H, W, C); invalid
                     pixels are NaN-ed out before plotting.
        suptitle:    Optional figure suptitle (e.g. source name + sample id).
        save_path:   If provided, save the figure here (SVG).

    Returns:
        (fig, list of Axes) tuple; the axes list is row-major (channel, column).
    """
    # One row per channel; one target column plus a (pred, diff) pair per model.
    n_channels = target.shape[-1]
    model_names = list(predictions.keys())
    n_cols = 1 + 2 * len(model_names)

    fig, axes_arr = plt.subplots(
        n_channels,
        n_cols,
        figsize=(_PANEL_WIDTH * n_cols, _PANEL_HEIGHT * n_channels),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    # Diverging colormap for the signed diff columns; shared field colormap for
    # target/prediction columns.
    diff_cmap = get_cmap("anomaly")
    field_cmap = get_cmap(cmap_key)
    axes_flat: list[Axes] = []

    # Draw one row of panels per channel.
    for ch_idx in range(n_channels):
        ch_name = channels[ch_idx]
        # Masked target channel and its finite-value color scale, shared by
        # the target and every prediction panel in this row.
        tgt = masked_channel(target, mask, ch_idx)
        finite = np.isfinite(tgt)
        if finite.any():
            vmin = float(np.nanmin(tgt))
            vmax = float(np.nanmax(tgt))
        else:
            vmin, vmax = None, None

        # Masked prediction channels and their signed errors, one per model.
        preds = {name: masked_channel(pred, mask, ch_idx) for name, pred in predictions.items()}
        errors = {name: pred - tgt for name, pred in preds.items()}
        # Symmetric diff scale shared across all models so magnitudes compare
        # directly; computed from the largest finite absolute error.
        finite_abs_errors = [
            float(np.nanmax(np.abs(error))) for error in errors.values() if np.isfinite(error).any()
        ]
        err_lim = max(finite_abs_errors) if finite_abs_errors else None

        # Column headers (model identity) only belong on the top row; row labels
        # (channel identity) only belong on the leftmost column. This factors
        # each piece of information out to a single panel instead of repeating
        # it in every one of the row's ``n_cols`` titles.
        is_top_row = ch_idx == 0

        # Target panel (column 0).
        ax_tgt = axes_arr[ch_idx, 0]
        draw_mesh_panel(
            fig,
            ax_tgt,
            tgt,
            lats,
            lons,
            cmap=field_cmap,
            vmin=vmin,
            vmax=vmax,
            label=unit or ch_name,
            title="Target" if is_top_row else "",
            row_label=ch_name,
        )
        axes_flat.append(ax_tgt)

        # One (prediction, diff) column pair per model, in declaration order.
        for model_idx, model_name in enumerate(model_names):
            pred_col = 1 + 2 * model_idx
            diff_col = pred_col + 1

            ax_pred = axes_arr[ch_idx, pred_col]
            draw_mesh_panel(
                fig,
                ax_pred,
                preds[model_name],
                lats,
                lons,
                cmap=field_cmap,
                vmin=vmin,
                vmax=vmax,
                label=unit or ch_name,
                title=f"{model_name}\n(prediction)" if is_top_row else "",
            )
            axes_flat.append(ax_pred)

            ax_diff = axes_arr[ch_idx, diff_col]
            draw_mesh_panel(
                fig,
                ax_diff,
                errors[model_name],
                lats,
                lons,
                cmap=diff_cmap,
                vmin=-err_lim if err_lim is not None else None,
                vmax=err_lim,
                label=unit or "diff",
                title=f"{model_name}\n(diff)" if is_top_row else "",
            )
            axes_flat.append(ax_diff)

    # Title and layout; leave headroom for the suptitle when present.
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    return fig, axes_flat
