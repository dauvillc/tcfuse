"""StormDataVisualizer: geographic and temporal overview plots for a StormData object."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from pathlib import Path
from typing import Literal, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import rcParams
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull, QhullError

from tcfuse.data.sources.source import Source, SourceKind
from tcfuse.data.sources.storm_data import StormData
from tcfuse.data.visualization.style import AR_GOLDEN, COL2, SOURCE_COLORS, save_fig, setup_style

# Discriminated union for the three footprint shapes returned by _footprint_from_source.
_Footprint = (
    tuple[Literal["point"], float, float]
    | tuple[Literal["hull"], np.ndarray, np.ndarray]
    | tuple[Literal["empty"]]
)

# Apply publication-quality rcParams at import time (project-wide convention).
setup_style()


def _get_source_color(
    source_name: str,
    color_cache: dict[str, str],
    fallback_cycle: Iterator[str],
) -> str:
    """Return a stable color for a source name, cached across calls.

    Args:
        source_name:    Source identifier, e.g. ``"pmw_amsr2_gcomw1"``.
        color_cache:    Mutable dict accumulating already-assigned colors.
        fallback_cycle: Infinite iterator over matplotlib prop_cycle colors.

    Returns:
        Hex color string.
    """
    # Return cached assignment if already seen.
    if source_name in color_cache:
        return color_cache[source_name]

    # Substring-match against known SOURCE_COLORS keys (longest key first for determinism).
    matched = next(
        (v for k, v in SOURCE_COLORS.items() if k in source_name),
        None,
    )
    color = matched if matched is not None else next(fallback_cycle)
    color_cache[source_name] = color
    return color


def _footprint_from_source(
    source: Source,
) -> _Footprint:
    """Extract the geographic footprint of a single Source snapshot.

    Returns a discriminated tuple describing the footprint type:

    - ``('point', lon, lat)``   — for SCALAR / PROFILE, or a FIELD with ≤2 valid pixels
    - ``('hull', hull_lons, hull_lats)``  — convex hull arrays for a FIELD with ≥3 valid pixels
    - ``('empty',)``            — FIELD with no valid pixels

    Args:
        source: The source whose footprint to compute.

    Returns:
        Discriminated tuple as described above.
    """
    coords_np = source.coords.detach().cpu().numpy()

    # --- SCALAR: single (time, lat, lon) coordinate ---
    if source.kind == SourceKind.SCALAR:
        lat = float(coords_np[1])
        lon = float(coords_np[2])
        return ("point", lon, lat)

    # --- PROFILE: (L, 4) coords — use mean lat/lon across levels ---
    if source.kind == SourceKind.PROFILE:
        lat = float(coords_np[:, 1].mean())
        lon = float(coords_np[:, 2].mean())
        return ("point", lon, lat)

    # --- FIELD: (H, W, 3) coords — convex hull of non-NaN pixels ---
    values_np = source.values.detach().cpu().numpy()  # (H, W, C)

    # A pixel is valid if any of its C channels is finite (not NaN).
    valid = np.any(np.isfinite(values_np), axis=-1)  # (H, W) bool

    # Combine with the explicit mask when present (True = valid).
    if source.mask is not None:
        mask_np = source.mask.detach().cpu().numpy()
        # Collapse channel dimension if mask has shape (H, W, C).
        if mask_np.ndim == 3:
            mask_np = np.all(mask_np, axis=-1)  # (H, W)
        valid = valid & mask_np

    lats = coords_np[valid, 1]  # (N,)
    lons = coords_np[valid, 2]  # (N,)

    # No valid pixels — skip this source entirely.
    if len(lats) == 0:
        return ("empty",)

    # Too few points for a meaningful convex hull — fall back to centroid.
    if len(lats) < 3:
        return ("point", float(lons.mean()), float(lats.mean()))

    # Compute convex hull; fall back to centroid if points are collinear.
    pts = np.column_stack([lons, lats])  # (N, 2) in (x=lon, y=lat) order
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return ("point", float(lons.mean()), float(lats.mean()))

    hull_pts = pts[hull.vertices]  # (V, 2)
    return ("hull", hull_pts[:, 0], hull_pts[:, 1])


class StormDataVisualizer:
    """Visualization helpers for a single :class:`~tcfuse.data.sources.storm_data.StormData`.

    Args:
        storm_data: The assembled multi-source storm data to visualize.
    """

    def __init__(self, storm_data: StormData) -> None:
        self._storm_data = storm_data

    def show_footprints(
        self,
        ax: GeoAxes | None = None,
        *,
        title: str = "",
        save_path: Path | str | None = None,
    ) -> tuple[Figure, GeoAxes]:
        """Display the geographic footprint of every source snapshot on a world map.

        For SCALAR and PROFILE sources the footprint is a single point (lat, lon).
        For FIELD sources the footprint is the convex hull of non-NaN pixels; a pixel
        counts as non-NaN when at least one of its channel values is finite.
        Each footprint is annotated with the source name and its UTC snapshot time.

        Args:
            ax:         Pre-existing :class:`cartopy.mpl.geoaxes.GeoAxes` in PlateCarree
                        projection. A new figure is created when ``None``.
            title:      Optional figure title; defaults to ``"Footprints — {storm_id}"``.
            save_path:  If provided, save the figure as SVG via :func:`save_fig`.

        Returns:
            ``(fig, ax)`` tuple — the caller may adjust further before displaying.
        """
        # Create a new figure with a global PlateCarree map if no axes provided.
        if ax is None:
            _fig, _ax = plt.subplots(
                1,
                1,
                figsize=(COL2, COL2 * AR_GOLDEN),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            fig: Figure = _fig
            ax = cast(GeoAxes, _ax)
        else:
            if not isinstance(ax, GeoAxes):
                raise TypeError(f"ax must be a cartopy GeoAxes, got {type(ax)}")
            fig_or_none = ax.get_figure()
            if fig_or_none is None:
                raise RuntimeError("ax has no associated figure")
            # get_figure() returns Figure | SubFigure; a top-level axes is always on a Figure.
            fig = cast(Figure, fig_or_none)

        # --- Compute geographic bounding box across all sources ---
        all_lats: list[float] = []
        all_lons: list[float] = []
        for source in self._storm_data.sources.values():
            fp = _footprint_from_source(source)
            if fp[0] == "empty":
                continue
            if fp[0] == "point":
                _, fp_lon, fp_lat = fp
                all_lons.append(float(fp_lon))
                all_lats.append(float(fp_lat))
            else:  # 'hull'
                _, hull_lons, hull_lats = fp
                all_lons.extend(hull_lons.tolist())
                all_lats.extend(hull_lats.tolist())

        # Apply 5% margin and clamp to valid lat/lon ranges.
        ext_lat_min = ext_lat_max = ext_lon_min = ext_lon_max = 0.0
        if all_lats and all_lons:
            lat_min, lat_max = min(all_lats), max(all_lats)
            lon_min, lon_max = min(all_lons), max(all_lons)
            # Enforce a minimum 1° span so a single-point storm still shows context.
            lat_span = max(lat_max - lat_min, 1.0)
            lon_span = max(lon_max - lon_min, 1.0)
            margin_lat = 0.05 * lat_span
            margin_lon = 0.05 * lon_span
            ext_lat_min = max(lat_min - margin_lat, -90.0)
            ext_lat_max = min(lat_max + margin_lat, 90.0)
            ext_lon_min = max(lon_min - margin_lon, -180.0)
            ext_lon_max = min(lon_max + margin_lon, 180.0)
            use_global = False
        else:
            use_global = True

        # --- Base map features: zoomed local view or global fallback ---
        if use_global:
            ax.set_global()
            ax.coastlines(resolution="110m", linewidth=0.5, color="0.35")
            ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.93", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d5e8f0", zorder=0)
        else:
            ax.set_extent(
                [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
                crs=ccrs.PlateCarree(),
            )
            # Use 50m resolution for better coastline detail when zoomed in.
            ax.coastlines(resolution="50m", linewidth=0.5, color="0.35")
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.93", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d5e8f0", zorder=0)
        ax.gridlines(draw_labels=False, linewidth=0.3, color="0.6", alpha=0.5)

        # --- Color state: stable per source_name across all its snapshots ---
        prop_colors = [c["color"] for c in rcParams["axes.prop_cycle"]]
        fallback_cycle: Iterator[str] = cycle(prop_colors)
        color_cache: dict[str, str] = {}
        # Deduplicated legend handles — one entry per unique source_name.
        legend_handles: dict[str, mpatches.Patch] = {}

        transform = ccrs.PlateCarree()

        # --- Plot footprint for each (source_name, snapshot_time_utc) pair ---
        for (source_name, snapshot_time_utc), source in self._storm_data.sources.items():
            color = _get_source_color(source_name, color_cache, fallback_cycle)
            footprint = _footprint_from_source(source)

            # Skip sources with no valid data.
            if footprint[0] == "empty":
                continue

            # Only annotate ibtracs best-track entries at exactly 00:00 UTC.
            is_ibtracs = source_name == "ibtracs_best_track"
            is_midnight = snapshot_time_utc[11:16] == "00:00"  # "YYYY-MM-DDTHH:MM..."
            if is_ibtracs and is_midnight:
                mm = snapshot_time_utc[5:7]
                dd = snapshot_time_utc[8:10]
                label = f"{mm}/{dd} 00Z"
            else:
                label = ""

            if footprint[0] == "point":
                _, lon, lat = footprint

                # Single point marker at the geographic coordinate.
                ax.scatter(
                    [lon],
                    [lat],
                    transform=transform,
                    color=color,
                    s=18,
                    zorder=4,
                    linewidths=0.5,
                    edgecolors="white",
                )

                # Text annotation offset slightly from the marker (ibtracs 00Z only).
                if label:
                    ax.text(
                        lon + 1.5,
                        lat + 1.0,
                        label,
                        transform=transform,
                        fontsize=5,
                        color=color,
                        ha="left",
                        va="bottom",
                        clip_on=True,
                        zorder=5,
                    )

            else:  # 'hull'
                _, hull_lons, hull_lats = footprint

                # Semi-transparent filled polygon for the convex hull.
                ax.fill(
                    hull_lons,
                    hull_lats,
                    transform=transform,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

                # Closed boundary line around the hull.
                ax.plot(
                    np.append(hull_lons, hull_lons[0]),
                    np.append(hull_lats, hull_lats[0]),
                    transform=transform,
                    color=color,
                    linewidth=0.8,
                    zorder=3,
                )

                # Annotation at the centroid of the hull vertices (ibtracs 00Z only).
                if label:
                    cx = float(hull_lons.mean())
                    cy = float(hull_lats.mean())
                    ax.text(
                        cx,
                        cy,
                        label,
                        transform=transform,
                        fontsize=5,
                        color=color,
                        ha="center",
                        va="center",
                        clip_on=True,
                        zorder=5,
                    )

            # Register this source_name in the legend (once per unique name).
            if source_name not in legend_handles:
                legend_handles[source_name] = mpatches.Patch(
                    color=color, label=source_name, alpha=0.8
                )

        # Add legend if any sources were rendered.
        if legend_handles:
            ax.legend(
                handles=list(legend_handles.values()),
                loc="lower left",
                fontsize=6,
                framealpha=0.75,
                edgecolor="0.7",
            )

        # Figure title — fall back to storm ID if none provided.
        ax.set_title(
            title or f"Footprints — {self._storm_data.storm_id}",
            fontsize=8,
        )

        # Optional save.
        if save_path is not None:
            save_fig(fig, save_path)

        return fig, ax
