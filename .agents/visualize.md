# TC-Fuse visualization

Claude Code: invoke `/visualize` (reads this skill).

Figures target AMS / AGU journal layout and are saved as SVG. No external LaTeX installation is required.

**Coding style:** follow [`.agents/context.md`](context.md) § Human-readable code (priority).

## When to use

- Writing or modifying any function under `src/tcfuse/data/visualization/`.
- Adding a new figure type or thematic plotting module.
- Defining or extending shared style constants in `style.py`.
- Producing figures for the paper or preprint pipeline.

## Agent behavior rules

1. **Ask before assuming.** When in doubt about color scale, map projection, domain extent, or which channels to display — stop and ask the user. A wrong default wastes more time than one clarifying question.
2. **Consistency first.** Always check `style.py` for existing constants (colormaps, colors, figure widths) before defining new ones. Extend `style.py` when a new constant is genuinely reusable; keep one-off values local.
3. **Keep code simple.** Each function does one thing. No multi-purpose god-functions. Prefer explicit arguments over kwargs catch-alls. Aim for ~40–60 lines per function.
4. **Comment every block.** Add a short `# comment` above every logical code block, even small ones (this is a project-wide convention).
5. **Type hints and docstrings on everything.** One-line summary + Args/Returns for non-trivial functions. Document tensor/array shapes in comments: `# (H, W)`.
6. **Always use `setup_style()` and `save_fig()`.** Call `setup_style()` at module import time. Save all figures with `save_fig(fig, path)` to ensure SVG output and consistent margins.
7. **No hardcoded paths.** Output paths come from function arguments or `cfg.paths.*` — never hardcoded strings.
8. **Install missing packages with pixi.** If a required package is absent, add it to `pixi.toml` with `pixi add <package>` (conda-forge). Do not use pip.

## Package inventory

| Package | Status | Purpose |
|---|---|---|
| `matplotlib` | ✓ pixi.toml | Base plotting |
| `cartopy` | ✓ pixi.toml | Geographic projections, coastlines, gridlines |
| `numpy` | ✓ pixi.toml | Array operations |
| `scipy` | ✓ pixi.toml | Interpolation, smoothing |
| `xarray` | ✓ pixi.toml | Labeled arrays (useful for field data) |
| `cmocean` | add if needed | Perceptually uniform colormaps (`pixi add cmocean`) |
| `metpy` | optional | Unit handling, skew-T plots (`pixi add metpy`) |

## Style conventions

All constants and helpers live in `src/tcfuse/data/visualization/style.py`.

### Figure widths

```python
from tcfuse.data.visualization.style import COL1, COL2, AR_GOLDEN

fig, ax = plt.subplots(figsize=(COL1, COL1 * AR_GOLDEN))  # single-column
fig, ax = plt.subplots(figsize=(COL2, COL2 * AR_GOLDEN))  # double-column
```

| Constant | Value | Use |
|---|---|---|
| `COL1` | 3.5 in | Single-column figure |
| `COL2` | 7.0 in | Double-column / full-width figure |
| `AR_GOLDEN` | 0.618 | Default aspect ratio (height / width) |

### Colormaps

```python
from tcfuse.data.visualization.style import get_cmap

cmap = get_cmap("tb")       # brightness temperature — cmocean.thermal or "inferno"
cmap = get_cmap("wind")     # wind speed             — cmocean.speed or "YlOrRd"
cmap = get_cmap("sar_wind") # SAR wind speed         — cmocean.speed or "YlOrRd" (alias for wind)
cmap = get_cmap("anomaly")  # signed anomaly         — cmocean.balance or "RdBu_r"
```

Available keys: `"tb"`, `"wind"`, `"sar_wind"`, `"sst"`, `"precip"`, `"anomaly"`, `"depth"`.
`get_cmap()` returns the cmocean version when installed, otherwise a matplotlib fallback.

### Categorical colors

```python
from tcfuse.data.visualization.style import INTENSITY_COLORS, SOURCE_COLORS

color = INTENSITY_COLORS["C3"]          # Saffir-Simpson intensity color
color = SOURCE_COLORS["pmw"]            # source-type color for legends
color = SOURCE_COLORS["sar"]            # SAR source color (#17becf teal)
```

### Setup and save

```python
from tcfuse.data.visualization.style import setup_style, save_fig

setup_style()  # call once at module import

fig, ax = plt.subplots(...)
# ... draw ...
path = save_fig(fig, "figures/my_plot")  # saves to figures/my_plot.svg
```

`plot_field()` rasterizes the `pcolormesh` layer on save so SVG/PDF files stay small while
titles, ticks, and coastlines remain vector.

`plot_source_timeline()` histograms snapshot times into UTC bins and rasterizes the
availability strip so large assembled indexes do not emit one SVG path per snapshot;
titles and axis labels remain vector.

`setup_style()` sets `text.usetex=False` and uses system serif fonts (DejaVu Serif, STIX, Times).

Use `UNIT_K`, `UNIT_MM_H`, `UNIT_M_S` from `style.py` for colorbar units (matplotlib mathtext
exponents, e.g. `$^{-1}$`; not Unicode superscripts). Pass channel names and titles as plain strings.

## Visualization module conventions

### File location

All visualization modules live in `src/tcfuse/data/visualization/`.
One file per thematic domain:

```
visualization/
├── style.py              ← shared style foundation (do not scatter style config elsewhere)
├── storm_data_visu.py    ← StormDataVisualizer class: show_footprints and future overview plots
├── timeline.py           ← plot_source_timeline(): source availability eventplot from assembled index
├── tracks.py             ← TC track and intensity maps
├── fields.py             ← 2D field plots: plot_field(), plot_field_from_source(), plot_field_source_channels(), plot_sar_wind()
├── profiles.py           ← vertical profile plots (dropsonde, Argo)
├── data_profile.py       ← windows-setup profiling from the windows-index parquet: compute_split_summary() + plot_sample_timeline/samples_per_season/source_availability/target_distribution/sources_per_window_hist/basin_distribution/windows_per_storm (each takes {split: DataFrame})
├── comparison.py         ← model-comparison figures for the evaluation suite: plot_metric_comparison() (grouped bars, x=channel, one bar per model, per metric/source)
└── training.py           ← model diagnostics: plot_field_reconstruction() (Target|Pred|Error per channel), loss curves, attention weights
```

**CLI scripts** under `scripts/visualization/` (Hydra + `preproc` config):

| Script | Output |
|---|---|
| `source_timeline.py` | `figures/source_timeline.svg` — assembled-index source availability |
| `plot_source_examples.py` | `figures/source_examples/{source_name}.svg` — one multi-panel example per Stage 1 source (all PMW sensors, IR, radar, ERA5 surface, SAR) |

### Function signature pattern

```python
def plot_<thing>(
    <data_arg>,            # the data to plot (Source, np.ndarray, pd.DataFrame, …)
    ax: "GeoAxes | None" = None,  # pass an existing axes to embed in a subplot layout
    *,
    title: str = "",
    save_path: "Path | str | None" = None,
) -> tuple["Figure", "Axes"]:
    """One-line summary.

    Args:
        <data_arg>: Description.
        ax:         Axes to draw into; a new figure is created when None.
        title:      Optional figure title.
        save_path:  If provided, save the figure here (SVG).

    Returns:
        (fig, ax) tuple.
    """
```

- Accept an optional `ax` so the function can be embedded in multi-panel layouts.
- Return `(fig, ax)` always — lets the caller adjust further before saving.
- Accept `save_path` as an optional shortcut; the caller can also call `save_fig` manually.

## Priority plot types

### 1. TC track map

**Module:** `tracks.py`
**When to use:** Overview of a storm's life cycle, intensity evolution, or dataset coverage.

Key design decisions:
- Projection: `ccrs.PlateCarree()` (simple, works for regional TC domains)
- Domain: storm-relative, ±15° around the mean track center (adjust per storm)
- Track markers: colored by intensity using `INTENSITY_COLORS`; size proportional to `vmax_kt`
- Coastlines: Natural Earth 50 m resolution (`feature.COASTLINE` with `scale="50m"`)
- Gridlines: subtle (alpha=0.3, linewidth=0.4), labels on left and bottom only

```python
setup_style()

def plot_track(lats, lons, vmax_kt, ax=None, *, title="", save_path=None):
    # Create GeoAxes(ccrs.PlateCarree(), figsize=(COL1, COL1)) if ax is None
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), ...)  # + LAND fill (#f0f0f0)
    for i in range(len(lats) - 1):  # color each segment by INTENSITY_COLORS[_vmax_to_category(vmax_kt[i])]
        ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], color=..., transform=ccrs.PlateCarree())
    ax.set_extent([lons.min()-10, lons.max()+10, lats.min()-10, lats.max()+10])
    if save_path: save_fig(fig, save_path)
    return fig, ax
```

### 2. 2D satellite field

**Module:** `fields.py`
**When to use:** Display one channel of a PMW / IR / ERA5 / SAR field source.

Key design decisions:
- Projection: `ccrs.PlateCarree()` (field coords are already lat/lon)
- Colormap: `get_cmap("tb")` for PMW, `get_cmap("wind")` for ERA5 wind, `get_cmap("sar_wind")` for SAR, etc.
- NaN masking: pass `values` as a masked array or rely on `pcolormesh` NaN handling
- Colorbar: horizontal, below the plot, label = physical quantity + unit
- Storm center: small crosshair marker at `(storm_lat, storm_lon)`
- Domain: derived from field coordinate bounding box (no hardcoded extents)

```python
from tcfuse.data.visualization.fields import plot_field, plot_sar_wind

# Generic field (raw numpy arrays):
fig, ax = plot_field(values, lats, lons, channel="sar_wind", unit=UNIT_M_S,
                     storm_lat=15.0, storm_lon=-75.0)

# SAR convenience wrapper (accepts a Source object directly):
fig, ax = plot_sar_wind(source, storm_lat=15.0, storm_lon=-75.0)
```

`plot_sar_wind` delegates to `plot_field_from_source()` (channel 0, `sar_wind` colormap).
`plot_field_source_channels()` lays out every channel in a 2×2 (PMW), 1×3 (radar), or 1×1 grid;
use it from `plot_source_examples.py` or custom gallery scripts.

### 3. Field reconstruction comparison

**Module:** `training.py`
**When to use:** Spot-check a reconstructed FIELD source against its ground truth (e.g. during validation, or from an offline inference run).

`plot_field_reconstruction(target, prediction, lats, lons, *, channels, cmap_key="tb", unit="", mask=None, suptitle="", save_path=None)`
takes plain numpy arrays (`target`/`prediction`/`mask` shape `(H, W, C)`, `lats`/`lons` shape `(H, W)`)
so it is dataset-agnostic and unit-testable with synthetic data. It draws one row per channel with
three panels: **Target | Prediction** (shared per-channel color scale) and **Error** = prediction − target
(symmetric `anomaly` diverging scale).

`render_field_reconstruction(target, prediction, lats, lons, *, channels, source_name, save_path, mask=None, suptitle="")`
is the convenience wrapper used by training: it derives the colormap/unit from `source_name`, builds and
saves the SVG, and returns a fixed-size RGB raster (so an image logger gets consistent dimensions) — all
matplotlib stays here, not in the Lightning module. `MaskedReconstructionLightningModule._render_validation_figures`
calls it on the first `num_val_figure_samples` validation samples each epoch, saving SVGs under
`validation_dir/<step>/` and logging the rasters to W&B under `val/reconstruction/<source>`.

## Adding a new visualization module

1. Create `src/tcfuse/data/visualization/<name>.py`.
2. Import `setup_style` and call it at module level.
3. Use `COL1` / `COL2` / `AR_GOLDEN` for all `figsize` values.
4. Use `get_cmap()` for colormaps; add a new key to `style.py` if the quantity is new.
5. Follow the `plot_<thing>(data, ax=None, *, ...) -> (fig, ax)` signature pattern.
6. Save with `save_fig(fig, path)`.
7. Add the new file to the module table above.

## Maintenance

When changing `src/tcfuse/data/visualization/style.py` or adding a new visualization module,
update this skill in the same PR. If triggers or behavior rules change, also update
`.claude/commands/visualize.md`.
