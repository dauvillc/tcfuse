# /visualize — Publication-Quality Visualization Agent

This skill helps you write visualization functions and modules for the tcfuse project.
Figures are intended for LaTeX articles and preprints (AMS / AGU style), saved as SVG.

---

## Agent behavior rules

1. **Ask before assuming.** When in doubt about color scale, map projection, domain extent, or which channels to display — stop and ask the user. A wrong default wastes more time than one clarifying question.
2. **Consistency first.** Always check `style.py` for existing constants (colormaps, colors, figure widths) before defining new ones. Extend `style.py` when a new constant is genuinely reusable; keep one-off values local.
3. **Keep code simple.** Each function does one thing. No multi-purpose god-functions. Prefer explicit arguments over kwargs catch-alls. Aim for ~40–60 lines per function.
4. **Comment every block.** Add a short `# comment` above every logical code block, even small ones (this is a project-wide convention).
5. **Type hints and docstrings on everything.** One-line summary + Args/Returns for non-trivial functions. Document tensor/array shapes in comments: `# (H, W)`.
6. **Always use `setup_style()` and `save_fig()`.** Call `setup_style()` at module import time. Save all figures with `save_fig(fig, path)` to ensure SVG output and consistent margins.
7. **No hardcoded paths.** Output paths come from function arguments or `cfg.paths.*` — never hardcoded strings.
8. **Install missing packages with pixi.** If a required package is absent, add it to `pixi.toml` with `pixi add <package>` (conda-forge). Do not use pip.

---

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

---

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

cmap = get_cmap("tb")      # brightness temperature — cmocean.thermal or "inferno"
cmap = get_cmap("wind")    # wind speed          — cmocean.speed or "YlOrRd"
cmap = get_cmap("anomaly") # signed anomaly      — cmocean.balance or "RdBu_r"
```

Available keys: `"tb"`, `"wind"`, `"sst"`, `"precip"`, `"anomaly"`, `"depth"`.
`get_cmap()` returns the cmocean version when installed, otherwise a matplotlib fallback.

### Categorical colors

```python
from tcfuse.data.visualization.style import INTENSITY_COLORS, SOURCE_COLORS

color = INTENSITY_COLORS["C3"]          # Saffir-Simpson intensity color
color = SOURCE_COLORS["pmw"]            # source-type color for legends
```

### Setup and save

```python
from tcfuse.data.visualization.style import setup_style, save_fig

setup_style()  # call once at module import

fig, ax = plt.subplots(...)
# ... draw ...
path = save_fig(fig, "figures/my_plot")  # saves to figures/my_plot.svg
```

Set `TCFUSE_NO_LATEX=1` to disable the LaTeX renderer (e.g. on nodes without a LaTeX install).

---

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
├── fields.py             ← 2D satellite / model field plots (PMW, IR, ERA5)
├── profiles.py           ← vertical profile plots (dropsonde, Argo)
└── training.py           ← loss curves, attention weights, model diagnostics
```

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

---

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tcfuse.data.visualization.style import setup_style, COL1, AR_GOLDEN, INTENSITY_COLORS, save_fig

setup_style()

def plot_track(lats, lons, vmax_kt, ax=None, *, title="", save_path=None):
    # Create figure and geo-axes if not provided
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(COL1, COL1),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = ax.get_figure()

    # Add coastlines and land fill
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0f0f0", zorder=0)

    # Color-code each segment by intensity category
    for i in range(len(lats) - 1):
        cat = _vmax_to_category(vmax_kt[i])
        ax.plot(
            [lons[i], lons[i+1]], [lats[i], lats[i+1]],
            color=INTENSITY_COLORS[cat], linewidth=1.2,
            transform=ccrs.PlateCarree(),
        )

    # Set storm-relative domain
    pad = 10.0
    ax.set_extent([lons.min()-pad, lons.max()+pad, lats.min()-pad, lats.max()+pad])

    if title:
        ax.set_title(title)
    if save_path:
        save_fig(fig, save_path)
    return fig, ax
```

### 2. 2D satellite field

**Module:** `fields.py`
**When to use:** Display one channel of a PMW / IR / ERA5 field source.

Key design decisions:
- Projection: `ccrs.PlateCarree()` (field coords are already lat/lon)
- Colormap: `get_cmap("tb")` for PMW, `get_cmap("wind")` for ERA5 wind, etc.
- NaN masking: pass `values` as a masked array or rely on `pcolormesh` NaN handling
- Colorbar: horizontal, below the plot, label = physical quantity + unit
- Storm center: small crosshair marker at `(storm_lat, storm_lon)`
- Domain: derived from field coordinate bounding box (no hardcoded extents)

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tcfuse.data.visualization.style import setup_style, COL1, get_cmap, save_fig

setup_style()

def plot_field(values, lats, lons, *, channel="tb", unit="K",
               storm_lat=None, storm_lon=None, ax=None, title="", save_path=None):
    # (values: H×W, lats: H×W, lons: H×W)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(COL1, COL1),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = ax.get_figure()

    # Draw the field with the appropriate colormap
    im = ax.pcolormesh(lons, lats, values, cmap=get_cmap(channel),
                       transform=ccrs.PlateCarree(), shading="auto")

    # Add coastlines
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)

    # Colorbar below the axes with physical unit label
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label(f"{channel} ({unit})")

    # Storm center crosshair (if provided)
    if storm_lat is not None and storm_lon is not None:
        ax.plot(storm_lon, storm_lat, "+", color="white", markersize=6,
                markeredgewidth=1.0, transform=ccrs.PlateCarree())

    # Set domain from field bounding box
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    if title:
        ax.set_title(title)
    if save_path:
        save_fig(fig, save_path)
    return fig, ax
```

---

## Adding a new visualization module

1. Create `src/tcfuse/data/visualization/<name>.py`.
2. Import `setup_style` and call it at module level.
3. Use `COL1` / `COL2` / `AR_GOLDEN` for all `figsize` values.
4. Use `get_cmap()` for colormaps; add a new key to `style.py` if the quantity is new.
5. Follow the `plot_<thing>(data, ax=None, *, ...) -> (fig, ax)` signature pattern.
6. Save with `save_fig(fig, path)`.
7. Add the new file to the module table above.
