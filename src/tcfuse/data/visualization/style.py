"""Shared publication-style configuration for all tcfuse visualization modules.

Every visualization module should call setup_style() at import time and use the
width constants (COL1, COL2) and colormaps defined here to keep figures consistent
across the project.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

# --- Figure width constants (inches, AMS / AGU two-column style) ---
COL1 = 3.5  # single-column width
COL2 = 7.0  # double-column (full-width) figure
AR_GOLDEN = 0.618  # golden-ratio height/width — good default for single-panel figures

# Physical units for colorbar labels (valid with usetex and mathtext).
UNIT_K = "K"
UNIT_MM_H = r"mm h$^{-1}$"
UNIT_M_S = r"m s$^{-1}$"


def format_text_for_renderer(text: str) -> str:
    """Escape plain-text labels when ``text.usetex`` is enabled.

    Unit strings that already contain math (e.g. :data:`UNIT_MM_H`) should be passed
    through unchanged. Apply this helper to channel names, titles, and suptitles.

    Args:
        text: Raw label text, possibly with ``_`` or ``%``.

    Returns:
        Text safe for the active matplotlib text backend.
    """
    if not plt.rcParams.get("text.usetex", False):
        return text
    escaped = text
    for char, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        escaped = escaped.replace(char, replacement)
    return escaped


# --- Colormap catalogue ---
# Built lazily so the module can be imported even when cmocean is not installed.
# Access via: cmap = CMAPS.get("tb", CMAPS_FALLBACK["tb"])
def _build_cmap_catalogue() -> dict:
    """Return a dict of cmocean colormaps, or empty dict if cmocean is absent."""
    try:
        import cmocean.cm as cmo

        return {
            "tb": cmo.thermal,  # brightness temperature (PMW)
            "wind": cmo.speed,  # surface wind speed
            "sar_wind": cmo.speed,  # SAR-derived wind speed (alias for wind)
            "sst": cmo.thermal,  # sea surface temperature
            "precip": cmo.rain,  # rainfall / rain rate
            "anomaly": cmo.balance,  # signed anomalies (diverging, zero-centered)
            "depth": cmo.deep,  # ocean depth / bathymetry
        }
    except ImportError:
        return {}


CMAPS: dict = _build_cmap_catalogue()

# Fallback colormaps from matplotlib — always available, no extra dependency
CMAPS_FALLBACK: dict[str, str] = {
    "tb": "inferno",
    "wind": "YlOrRd",
    "sar_wind": "YlOrRd",  # SAR-derived wind speed (alias for wind)
    "sst": "inferno",
    "precip": "Blues",
    "anomaly": "RdBu_r",
    "depth": "Blues_r",
}


def get_cmap(key: str):
    """Return the best available colormap for the given physical quantity.

    Prefers cmocean if installed; falls back to the matplotlib equivalent.

    Args:
        key: Physical quantity key, e.g. "tb", "wind", "anomaly".

    Returns:
        A matplotlib-compatible colormap object or name string.
    """
    return CMAPS.get(key, CMAPS_FALLBACK.get(key, "viridis"))


# --- Categorical colors for TC intensity (Saffir-Simpson + sub-tropical) ---
INTENSITY_COLORS: dict[str, str] = {
    "TD": "#9ecae1",  # tropical depression
    "TS": "#4292c6",  # tropical storm
    "C1": "#fee090",  # category 1
    "C2": "#fdae61",  # category 2
    "C3": "#f46d43",  # category 3
    "C4": "#d73027",  # category 4
    "C5": "#a50026",  # category 5
}

# --- Source-type colors (for multi-source comparison legends) ---
SOURCE_COLORS: dict[str, str] = {
    "pmw": "#1f77b4",
    "ir": "#ff7f0e",
    "era5": "#2ca02c",
    "best_track": "#d62728",
    "dropsonde": "#9467bd",
    "argo": "#8c564b",
    "radar": "#e377c2",
    "sar": "#17becf",  # C-band SAR wind speed (teal/cyan)
}


def setup_style() -> None:
    """Apply publication-quality rcParams for LaTeX journal figures.

    Call once per module (at import time) or at the top of a notebook cell.
    Requires a working LaTeX installation for full rendering.  Set the
    environment variable TCFUSE_NO_LATEX=1 to fall back to matplotlib's
    mathtext renderer (no LaTeX required).
    """
    # Check whether the user has opted out of the LaTeX renderer
    use_latex = os.environ.get("TCFUSE_NO_LATEX", "0") != "1"

    plt.rcParams.update(
        {
            # --- Typography ---
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.titlesize": 8,
            # --- Lines and tick marks ---
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            # --- Axes appearance ---
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            # --- Output quality ---
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "figure.dpi": 100,
        }
    )


def save_fig(fig: plt.Figure, path: "Path | str", *, svg: bool = True) -> Path:
    """Save a figure as SVG (default) or PDF.

    The file extension in `path` is always replaced by .svg or .pdf.
    Parent directories are created automatically.

    Args:
        fig:  The matplotlib Figure to save.
        path: Output path; extension is replaced as needed.
        svg:  Save as SVG when True (default), PDF when False.

    Returns:
        The resolved Path of the saved file.
    """
    # Normalise path and enforce the correct extension
    path = Path(path).with_suffix(".svg" if svg else ".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the figure
    fig.savefig(path, format="svg" if svg else "pdf")
    return path
