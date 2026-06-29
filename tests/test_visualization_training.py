"""Unit tests for training/validation diagnostic visualizations (synthetic only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tcfuse.data.visualization.training import (
    plot_field_reconstruction,
    render_field_reconstruction,
)


def _synthetic_field(
    h: int, w: int, c: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (target, prediction, lats, lons) synthetic arrays for a FIELD source."""
    rng = np.random.default_rng(0)
    # Random target and a noisy prediction of the same shape.
    target = rng.normal(size=(h, w, c)).astype(np.float32)
    prediction = target + rng.normal(scale=0.1, size=(h, w, c)).astype(np.float32)
    # Simple regular lat/lon grid.
    lat_vec = np.linspace(10.0, 20.0, h)
    lon_vec = np.linspace(-80.0, -70.0, w)
    lons, lats = np.meshgrid(lon_vec, lat_vec)
    return target, prediction, lats.astype(np.float32), lons.astype(np.float32)


def test_plot_field_reconstruction_panel_count() -> None:
    """Each channel yields three panels (target, prediction, error)."""
    c = 3
    target, prediction, lats, lons = _synthetic_field(6, 8, c)
    channels = [f"ch{i}" for i in range(c)]

    fig, axes = plot_field_reconstruction(
        target, prediction, lats, lons, channels=channels, unit="K"
    )

    assert len(axes) == 3 * c
    plt.close(fig)


def test_plot_field_reconstruction_handles_mask() -> None:
    """A mask with all-invalid pixels is tolerated (no finite color range)."""
    c = 1
    target, prediction, lats, lons = _synthetic_field(5, 5, c)
    # All pixels unavailable -> every channel is NaN after masking.
    mask = np.zeros((5, 5, c), dtype=bool)

    fig, axes = plot_field_reconstruction(
        target, prediction, lats, lons, channels=["ch0"], mask=mask
    )

    assert len(axes) == 3
    plt.close(fig)


def test_render_field_reconstruction_saves_svg_and_returns_raster(tmp_path) -> None:
    """The wrapper writes an SVG and returns a fixed-size RGB raster."""
    c = 2
    target, prediction, lats, lons = _synthetic_field(6, 8, c)
    channels = [f"ch{i}" for i in range(c)]
    save_path = tmp_path / "recon"

    # Different suptitles must still produce identically sized rasters (the W&B
    # same-key gallery requires it).
    rgb_a = render_field_reconstruction(
        target,
        prediction,
        lats,
        lons,
        channels=channels,
        source_name="pmw_gmi",
        save_path=save_path,
        suptitle="short",
    )
    rgb_b = render_field_reconstruction(
        target,
        prediction,
        lats,
        lons,
        channels=channels,
        source_name="pmw_gmi",
        save_path=save_path,
        suptitle="a_much_longer_suptitle_2019141N28291",
    )

    # SVG written next to the requested path.
    assert save_path.with_suffix(".svg").exists()
    # RGB raster with a stable shape regardless of suptitle length.
    assert rgb_a.ndim == 3 and rgb_a.shape[2] == 3
    assert rgb_a.shape == rgb_b.shape
