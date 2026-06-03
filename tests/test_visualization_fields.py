"""Unit tests for field visualization helpers (synthetic tensors only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tcfuse.data.visualization.fields import ChannelPlotSpec, plot_field_source_channels
from tcfuse.data.visualization.style import UNIT_MM_H, format_text_for_renderer
from tests.test_sources import make_field_source


def test_format_text_for_renderer_escapes_underscore_with_usetex() -> None:
    """Underscores in channel names must be escaped when usetex is on."""
    plt.rcParams["text.usetex"] = True
    assert format_text_for_renderer("tb_36.5h") == r"tb\_36.5h"
    plt.rcParams["text.usetex"] = False
    assert format_text_for_renderer("tb_36.5h") == "tb_36.5h"


def test_unit_constants_are_latex_safe() -> None:
    """Canonical units use mathtext exponents, not Unicode superscripts."""
    assert "$^{-1}$" in UNIT_MM_H
    assert "\u207b" not in UNIT_MM_H


def test_plot_field_source_channels_multi_panel() -> None:
    """Multi-channel FIELD source produces one axes per channel."""
    source = make_field_source(H=6, W=8, C=4, source_name="pmw_test")
    specs = [ChannelPlotSpec("tb", "K") for _ in range(4)]

    fig, axes = plot_field_source_channels(source, specs, suptitle="test")

    assert len(axes) == 4
    plt.close(fig)


