"""Unit tests for storm footprint visualization helpers."""

from __future__ import annotations

import pytest

from tcfuse.data.visualization.storm_data_visu import _footprint_from_source
from tests.test_sources import make_batched_field_source


def test_footprint_from_source_rejects_batched_source() -> None:
    """Storm footprint visualization should reject batched Source inputs."""
    source = make_batched_field_source(B=2, H=6, W=8, C=2, source_name="pmw_test")
    with pytest.raises(ValueError, match="expects non-batched Source objects"):
        _footprint_from_source(source)
