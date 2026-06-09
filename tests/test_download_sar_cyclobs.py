"""Tests for CyclObs SAR download metadata filtering."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.preprocess.sar.download_sar_cyclobs import filter_sar_acquisitions_metadata


def _metadata() -> pd.DataFrame:
    """Create representative CyclObs acquisition metadata."""
    return pd.DataFrame(
        {
            "sid": ["al012020", "ep022020", "al032021", "wp042022"],
            "data_url": [
                "https://example.test/al012020.nc",
                "https://example.test/ep022020.nc",
                "https://example.test/al032021.nc",
                "https://example.test/wp042022.nc",
            ],
        }
    )


def test_filter_sar_acquisitions_metadata_without_filters_leaves_rows_unchanged() -> None:
    """No filters should preserve the metadata rows and columns."""
    metadata = _metadata()

    filtered = filter_sar_acquisitions_metadata(metadata)

    pd.testing.assert_frame_equal(filtered, metadata)


def test_filter_sar_acquisitions_metadata_selects_multiple_seasons() -> None:
    """Season filters should match the year suffix in CyclObs sid values."""
    filtered = filter_sar_acquisitions_metadata(
        _metadata(),
        include_seasons=[2020, 2022],
    )

    assert filtered["sid"].tolist() == ["al012020", "ep022020", "wp042022"]


def test_filter_sar_acquisitions_metadata_selects_basins_case_insensitively() -> None:
    """Basin filters should normalize user input and sid prefixes to uppercase."""
    filtered = filter_sar_acquisitions_metadata(
        _metadata(),
        include_basins=["al", "WP"],
    )

    assert filtered["sid"].tolist() == ["al012020", "al032021", "wp042022"]


def test_filter_sar_acquisitions_metadata_combines_seasons_and_basins() -> None:
    """Combined filters should keep only rows matching both constraints."""
    filtered = filter_sar_acquisitions_metadata(
        _metadata(),
        include_seasons=[2020, 2021],
        include_basins="AL",
    )

    assert filtered["sid"].tolist() == ["al012020", "al032021"]


def test_filter_sar_acquisitions_metadata_requires_sid_when_filtering() -> None:
    """Filtering requires the CyclObs storm identifier column."""
    metadata = pd.DataFrame({"data_url": ["https://example.test/file.nc"]})

    with pytest.raises(ValueError, match="'sid' column is required"):
        filter_sar_acquisitions_metadata(metadata, include_seasons=[2020])
