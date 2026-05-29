"""Tests for TCWindowDataset and WindowSample."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess.build_splits import build_window_index
from tcfuse.data.dataset import TCWindowDataset, WindowSample
from tcfuse.data.sources import StormData
from tests.test_build_splits import (
    INIT_TIME,
    LEADS_HOURS,
    REQUIRED_COLUMNS,
    REQUIRED_LEADS_HOURS,
    SOURCE_NAME,
    _make_index,
)
from tests.test_sources import make_field_source, make_scalar_source

_SID = "2016292N14270"
_BASIN = "AL"
_SUBBASIN = "GM"
_SEASON = 2016
_ATCF_ID = "AL102016"
_TIME_INSIDE_EARLY = (INIT_TIME - pd.Timedelta(hours=6)).isoformat()
_TIME_INSIDE_LATE = (INIT_TIME + pd.Timedelta(hours=6)).isoformat()
_TIME_OUTSIDE = (INIT_TIME + pd.Timedelta(hours=30)).isoformat()


def _build_window_index_row() -> pd.DataFrame:
    """Build one valid window-index row using the same logic as build_splits."""
    assembled_index = _make_index(LEADS_HOURS)
    samples = build_window_index(
        assembled_index,
        source_name=SOURCE_NAME,
        leads_hours=LEADS_HOURS,
        required_leads_hours=REQUIRED_LEADS_HOURS,
        required_columns=REQUIRED_COLUMNS,
    )
    assert len(samples) == 1
    return samples


def _write_storm_with_mixed_snapshots(assembled_root: Path) -> None:
    """Write an assembled storm file with in-window and out-of-window snapshots."""
    init_time = INIT_TIME.isoformat()
    storm_data = StormData(
        storm_id=_SID,
        basin=_BASIN,
        subbasin=_SUBBASIN,
        season=_SEASON,
        atcf_id=_ATCF_ID,
        sources={
            ("pmw_ssmi", _TIME_INSIDE_EARLY): make_field_source(source_name="pmw_ssmi"),
            ("pmw_ssmi", _TIME_INSIDE_LATE): make_field_source(source_name="pmw_ssmi"),
            ("pmw_ssmi", _TIME_OUTSIDE): make_field_source(source_name="pmw_ssmi"),
            (SOURCE_NAME, init_time): make_scalar_source(source_name=SOURCE_NAME),
        },
    )
    storm_data.write(assembled_root)


class TestTCWindowDataset:
    def test_len_matches_index(self) -> None:
        index = _build_window_index_row()
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            dataset = TCWindowDataset(assembled_root, split="train", index=index)
            assert len(dataset) == 1

    def test_getitem_returns_window_sample_with_filtered_sources(self) -> None:
        index = _build_window_index_row()
        row = index.iloc[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            dataset = TCWindowDataset(assembled_root, split="train", index=index)
            sample = dataset[0]

        assert isinstance(sample, WindowSample)
        assert sample.sample_id == row["sample_id"]
        assert sample.init_time_utc == row["init_time_utc"]
        assert sample.sid == _SID
        assert sample.season == _SEASON
        assert sample.basin == _BASIN
        assert sample.subbasin == _SUBBASIN
        assert sample.usa_atcf_id == _ATCF_ID

        source_keys = set(sample.storm_data.sources)
        assert ("pmw_ssmi", _TIME_INSIDE_EARLY) in source_keys
        assert ("pmw_ssmi", _TIME_INSIDE_LATE) in source_keys
        assert (SOURCE_NAME, INIT_TIME.isoformat()) in source_keys
        assert ("pmw_ssmi", _TIME_OUTSIDE) not in source_keys

    def test_labels_expose_lead_columns_from_index_row(self) -> None:
        index = _build_window_index_row()
        row = index.iloc[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            dataset = TCWindowDataset(assembled_root, split="train", index=index)
            sample = dataset[0]

        lead_columns = [column for column in row.index if str(column).startswith("lead_")]
        assert len(sample.labels) == len(lead_columns)
        assert sample.labels["lead_+000h_usa_wind"] == pytest.approx(65.0)

    def test_index_property_returns_backing_dataframe(self) -> None:
        index = _build_window_index_row()
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            dataset = TCWindowDataset(assembled_root, split="train", index=index)
            assert dataset.index is index
