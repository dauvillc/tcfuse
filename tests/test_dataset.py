"""Tests for TCWindowDataset and WindowSample."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tcfuse.data.dataset import TCWindowDataset, WindowSample
from tcfuse.data.sources import StormData
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tests.test_build_splits import INIT_TIME, SOURCE_NAME
from tests.test_sources import make_field_source, make_scalar_source

_SID = "2016292N14270"
_BASIN = "AL"
_SUBBASIN = "GM"
_SEASON = 2016
_ATCF_ID = "AL102016"
_WINDOWS_SETUP = "test_windows"
_TIME_INSIDE_EARLY = (INIT_TIME - pd.Timedelta(hours=6)).isoformat()
_TIME_INSIDE_LATE = (INIT_TIME + pd.Timedelta(hours=6)).isoformat()
_TIME_OUTSIDE = (INIT_TIME + pd.Timedelta(hours=30)).isoformat()
_WINDOW_ID = f"{_SID}_{SOURCE_NAME}_{INIT_TIME.isoformat()}"


def _make_windows_df() -> pd.DataFrame:
    """Build a minimal long-format window index with one window and three snapshots."""
    init_time_str = INIT_TIME.isoformat()
    common = {
        "window_id": _WINDOW_ID,
        "sid": _SID,
        "basin": _BASIN,
        "subbasin": _SUBBASIN,
        "season": _SEASON,
        "usa_atcf_id": _ATCF_ID,
        "window_start_time_utc": _TIME_INSIDE_EARLY,
        "window_end_time_utc": _TIME_INSIDE_LATE,
        "window_ref_time_utc": init_time_str,
    }
    return pd.DataFrame(
        [
            {**common, "source_name": SOURCE_NAME, "time_utc": init_time_str, "is_target": True},
            {
                **common,
                "source_name": "pmw_ssmi",
                "time_utc": _TIME_INSIDE_EARLY,
                "is_target": False,
            },
            {
                **common,
                "source_name": "pmw_ssmi",
                "time_utc": _TIME_INSIDE_LATE,
                "is_target": False,
            },
        ]
    )


def _write_windows_parquet(assembled_root: Path) -> None:
    """Write the test window index to the expected subdirectory."""
    windows_dir = assembled_root / _WINDOWS_SETUP
    windows_dir.mkdir(parents=True, exist_ok=True)
    _make_windows_df().to_parquet(windows_dir / "train_windows.parquet", index=False)


def _write_storm_with_mixed_snapshots(assembled_root: Path) -> None:
    """Write an assembled storm file with in-window and out-of-window snapshots."""
    init_time_str = INIT_TIME.isoformat()
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
            (SOURCE_NAME, init_time_str): make_scalar_source(source_name=SOURCE_NAME),
        },
    )
    storm_data.write(assembled_root)


def _write_sources_metadata(assembled_root: Path) -> None:
    """Write a minimal sources_metadata.yaml for dataset tests."""
    metadata = MultisourceMetadata(
        sources={
            "pmw_ssmi": SourceMetadata(
                name="pmw_ssmi",
                type="microwave",
                kind=SourceKind.FIELD,
                channels=["tb_22.0v", "tb_22.0h"],
                shape=(400, 400),
            ),
            SOURCE_NAME: SourceMetadata(
                name=SOURCE_NAME,
                type="best_track",
                kind=SourceKind.SCALAR,
                channels=["usa_wind", "usa_pres", "lat", "lon"],
                shape=(),
            ),
        }
    )
    metadata.to_yaml(assembled_root / "sources_metadata.yaml")


def _make_dataset(assembled_root: Path) -> TCWindowDataset:
    """Helper: write all fixtures and return a TCWindowDataset."""
    _write_storm_with_mixed_snapshots(assembled_root)
    _write_sources_metadata(assembled_root)
    _write_windows_parquet(assembled_root)
    return TCWindowDataset(assembled_root, _WINDOWS_SETUP, split="train")


class TestTCWindowDataset:
    def test_len_matches_number_of_unique_window_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))
            assert len(dataset) == 1

    def test_getitem_returns_window_sample_with_correct_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))
            sample = dataset[0]

        assert isinstance(sample, WindowSample)
        assert sample.sample_id == _WINDOW_ID
        assert sample.window_ref_time_utc == INIT_TIME.isoformat()
        assert sample.window_start_time_utc == _TIME_INSIDE_EARLY
        assert sample.window_end_time_utc == _TIME_INSIDE_LATE
        assert sample.sid == _SID
        assert sample.season == _SEASON
        assert sample.basin == _BASIN
        assert sample.subbasin == _SUBBASIN
        assert sample.usa_atcf_id == _ATCF_ID

    def test_getitem_loads_sources_in_index_time_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))
            sample = dataset[0]

        source_keys = set(sample.storm_data.sources)
        # The three snapshots referenced in the window index should be loaded.
        assert ("pmw_ssmi", _TIME_INSIDE_EARLY) in source_keys
        assert ("pmw_ssmi", _TIME_INSIDE_LATE) in source_keys
        assert (SOURCE_NAME, INIT_TIME.isoformat()) in source_keys
        # The snapshot outside the index time range must be excluded.
        assert ("pmw_ssmi", _TIME_OUTSIDE) not in source_keys

    def test_getitem_is_target_dict_matches_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))
            sample = dataset[0]

        # SOURCE_NAME at INIT_TIME is the only target in the index.
        # Chronological ordering: pmw_ssmi@early=idx0, SOURCE_NAME@init=idx0, pmw_ssmi@late=idx1.
        assert sample.is_target.get((SOURCE_NAME, 0)) is True
        assert sample.is_target.get(("pmw_ssmi", 0)) is False
        assert sample.is_target.get(("pmw_ssmi", 1)) is False

    def test_index_property_returns_long_format_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))
            # The index is the full long-format DataFrame (3 rows for 1 window).
            assert len(dataset.index) == 3
            assert "window_id" in dataset.index.columns
            assert "is_target" in dataset.index.columns

    def test_sources_metadata_loads_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))

        assert "pmw_ssmi" in dataset.sources_metadata
        pmw = dataset.sources_metadata["pmw_ssmi"]
        assert pmw.kind == SourceKind.FIELD
        assert pmw.shape == (400, 400)
        assert pmw.num_channels == 2

    def test_sources_metadata_property_returns_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _make_dataset(Path(tmpdir))

            returned = dataset.sources_metadata
            returned.sources["pmw_ssmi"].channels.append("tb_99.0v")
            returned.sources["extra"] = SourceMetadata(
                name="extra",
                type="microwave",
                kind=SourceKind.FIELD,
                channels=["tb"],
                shape=(10, 10),
            )

        assert dataset.sources_metadata["pmw_ssmi"].channels == ["tb_22.0v", "tb_22.0h"]
        assert "extra" not in dataset.sources_metadata

    def test_raises_when_sources_metadata_yaml_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            _write_windows_parquet(assembled_root)
            # Deliberately omit sources_metadata.yaml.
            with pytest.raises(FileNotFoundError, match=r"sources_metadata.yaml"):
                TCWindowDataset(assembled_root, _WINDOWS_SETUP, split="train")

    def test_raises_when_windows_parquet_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assembled_root = Path(tmpdir)
            _write_storm_with_mixed_snapshots(assembled_root)
            _write_sources_metadata(assembled_root)
            # Deliberately omit the windows parquet directory.
            with pytest.raises(Exception):
                TCWindowDataset(assembled_root, _WINDOWS_SETUP, split="train")
