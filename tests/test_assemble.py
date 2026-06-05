"""Unit tests for scripts/preprocess/assemble.py and tcfuse.data.ibtracs.

All tests use synthetic DataFrames — no real IBTrACS file is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scripts.preprocess.assemble import assemble_storm, build_assembled_index

from tcfuse.data.ibtracs import (
    IBTRACS_CHANNELS,
    IBTRACS_SOURCE_NAME,
    group_ibtracs_by_sid,
    ibtracs_paths,
    ibtracs_rows_to_sources,
    load_atcf_to_sid,
    load_ibtracs_snapshots,
)
from tcfuse.data.sources import Source, SourceKind, StormData
from tests.test_sources import make_field_source

# ---------------------------------------------------------------------------
# IBTrACS row fixtures
# ---------------------------------------------------------------------------


def _make_ibtracs_row(
    sid: str = "2016292N14270",
    iso_time: str = "2016-10-05T00:00:00",
    lat: float = 15.0,
    lon: float = -60.0,
    basin: str = "AL",
    subbasin: str = "GM",
    season: int = 2016,
    name: str = "MATTHEW",
    number: int = 14,
    nature: str = "TS",
    usa_atcf_id: str = "AL102016",
    usa_wind: float | None = 65.0,
    usa_pres: float | None = 970.0,
    usa_sshs: int | None = 2,
) -> dict[str, Any]:
    """Build a single Stage-0 IBTrACS-style snapshot dict (canonical lowercase)."""
    row: dict[str, Any] = {
        "sid": sid,
        "season": season,
        "basin": basin,
        "subbasin": subbasin,
        "name": name,
        "number": number,
        "iso_time": iso_time,
        "nature": nature,
        "lat": lat,
        "lon": lon,
        "usa_atcf_id": usa_atcf_id,
        "usa_wind": usa_wind,
        "usa_pres": usa_pres,
        "usa_sshs": usa_sshs,
    }
    for prefix in ("usa_r34", "usa_r50", "usa_r64"):
        for quad in ("ne", "se", "sw", "nw"):
            row[f"{prefix}_{quad}"] = 100.0
    return row


def _make_ibtracs_df(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Stage-0 loaders (read parquet/CSV from disk)
# ---------------------------------------------------------------------------


def _write_stage0(sources_root: Path, snapshots: pd.DataFrame) -> None:
    """Persist a synthetic Stage 0 parquet + translation CSV under sources_root."""
    parquet_path, csv_path = ibtracs_paths(sources_root)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(parquet_path, index=False)

    keep = cast(
        pd.DataFrame,
        snapshots[["sid", "season", "basin", "subbasin", "name", "usa_atcf_id"]],
    )
    deduped = cast(pd.DataFrame, keep.drop_duplicates(subset=["sid", "usa_atcf_id"]))
    deduped.reset_index(drop=True).to_csv(csv_path, index=False)


class TestStage0Loaders:
    def test_load_ibtracs_snapshots_round_trip(self, tmp_path: Path) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(sid="A"), _make_ibtracs_row(sid="B"))
        _write_stage0(tmp_path, df)
        loaded = load_ibtracs_snapshots(tmp_path)
        assert set(loaded["sid"]) == {"A", "B"}

    def test_load_atcf_to_sid_columns(self, tmp_path: Path) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(sid="A", usa_atcf_id="AL102016"))
        _write_stage0(tmp_path, df)
        table = load_atcf_to_sid(tmp_path)
        assert set(table.columns) == {
            "sid",
            "season",
            "basin",
            "subbasin",
            "name",
            "usa_atcf_id",
        }
        assert (table["sid"] == "A").all()


# ---------------------------------------------------------------------------
# ibtracs_rows_to_sources
# ---------------------------------------------------------------------------


class TestIbtracsRowsToSources:
    def test_returns_one_source_per_row(self) -> None:
        df = _make_ibtracs_df(
            _make_ibtracs_row(iso_time="2016-10-05T00:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05T06:00:00"),
        )
        result = ibtracs_rows_to_sources(df, "SID1", "AL")
        assert len(result) == 2

    def test_source_kind_is_scalar(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert source.kind is SourceKind.SCALAR

    def test_source_name(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert source.source_name == IBTRACS_SOURCE_NAME

    def test_channels_match_constant(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert source.channels == IBTRACS_CHANNELS

    def test_values_shape_is_16(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert source.values.shape == (16,)

    def test_coords_shape_is_2(self) -> None:
        # SCALAR coords are [lat, lon] — no time channel.
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert source.coords.shape == (2,)

    def test_lat_lon_duplicated_in_values(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(lat=15.0, lon=-60.0))
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        lat_idx = IBTRACS_CHANNELS.index("lat")
        lon_idx = IBTRACS_CHANNELS.index("lon")
        assert float(source.values[lat_idx]) == pytest.approx(15.0)
        assert float(source.values[lon_idx]) == pytest.approx(-60.0)
        # And also in coords: coords = [lat, lon].
        assert float(source.coords[0]) == pytest.approx(15.0)
        assert float(source.coords[1]) == pytest.approx(-60.0)

    def test_usa_wind_first_channel(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=80.0))
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert float(source.values[0]) == pytest.approx(80.0)

    def test_missing_usa_wind_is_nan(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=None))
        _, source = ibtracs_rows_to_sources(df, "SID1", "AL")[0]
        assert np.isnan(float(source.values[0]))

    def test_row_with_nan_lat_is_skipped(self) -> None:
        df = _make_ibtracs_df(
            _make_ibtracs_row(lat=float("nan")),
            _make_ibtracs_row(iso_time="2016-10-05T06:00:00", lat=15.0),
        )
        result = ibtracs_rows_to_sources(df, "SID1", "AL")
        assert len(result) == 1

    def test_chronological_order(self) -> None:
        df = _make_ibtracs_df(
            _make_ibtracs_row(iso_time="2016-10-05T12:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05T00:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05T06:00:00"),
        )
        result = ibtracs_rows_to_sources(df, "SID1", "AL")
        times = [t for t, _ in result]
        assert times == sorted(times)

    def test_meta_carries_sid_and_basin(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "MY_SID", "EP")[0]
        assert source.meta["storm_id"] == "MY_SID"
        assert source.meta["basin"] == "EP"


# ---------------------------------------------------------------------------
# assemble_storm
# ---------------------------------------------------------------------------


def _stage1_snapshot(
    tmp_path: Path,
    sid: str,
    snapshot_time: str,
    source_name: str = "pmw_ssmi",
) -> Path:
    """Write a synthetic per-source HDF5 snapshot under Source.path."""
    src = make_field_source(H=2, W=3, C=1, source_name=source_name)
    src.meta = {
        "storm_id": sid,
        "time_utc": snapshot_time,
    }
    # Mirror real Stage 1 file naming.
    from tcfuse.utils.time import to_compact_time

    path = Source.path(tmp_path, source_name, sid, to_compact_time(snapshot_time))
    src.write(path)
    return path


class TestAssembleStorm:
    _SID = "2016292N14270"

    def _fixtures(
        self,
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, dict[str, Any]],
        dict[str, str],
    ]:
        df = _make_ibtracs_df(_make_ibtracs_row(sid=self._SID, usa_atcf_id="AL102016"))
        ibtracs_by_sid = group_ibtracs_by_sid(df)
        sid_attrs = {self._SID: {"season": 2016, "basin": "AL", "subbasin": "GM"}}
        atcf_for_sid = {self._SID: "AL102016"}
        return ibtracs_by_sid, sid_attrs, atcf_for_sid

    def test_streams_disk_snapshot_into_assembled_file(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        assembled_root = tmp_path / "assembled"
        snapshot_time = "2016-10-05T00:00:00"
        _stage1_snapshot(sources_root, self._SID, snapshot_time, "pmw_ssmi")

        rows = pd.DataFrame(
            [
                {
                    "sid": self._SID,
                    "source_name": "pmw_ssmi",
                    "time_utc": snapshot_time,
                    "season": 2016,
                    "basin": "AL",
                    "subbasin": "GM",
                }
            ]
        )
        ibtracs_by_sid, sid_attrs, atcf_for_sid = self._fixtures()

        result = assemble_storm(
            self._SID,
            rows,
            sources_root,
            assembled_root,
            skip_existing=False,
            ibtracs_by_sid=ibtracs_by_sid,
            sid_attrs=sid_attrs,
            atcf_for_sid=atcf_for_sid,
        )

        assert result == self._SID
        assert (assembled_root / "storm_data" / f"{self._SID}.h5").exists()
        storm = StormData.from_disk(assembled_root, self._SID)
        assert storm.subbasin == "GM"
        assert storm.season == 2016
        assert storm.atcf_id == "AL102016"
        assert ("pmw_ssmi", snapshot_time) in storm.sources

    def test_skips_storm_without_attrs(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        assembled_root = tmp_path / "assembled"
        rows = pd.DataFrame(columns=["sid", "source_name", "time_utc"])
        # Empty sid_attrs → not in IBTrACS.
        result = assemble_storm(
            "UNKNOWN_SID",
            rows,
            sources_root,
            assembled_root,
            skip_existing=False,
            ibtracs_by_sid={},
            sid_attrs={},
            atcf_for_sid={},
        )
        assert result is None
        assert not (assembled_root / "storm_data").exists()

    def test_ibtracs_only_storm_still_written(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        assembled_root = tmp_path / "assembled"
        ibtracs_by_sid, sid_attrs, atcf_for_sid = self._fixtures()
        empty_rows = pd.DataFrame(columns=["sid", "source_name", "time_utc"])

        result = assemble_storm(
            self._SID,
            empty_rows,
            sources_root,
            assembled_root,
            skip_existing=False,
            ibtracs_by_sid=ibtracs_by_sid,
            sid_attrs=sid_attrs,
            atcf_for_sid=atcf_for_sid,
        )
        assert result == self._SID
        storm = StormData.from_disk(assembled_root, self._SID)
        assert any(name == IBTRACS_SOURCE_NAME for name, _ in storm.sources)


# ---------------------------------------------------------------------------
# build_assembled_index
# ---------------------------------------------------------------------------


class TestBuildAssembledIndex:
    _SID = "2016292N14270"

    def _setup(
        self, tmp_path: Path
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Path, dict[str, str]]:
        sources_root = tmp_path / "sources"
        assembled_root = tmp_path / "assembled"

        snapshot_time = "2016-10-05T00:00:00"
        _stage1_snapshot(sources_root, self._SID, snapshot_time, "pmw_ssmi")

        rows = pd.DataFrame(
            [
                {
                    "sid": self._SID,
                    "source_name": "pmw_ssmi",
                    "time_utc": snapshot_time,
                    "season": 2016,
                    "basin": "AL",
                    "subbasin": "GM",
                }
            ]
        )
        df = _make_ibtracs_df(_make_ibtracs_row(sid=self._SID))
        ibtracs_by_sid = group_ibtracs_by_sid(df)
        sid_attrs = {self._SID: {"season": 2016, "basin": "AL", "subbasin": "GM"}}
        atcf_for_sid = {self._SID: "AL102016"}

        assemble_storm(
            self._SID,
            rows,
            sources_root,
            assembled_root,
            skip_existing=False,
            ibtracs_by_sid=ibtracs_by_sid,
            sid_attrs=sid_attrs,
            atcf_for_sid=atcf_for_sid,
        )

        return df, sid_attrs, assembled_root, atcf_for_sid

    def test_ibtracs_rows_present(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        assert IBTRACS_SOURCE_NAME in result["source_name"].values

    def test_satellite_rows_present(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        assert "pmw_ssmi" in result["source_name"].values

    def test_ibtracs_numeric_channels_excluded_from_index(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        assert "usa_wind" not in result.columns
        ibt_atcf = pd.Series(
            result.loc[result["source_name"] == IBTRACS_SOURCE_NAME, "usa_atcf_id"]
        )
        assert bool(ibt_atcf.notna().all())

    def test_satellite_rows_get_usa_atcf_id_from_translation(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        sat_atcf = pd.Series(result.loc[result["source_name"] == "pmw_ssmi", "usa_atcf_id"])
        assert sat_atcf.iloc[0] == atcf_for_sid[self._SID]

    def test_snapshot_time_renamed_from_iso_time(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        assert "time_utc" in result.columns
        assert "iso_time" not in result.columns

    def test_columns_include_trimmed_schema(self, tmp_path: Path) -> None:
        ibtracs_snapshots, sid_attrs, assembled_root, atcf_for_sid = self._setup(tmp_path)
        result = build_assembled_index(
            ibtracs_snapshots, assembled_root, [self._SID], sid_attrs, atcf_for_sid
        )
        expected = [
            "sid",
            "source_name",
            "time_utc",
            "season",
            "basin",
            "subbasin",
            "usa_atcf_id",
        ]
        assert list(result.columns) == expected
