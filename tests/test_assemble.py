"""Unit tests for IBTrACS helper functions in scripts/preprocess/assemble.py.

All tests use synthetic DataFrames — no real IBTrACS file is required.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess.assemble import (
    _IBTRACS_CHANNELS,
    _IBTRACS_SOURCE_NAME,
    build_assembled_index,
    ibtracs_rows_to_sources,
    load_ibtracs,
)
from tcfuse.data.sources import SourceKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ibtracs_row(
    sid: str = "2016292N14270",
    atcf_id: str = "BAL102016",
    iso_time: str = "2016-10-05 00:00:00",
    lat: float = 15.0,
    lon: float = -60.0,
    usa_wind: float | None = 65.0,
    wmo_wind: float | None = 60.0,
    usa_pres: float | None = 970.0,
    wmo_pres: float | None = 972.0,
    usa_rmw: float | None = 25.0,
    r34_ne: float | None = 120.0,
    r34_se: float | None = 110.0,
    r34_sw: float | None = 100.0,
    r34_nw: float | None = 115.0,
    track_type: str = "MAIN",
) -> dict:
    return {
        "SID": sid,
        "USA_ATCF_ID": atcf_id,
        "ISO_TIME": iso_time,
        "LAT": lat,
        "LON": lon,
        "USA_WIND": usa_wind,
        "WMO_WIND": wmo_wind,
        "USA_PRES": usa_pres,
        "WMO_PRES": wmo_pres,
        "USA_RMW": usa_rmw,
        "USA_R34_NE": r34_ne,
        "USA_R34_SE": r34_se,
        "USA_R34_SW": r34_sw,
        "USA_R34_NW": r34_nw,
        "TRACK_TYPE": track_type,
    }


def _make_ibtracs_df(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)
    return df


# ---------------------------------------------------------------------------
# load_ibtracs
# ---------------------------------------------------------------------------


def _write_ibtracs_csv(tmp_path: Path, data_rows: list[dict]) -> Path:
    """Write a minimal IBTrACS-like CSV (header + units row + data rows)."""
    cols = [
        "SID", "USA_ATCF_ID", "ISO_TIME", "LAT", "LON",
        "USA_WIND", "WMO_WIND", "USA_PRES", "WMO_PRES",
        "USA_RMW", "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
        "TRACK_TYPE",
    ]
    # Units row (row index 1 in the file after the header)
    units_row = {c: "units" for c in cols}
    units_row["ISO_TIME"] = "hours"

    df = pd.DataFrame([units_row] + data_rows, columns=cols)
    path = tmp_path / "ibtracs.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadIbtracs:
    def test_returns_two_dicts(self, tmp_path: Path) -> None:
        row = _make_ibtracs_row()
        path = _write_ibtracs_csv(tmp_path, [row])
        ibtracs_by_sid, atcf_to_sid = load_ibtracs(path)
        assert isinstance(ibtracs_by_sid, dict)
        assert isinstance(atcf_to_sid, dict)

    def test_storm_present_in_by_sid(self, tmp_path: Path) -> None:
        row = _make_ibtracs_row(sid="2016292N14270")
        path = _write_ibtracs_csv(tmp_path, [row])
        ibtracs_by_sid, _ = load_ibtracs(path)
        assert "2016292N14270" in ibtracs_by_sid

    def test_atcf_to_sid_mapping(self, tmp_path: Path) -> None:
        row = _make_ibtracs_row(sid="2016292N14270", atcf_id="BAL102016")
        path = _write_ibtracs_csv(tmp_path, [row])
        _, atcf_to_sid = load_ibtracs(path)
        assert atcf_to_sid["BAL102016"] == "2016292N14270"

    def test_filters_non_main_tracks(self, tmp_path: Path) -> None:
        main_row = _make_ibtracs_row(sid="MAIN_SID", track_type="MAIN")
        spur_row = _make_ibtracs_row(sid="SPUR_SID", atcf_id="BSPUR2016", track_type="spur")
        path = _write_ibtracs_csv(tmp_path, [main_row, spur_row])
        ibtracs_by_sid, _ = load_ibtracs(path)
        assert "MAIN_SID" in ibtracs_by_sid
        assert "SPUR_SID" not in ibtracs_by_sid

    def test_skips_blank_atcf_id(self, tmp_path: Path) -> None:
        # A row with blank USA_ATCF_ID must not pollute atcf_to_sid.
        row = _make_ibtracs_row(sid="NO_ATCF_SID", atcf_id=" ")
        path = _write_ibtracs_csv(tmp_path, [row])
        _, atcf_to_sid = load_ibtracs(path)
        assert "" not in atcf_to_sid
        assert " " not in atcf_to_sid

    def test_iso_time_is_utc_aware(self, tmp_path: Path) -> None:
        row = _make_ibtracs_row()
        path = _write_ibtracs_csv(tmp_path, [row])
        ibtracs_by_sid, _ = load_ibtracs(path)
        sid = list(ibtracs_by_sid.keys())[0]
        ts = ibtracs_by_sid[sid]["ISO_TIME"].iloc[0]
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# ibtracs_rows_to_sources
# ---------------------------------------------------------------------------


class TestIbtracsRowsToSources:
    def test_returns_one_source_per_row(self) -> None:
        df = _make_ibtracs_df(
            _make_ibtracs_row(iso_time="2016-10-05 00:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05 06:00:00"),
        )
        result = ibtracs_rows_to_sources(df, "2016AL10", "AL")
        assert len(result) == 2

    def test_source_kind_is_scalar(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.kind is SourceKind.SCALAR

    def test_source_name(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.source_name == _IBTRACS_SOURCE_NAME

    def test_channels_match_constant(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.channels == _IBTRACS_CHANNELS

    def test_values_dtype_is_float32(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.values.dtype == source.values.dtype  # always true
        import torch
        assert source.values.dtype == torch.float32

    def test_coords_dtype_is_float64(self) -> None:
        import torch
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.coords.dtype == torch.float64

    def test_values_shape(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.values.shape == (9,)

    def test_coords_shape(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.coords.shape == (3,)

    def test_usa_and_wmo_vmax_are_both_present(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=80.0, wmo_wind=70.0))
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert float(source.values[0]) == pytest.approx(80.0)
        assert float(source.values[1]) == pytest.approx(70.0)

    def test_usa_vmax_stays_nan_when_missing_even_if_wmo_exists(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=None, wmo_wind=70.0))
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert np.isnan(float(source.values[0]))
        assert float(source.values[1]) == pytest.approx(70.0)

    def test_wmo_vmax_stays_nan_when_missing_even_if_usa_exists(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=80.0, wmo_wind=None))
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert float(source.values[0]) == pytest.approx(80.0)
        assert np.isnan(float(source.values[1]))

    def test_usa_and_wmo_vmax_nan_when_both_missing(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(usa_wind=None, wmo_wind=None))
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert np.isnan(float(source.values[0]))
        assert np.isnan(float(source.values[1]))

    def test_row_with_nan_lat_is_skipped(self) -> None:
        df = _make_ibtracs_df(
            _make_ibtracs_row(lat=float("nan")),
            _make_ibtracs_row(iso_time="2016-10-05 06:00:00", lat=15.0),
        )
        result = ibtracs_rows_to_sources(df, "2016AL10", "AL")
        assert len(result) == 1

    def test_row_with_nan_lon_is_skipped(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row(lon=float("nan")))
        result = ibtracs_rows_to_sources(df, "2016AL10", "AL")
        assert len(result) == 0

    def test_chronological_order(self) -> None:
        # Rows passed in reverse order must come out sorted by time.
        df = _make_ibtracs_df(
            _make_ibtracs_row(iso_time="2016-10-05 12:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05 00:00:00"),
            _make_ibtracs_row(iso_time="2016-10-05 06:00:00"),
        )
        result = ibtracs_rows_to_sources(df, "2016AL10", "AL")
        times = [t for t, _ in result]
        assert times == sorted(times)

    def test_meta_contains_storm_id(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        _, source = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        assert source.meta["storm_id"] == "2016AL10"

    def test_snapshot_time_utc_key_is_isoformat(self) -> None:
        df = _make_ibtracs_df(_make_ibtracs_row())
        snapshot_time_utc, _ = ibtracs_rows_to_sources(df, "2016AL10", "AL")[0]
        # Must be parseable as a timestamp.
        ts = pd.Timestamp(snapshot_time_utc)
        assert ts is not None


# ---------------------------------------------------------------------------
# build_assembled_index
# ---------------------------------------------------------------------------


class TestBuildAssembledIndex:
    def _make_multi_meta_index(self, storm_id: str = "AL102016") -> pd.DataFrame:
        return pd.DataFrame([
            {
                "storm_id": storm_id,
                "source_name": "pmw_ssmi",
                "snapshot_time_utc": "2016-10-05T00:00:00+00:00",
                "lat": 15.0,
                "lon": -60.0,
                "file_path": "/data/pmw_ssmi/snap.h5",
            }
        ])

    def _make_ibtracs_fixtures(self) -> tuple[dict, dict]:
        rows = _make_ibtracs_df(
            _make_ibtracs_row(sid="2016292N14270", atcf_id="AL102016")
        )
        ibtracs_by_sid = {"2016292N14270": rows}
        atcf_to_sid = {"AL102016": "2016292N14270"}
        return ibtracs_by_sid, atcf_to_sid

    def test_ibtracs_rows_present(self) -> None:
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        assert _IBTRACS_SOURCE_NAME in result["source_name"].values

    def test_non_ibtracs_rows_present(self) -> None:
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        assert "pmw_ssmi" in result["source_name"].values

    def test_atcf_id_populated_from_table(self) -> None:
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        # All rows for matched storms should have the ATCF ID from the table.
        assert (result["atcf_id"] == "AL102016").all()

    def test_unmatched_storm_has_null_atcf_id(self) -> None:
        # Storm EP05 has no IBTrACS match → atcf_id must be None/NaN.
        meta_idx = self._make_multi_meta_index(storm_id="EP052021")
        ibtracs_by_sid: dict = {}
        atcf_to_sid: dict = {}
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["EP052021"])
        assert result["atcf_id"].isna().all()

    def test_ibtracs_rows_have_explicit_vmax_columns(self) -> None:
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        ibt_rows = result[result["source_name"] == _IBTRACS_SOURCE_NAME]
        assert ibt_rows["usa_vmax_kt"].notna().all()
        assert ibt_rows["wmo_vmax_kt"].notna().all()

    def test_non_ibtracs_explicit_vmax_columns_not_filled(self) -> None:
        # Non-IBTrACS rows must not receive best-track wind values implicitly.
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        pmw_rows = result[result["source_name"] == "pmw_ssmi"]
        assert pmw_rows["usa_vmax_kt"].isna().all()
        assert pmw_rows["wmo_vmax_kt"].isna().all()

    def test_output_columns(self) -> None:
        from scripts.preprocess.assemble import _ASSEMBLED_INDEX_COLUMNS
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        assert list(result.columns) == _ASSEMBLED_INDEX_COLUMNS

    def test_season_and_basin_derived(self) -> None:
        meta_idx = self._make_multi_meta_index()
        ibtracs_by_sid, atcf_to_sid = self._make_ibtracs_fixtures()
        result = build_assembled_index(meta_idx, ibtracs_by_sid, atcf_to_sid, ["AL102016"])
        assert (result["basin"] == "AL").all()
        assert (result["season"] == 2016).all()
