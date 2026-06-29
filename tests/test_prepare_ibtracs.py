"""Unit tests for scripts/preprocess/prepare_ibtracs.py.

Each test writes a synthetic IBTrACS-style CSV (with the units row) and
verifies that ``preprocess_ibtracs`` produces the expected parquet schema
and translation table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scripts.preprocess.prepare_ibtracs import preprocess_ibtracs, write_outputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RAW_COLUMNS: list[str] = [
    "SID",
    "USA_ATCF_ID",
    "BASIN",
    "SUBBASIN",
    "SEASON",
    "NAME",
    "NUMBER",
    "NATURE",
    "ISO_TIME",
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "USA_SSHS",
    "TRACK_TYPE",
    "USA_R34_NE",
    "USA_R34_SE",
    "USA_R34_SW",
    "USA_R34_NW",
    "USA_R50_NE",
    "USA_R50_SE",
    "USA_R50_SW",
    "USA_R50_NW",
    "USA_R64_NE",
    "USA_R64_SE",
    "USA_R64_SW",
    "USA_R64_NW",
]


def _make_raw_row(
    sid: str = "2016292N14270",
    atcf_id: str = "AL102016",
    iso_time: str = "2016-10-05 00:00:00",
    lat: float = 15.0,
    lon: float = -60.0,
    basin: str = "AL",
    subbasin: str = "GM",
    season: int = 2016,
    usa_wind: float | None = 65.0,
    usa_pres: float | None = 970.0,
    usa_sshs: int | None = 2,
    nature: str = "TS",
    name: str = "MATTHEW",
    number: int = 14,
    track_type: str = "MAIN",
) -> dict[str, Any]:
    """Build a single raw-CSV-style row dict (all radii set to a finite default)."""
    row: dict[str, Any] = {
        "SID": sid,
        "USA_ATCF_ID": atcf_id,
        "BASIN": basin,
        "SUBBASIN": subbasin,
        "SEASON": season,
        "NAME": name,
        "NUMBER": number,
        "NATURE": nature,
        "ISO_TIME": iso_time,
        "LAT": lat,
        "LON": lon,
        "USA_WIND": usa_wind,
        "USA_PRES": usa_pres,
        "USA_SSHS": usa_sshs,
        "TRACK_TYPE": track_type,
    }
    for prefix in ("USA_R34", "USA_R50", "USA_R64"):
        for quad in ("NE", "SE", "SW", "NW"):
            row[f"{prefix}_{quad}"] = 100.0
    return row


def _write_ibtracs_csv(tmp_path: Path, data_rows: list[dict[str, Any]]) -> Path:
    """Write a synthetic IBTrACS CSV with a units row after the header."""
    units_row = {c: "units" for c in _RAW_COLUMNS}
    units_row["ISO_TIME"] = "hours"
    df = pd.DataFrame([units_row, *data_rows], columns=_RAW_COLUMNS)
    path = tmp_path / "ibtracs.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# preprocess_ibtracs
# ---------------------------------------------------------------------------


class TestPreprocessIbtracs:
    def test_main_filter_keeps_only_main_track(self, tmp_path: Path) -> None:
        rows = [
            _make_raw_row(sid="MAIN_SID", track_type="MAIN"),
            _make_raw_row(sid="SPUR_SID", atcf_id="AL112016", track_type="spur"),
        ]
        csv = _write_ibtracs_csv(tmp_path, rows)
        snapshots, atcf_to_sid = preprocess_ibtracs(csv)
        assert "MAIN_SID" in snapshots["sid"].values
        assert "SPUR_SID" not in snapshots["sid"].values
        assert "AL112016" not in atcf_to_sid["usa_atcf_id"].values

    def test_atcf_to_sid_drops_blank_atcf(self, tmp_path: Path) -> None:
        rows = [
            _make_raw_row(sid="A", atcf_id="AL102016"),
            _make_raw_row(sid="B", atcf_id=" "),
        ]
        csv = _write_ibtracs_csv(tmp_path, rows)
        _, atcf_to_sid = preprocess_ibtracs(csv)
        assert list(atcf_to_sid["usa_atcf_id"]) == ["AL102016"]

    def test_duplicate_sid_to_atcf_resolves_by_max_wind(self, tmp_path: Path) -> None:
        rows = [
            _make_raw_row(sid="SID1", atcf_id="AL102016", usa_wind=30),
            _make_raw_row(sid="SID1", atcf_id="AL202016", usa_wind=80),
        ]
        csv = _write_ibtracs_csv(tmp_path, rows)
        _, atcf_to_sid = preprocess_ibtracs(csv)
        assert len(atcf_to_sid) == 1
        assert atcf_to_sid.iloc[0]["usa_atcf_id"] == "AL202016"

    def test_multiple_atcf_to_one_sid_is_fine(self, tmp_path: Path) -> None:
        # A single ATCF mapping to two SIDs is NOT raised by the new check
        # (the previous check was the reverse direction).
        rows = [
            _make_raw_row(sid="SID1", atcf_id="AL102016"),
            _make_raw_row(sid="SID2", atcf_id="AL102016"),
        ]
        csv = _write_ibtracs_csv(tmp_path, rows)
        _, atcf_to_sid = preprocess_ibtracs(csv)
        assert set(atcf_to_sid["sid"]) == {"SID1", "SID2"}

    def test_canonical_column_names(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row()])
        snapshots, _ = preprocess_ibtracs(csv)
        # Spot-check the full list of canonical lowercased columns.
        expected = {
            "sid",
            "season",
            "basin",
            "subbasin",
            "name",
            "number",
            "iso_time",
            "nature",
            "lat",
            "lon",
            "usa_atcf_id",
            "usa_wind",
            "usa_pres",
            "usa_sshs",
            "usa_r34_ne",
            "usa_r34_se",
            "usa_r34_sw",
            "usa_r34_nw",
            "usa_r50_ne",
            "usa_r50_se",
            "usa_r50_sw",
            "usa_r50_nw",
            "usa_r64_ne",
            "usa_r64_se",
            "usa_r64_sw",
            "usa_r64_nw",
        }
        assert expected.issubset(set(snapshots.columns))

    def test_translation_table_columns(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row()])
        _, atcf_to_sid = preprocess_ibtracs(csv)
        assert set(atcf_to_sid.columns) == {
            "sid",
            "season",
            "basin",
            "subbasin",
            "name",
            "usa_atcf_id",
        }

    def test_dtypes(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row()])
        snapshots, _ = preprocess_ibtracs(csv)
        assert snapshots["season"].dtype == "int64"
        assert snapshots["usa_sshs"].dtype == pd.Int64Dtype()
        assert snapshots["number"].dtype == pd.Int64Dtype()
        assert snapshots["usa_wind"].dtype.kind == "f"
        assert snapshots["usa_r34_ne"].dtype.kind == "f"
        assert snapshots["lat"].dtype.kind == "f"

    def test_iso_time_is_naive_iso_string(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row(iso_time="2016-10-05 06:30:00")])
        snapshots, _ = preprocess_ibtracs(csv)
        assert snapshots["iso_time"].iloc[0] == "2016-10-05T06:30:00"

    def test_missing_usa_wind_becomes_nan(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row(usa_wind=None)])
        snapshots, _ = preprocess_ibtracs(csv)
        assert pd.isna(snapshots["usa_wind"].iloc[0])

    def test_missing_usa_sshs_becomes_na(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row(usa_sshs=None)])
        snapshots, _ = preprocess_ibtracs(csv)
        assert pd.isna(snapshots["usa_sshs"].iloc[0])

    def test_north_atlantic_basin_not_coerced_to_nan(self, tmp_path: Path) -> None:
        # The IBTrACS basin code "NA" (North Atlantic) is in pandas' default
        # na_values; it must survive Stage 0 as the literal string "NA" in both
        # the snapshots table and the ATCF→SID translation table.
        csv = _write_ibtracs_csv(tmp_path, [_make_raw_row(basin="NA", subbasin="NA")])
        snapshots, atcf_to_sid = preprocess_ibtracs(csv)
        assert snapshots["basin"].iloc[0] == "NA"
        assert snapshots["subbasin"].iloc[0] == "NA"
        assert atcf_to_sid["basin"].iloc[0] == "NA"
        assert atcf_to_sid["subbasin"].iloc[0] == "NA"

    def test_multiple_storms_sorted(self, tmp_path: Path) -> None:
        rows = [
            _make_raw_row(sid="SID_B", atcf_id="AL202016", iso_time="2016-10-06 00:00:00"),
            _make_raw_row(sid="SID_A", atcf_id="AL102016", iso_time="2016-10-05 00:00:00"),
        ]
        csv = _write_ibtracs_csv(tmp_path, rows)
        snapshots, _ = preprocess_ibtracs(csv)
        # Sorted by (sid, iso_time).
        assert list(snapshots["sid"]) == ["SID_A", "SID_B"]


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------


class TestWriteOutputs:
    def test_round_trip_via_disk(self, tmp_path: Path) -> None:
        csv = _write_ibtracs_csv(
            tmp_path,
            [
                _make_raw_row(sid="A", atcf_id="AL102016"),
                _make_raw_row(sid="B", atcf_id="AL202016", iso_time="2016-10-06 00:00:00"),
            ],
        )
        snapshots, atcf_to_sid = preprocess_ibtracs(csv)
        out_dir = tmp_path / "out"
        write_outputs(snapshots, atcf_to_sid, out_dir)

        atcf_path = out_dir / "atcf_to_sid.csv"
        pq_path = out_dir / "ibtracs.parquet"
        assert atcf_path.exists()
        assert pq_path.exists()

        atcf_back = pd.read_csv(atcf_path)
        assert set(atcf_back.columns) == {
            "sid",
            "season",
            "basin",
            "subbasin",
            "name",
            "usa_atcf_id",
        }
        assert len(atcf_back) == 2

        snapshots_back = pd.read_parquet(pq_path)
        assert len(snapshots_back) == 2
        assert set(snapshots_back["sid"]) == {"A", "B"}
