"""Tests for season splits and legacy forecast-window sample construction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scripts.preprocess.build_splits import split_by_season

from tcfuse.utils.time import to_compact_time

SOURCE_NAME = "ibtracs_best_track"
LEADS_HOURS = [-6, 0, 6, 12, 18, 24]
REQUIRED_LEADS_HOURS = [-6, 0, 24]
REQUIRED_COLUMNS = ["usa_wind", "usa_sshs", "lat", "lon"]
INIT_TIME = cast(pd.Timestamp, pd.Timestamp("2016-10-05T06:00:00"))


def _make_index(
    lead_hours: list[int],
    *,
    sid: str = "2016292N14270",
    season: int = 2016,
    init_time: pd.Timestamp = INIT_TIME,
    nan_lead: int | None = None,
    nan_column: str = "usa_wind",
) -> pd.DataFrame:
    """Create a synthetic assembled index with ibtracs rows at selected leads."""
    rows = []
    for lead_hour in lead_hours:
        usa_wind = np.nan if lead_hour == nan_lead and nan_column == "usa_wind" else 65.0
        usa_sshs = np.nan if lead_hour == nan_lead and nan_column == "usa_sshs" else 2.0
        lat = np.nan if lead_hour == nan_lead and nan_column == "lat" else 15.0
        lon = np.nan if lead_hour == nan_lead and nan_column == "lon" else -60.0
        rows.append(
            {
                "sid": sid,
                "basin": "AL",
                "subbasin": "GM",
                "season": season,
                "usa_atcf_id": "AL102016",
                "source_name": SOURCE_NAME,
                "time_utc": (init_time + pd.Timedelta(hours=lead_hour)).isoformat(),
                "lat": lat,
                "lon": lon,
                "usa_wind": usa_wind,
                "usa_pres": 970.0,
                "usa_sshs": usa_sshs,
            }
        )
    return pd.DataFrame(rows)


def _parse_time(value: Any) -> pd.Timestamp:
    """Parse a timestamp value as UTC for exact lead-time matching."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize("UTC"))
    return cast(pd.Timestamp, timestamp.tz_convert("UTC"))


def _isoformat_naive_utc(value: Any) -> str:
    """Return the repository's naive-UTC ISO timestamp representation."""
    timestamp = _parse_time(value)
    return timestamp.tz_convert("UTC").tz_localize(None).isoformat()


def _lead_prefix(lead_hour: int) -> str:
    """Return the fixed column prefix for a lead relative to init time."""
    return f"lead_{lead_hour:+04d}h"


def _float_or_nan(value: Any) -> float:
    """Convert a scalar value to float while preserving missing values as NaN."""
    return np.nan if pd.isna(value) else float(value)


def _empty_forecast_sample_index(
    leads_hours: list[int], required_columns: list[str]
) -> pd.DataFrame:
    """Return an empty sample-index DataFrame with the expected schema."""
    columns = [
        "sample_id",
        "sid",
        "basin",
        "subbasin",
        "season",
        "usa_atcf_id",
        "init_time_utc",
        "window_start_time_utc",
        "window_end_time_utc",
    ]
    for lead_hour in leads_hours:
        prefix = _lead_prefix(lead_hour)
        columns.extend([f"{prefix}_time_utc", f"{prefix}_available"])
        columns.extend(f"{prefix}_{column}" for column in required_columns)
    return pd.DataFrame(columns=columns)


def _is_finite(row: pd.Series | None, columns: list[str]) -> bool:
    """Return True when all requested columns are present and finite in a row."""
    if row is None:
        return False
    for column in columns:
        value = row.at[column] if column in row.index else np.nan
        if bool(pd.isna(value)):
            return False
        if not np.isfinite(float(cast(float | int, value))):
            return False
    return True


def _first_rows_by_time(rows: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Index a storm's best-track rows by timestamp, keeping the first duplicate."""
    indexed: dict[pd.Timestamp, pd.Series] = {}
    sort_column = "_time" if "_time" in rows.columns else "time_utc"
    for _, row in rows.sort_values(sort_column).iterrows():
        timestamp = _parse_time(row["time_utc"])
        indexed.setdefault(timestamp, row)
    return indexed


def build_forecast_sample_index(
    assembled_index: pd.DataFrame,
    source_name: str,
    leads_hours: list[int],
    required_leads_hours: list[int],
    required_columns: list[str],
) -> pd.DataFrame:
    """Build one wide-format sample row per valid best-track forecast window."""
    if not set(required_leads_hours).issubset(set(leads_hours)):
        raise ValueError("required_leads_hours must be a subset of leads_hours.")

    best_track = assembled_index[assembled_index["source_name"] == source_name].copy()
    if best_track.empty:
        return _empty_forecast_sample_index(leads_hours, required_columns)

    # Parse timestamps once so sorting and lead matching use a consistent timezone.
    best_track["_time"] = cast(pd.Series, best_track["time_utc"]).map(_parse_time)
    best_track = cast(Any, best_track).sort_values(["sid", "_time"]).reset_index(drop=True)

    sample_rows: list[dict[str, Any]] = []
    for sid_value, storm_rows in best_track.groupby("sid", sort=True):
        sid = str(sid_value)
        rows_by_time = _first_rows_by_time(storm_rows)
        # Every distinct best-track time is a candidate assimilation anchor t₀.
        for init_time in sorted(rows_by_time):
            lead_rows: dict[int, pd.Series | None] = {}
            for lead_hour in leads_hours:
                lead_time = cast(pd.Timestamp, init_time + pd.Timedelta(hours=lead_hour))
                lead_rows[lead_hour] = rows_by_time.get(lead_time)

            # Required leads must be finite; optional leads may be absent (see _available).
            if not all(
                _is_finite(lead_rows[lead_hour], required_columns)
                for lead_hour in required_leads_hours
            ):
                continue

            init_row = rows_by_time[init_time]
            sample_id = f"{sid}_{to_compact_time(init_time)}"
            sample: dict[str, Any] = {
                "sample_id": sample_id,
                "sid": sid,
                "basin": init_row.get("basin"),
                "subbasin": init_row.get("subbasin"),
                "season": int(init_row.at["season"]),
                "usa_atcf_id": init_row.get("usa_atcf_id"),
                "init_time_utc": _isoformat_naive_utc(init_time),
                "window_start_time_utc": _isoformat_naive_utc(
                    init_time + pd.Timedelta(hours=min(leads_hours))
                ),
                "window_end_time_utc": _isoformat_naive_utc(
                    init_time + pd.Timedelta(hours=max(leads_hours))
                ),
            }

            # Fixed lead columns are easy to consume from Parquet in PyTorch datasets.
            for lead_hour in leads_hours:
                prefix = _lead_prefix(lead_hour)
                lead_time = init_time + pd.Timedelta(hours=lead_hour)
                row = lead_rows[lead_hour]
                sample[f"{prefix}_time_utc"] = _isoformat_naive_utc(lead_time)
                # _available distinguishes missing optional leads from NaN channel values.
                sample[f"{prefix}_available"] = row is not None
                for column in required_columns:
                    sample[f"{prefix}_{column}"] = (
                        _float_or_nan(row[column])
                        if row is not None and column in row.index
                        else np.nan
                    )

            sample_rows.append(sample)

    if not sample_rows:
        return _empty_forecast_sample_index(leads_hours, required_columns)
    return pd.DataFrame(sample_rows).sort_values(["sid", "init_time_utc"]).reset_index(drop=True)


def _build_samples(index: pd.DataFrame) -> pd.DataFrame:
    """Build window samples with the project default test configuration."""
    return build_forecast_sample_index(
        index,
        source_name=SOURCE_NAME,
        leads_hours=LEADS_HOURS,
        required_leads_hours=REQUIRED_LEADS_HOURS,
        required_columns=REQUIRED_COLUMNS,
    )


def test_complete_six_lead_window_creates_one_sample() -> None:
    index = _make_index(LEADS_HOURS)
    samples = _build_samples(index)
    assert len(samples) == 1
    assert samples.loc[0, "lead_+000h_usa_wind"] == pytest.approx(65.0)
    assert bool(samples.loc[0, "lead_+024h_available"])
    assert samples.loc[0, "init_time_utc"] == INIT_TIME.isoformat()


def test_missing_optional_lead_is_preserved_as_unavailable() -> None:
    index = _make_index([-6, 0, 6, 18, 24])
    samples = _build_samples(index)
    assert len(samples) == 1
    assert not bool(samples.loc[0, "lead_+012h_available"])
    assert np.isnan(samples.loc[0, "lead_+012h_usa_wind"])


def test_missing_required_zero_hour_lead_rejects_sample() -> None:
    index = _make_index([-6, 6, 12, 18, 24])
    samples = _build_samples(index)
    assert samples.empty


def test_missing_required_twenty_four_hour_lead_rejects_sample() -> None:
    index = _make_index([-6, 0, 6, 12, 18])
    samples = _build_samples(index)
    assert samples.empty


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_nan_required_value_rejects_sample(column: str) -> None:
    index = _make_index(LEADS_HOURS, nan_lead=0, nan_column=column)
    samples = _build_samples(index)
    assert samples.empty


def test_sample_row_carries_sid_and_subbasin() -> None:
    index = _make_index(LEADS_HOURS)
    samples = _build_samples(index)
    assert samples.loc[0, "sid"] == "2016292N14270"
    assert samples.loc[0, "subbasin"] == "GM"
    assert samples.loc[0, "usa_atcf_id"] == "AL102016"


def test_window_bounds_span_assimilation_window() -> None:
    index = _make_index(LEADS_HOURS)
    samples = _build_samples(index)
    assert (
        samples.loc[0, "window_start_time_utc"] == (INIT_TIME - pd.Timedelta(hours=6)).isoformat()
    )
    assert samples.loc[0, "window_end_time_utc"] == (INIT_TIME + pd.Timedelta(hours=24)).isoformat()


def test_season_split_assigns_samples_to_one_split() -> None:
    train_index = _make_index(LEADS_HOURS, season=2018)
    val_index = _make_index(LEADS_HOURS, sid="2019292N14270", season=2019)
    test_index = _make_index(
        LEADS_HOURS,
        sid="2020292N14270",
        season=2020,
    )
    samples = _build_samples(pd.concat([train_index, val_index, test_index], ignore_index=True))

    splits = split_by_season(samples, val_seasons={2019}, test_seasons={2020})

    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert set(splits["train"]["season"]) == {2018}
    assert set(splits["val"]["season"]) == {2019}
    assert set(splits["test"]["season"]) == {2020}
