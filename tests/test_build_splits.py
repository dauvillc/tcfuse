"""Tests for best-track window sample generation in build_splits.py."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from scripts.preprocess.build_splits import build_window_index, split_by_season

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
                "snapshot_time_utc": (init_time + pd.Timedelta(hours=lead_hour)).isoformat(),
                "lat": lat,
                "lon": lon,
                "usa_wind": usa_wind,
                "usa_pres": 970.0,
                "usa_sshs": usa_sshs,
            }
        )
    return pd.DataFrame(rows)


def _build_samples(index: pd.DataFrame) -> pd.DataFrame:
    """Build window samples with the project default test configuration."""
    return build_window_index(
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
