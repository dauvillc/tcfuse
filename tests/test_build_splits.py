"""Tests for best-track window sample generation in build_splits.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts.preprocess.build_splits import build_window_index, split_by_season

SOURCE_NAME = "ibtracs_best_track"
LEADS_HOURS = [0, 6, 12, 18, 24, 30]
REQUIRED_LEADS_HOURS = [0, 6, 30]
REQUIRED_COLUMNS = ["usa_vmax_kt", "lat", "lon"]


def _make_index(
    lead_hours: list[int],
    *,
    storm_id: str = "2016292N14270",
    season: int = 2016,
    nan_lead: int | None = None,
    nan_column: str = "usa_vmax_kt",
) -> pd.DataFrame:
    """Create a synthetic assembled index with ibtracs rows at selected leads."""
    anchor = pd.Timestamp("2016-10-05T00:00:00")
    rows = []
    for lead_hour in lead_hours:
        value = np.nan if lead_hour == nan_lead and nan_column == "usa_vmax_kt" else 65.0
        lat = np.nan if lead_hour == nan_lead and nan_column == "lat" else 15.0
        lon = np.nan if lead_hour == nan_lead and nan_column == "lon" else -60.0
        rows.append(
            {
                "storm_id": storm_id,
                "basin": "AL",
                "season": season,
                "atcf_id": "AL102016",
                "source_name": SOURCE_NAME,
                "snapshot_time_utc": (anchor + pd.Timedelta(hours=lead_hour)).isoformat(),
                "lat": lat,
                "lon": lon,
                "usa_vmax_kt": value,
                "wmo_vmax_kt": 60.0,
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
    index = _make_index([0, 6, 12, 18, 24, 30])
    samples = _build_samples(index)
    assert len(samples) == 1
    assert samples.loc[0, "lead_000h_usa_vmax_kt"] == pytest.approx(65.0)
    assert bool(samples.loc[0, "lead_030h_available"])


def test_missing_optional_lead_is_preserved_as_unavailable() -> None:
    index = _make_index([0, 6, 18, 24, 30])
    samples = _build_samples(index)
    assert len(samples) == 1
    assert not bool(samples.loc[0, "lead_012h_available"])
    assert np.isnan(samples.loc[0, "lead_012h_usa_vmax_kt"])


def test_missing_required_six_hour_lead_rejects_sample() -> None:
    index = _make_index([0, 12, 18, 24, 30])
    samples = _build_samples(index)
    assert samples.empty


def test_missing_required_thirty_hour_lead_rejects_sample() -> None:
    index = _make_index([0, 6, 12, 18, 24])
    samples = _build_samples(index)
    assert samples.empty


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_nan_required_value_rejects_sample(column: str) -> None:
    index = _make_index([0, 6, 12, 18, 24, 30], nan_lead=6, nan_column=column)
    samples = _build_samples(index)
    assert samples.empty


def test_season_split_assigns_samples_to_one_split() -> None:
    train_index = _make_index([0, 6, 12, 18, 24, 30], season=2018)
    val_index = _make_index([0, 6, 12, 18, 24, 30], storm_id="2019292N14270", season=2019)
    test_index = _make_index(
        [0, 6, 12, 18, 24, 30],
        storm_id="2020292N14270",
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
