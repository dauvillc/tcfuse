"""Schema and pivot tests for the tidy-long IBTrACS prediction table."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from tcfuse.data.predictions import (
    IBTRACS_LONG_COLUMNS,
    build_long_rows,
    empty_long_frame,
    ibtracs_long_schema,
    long_to_pivot,
)

_SAMPLE_ID = "2016AL10_20160912T000000Z"
_STORM_ID = "2016AL10"
_INIT_TIME = "2016-09-12T00:00:00"
_BASIN = "AL"
_SEASON = 2016
_CHANNELS = ["usa_vmax_kt", "usa_mslp_hpa"]


def _build_sample_block() -> pd.DataFrame:
    """Build a tidy-long block with three leads and two channels."""
    leads = [
        {
            "lead_hour": 0,
            "valid_time_utc": "2016-09-12T00:00:00",
            "pred": {"usa_vmax_kt": 65.0, "usa_mslp_hpa": 985.0},
            "target": {"usa_vmax_kt": 70.0, "usa_mslp_hpa": 980.0},
        },
        {
            "lead_hour": 6,
            "valid_time_utc": "2016-09-12T06:00:00",
            "pred": {"usa_vmax_kt": 80.0, "usa_mslp_hpa": 970.0},
            # Targets missing for this lead; mask should be False everywhere.
            "target": None,
        },
        {
            "lead_hour": 12,
            "valid_time_utc": "2016-09-12T12:00:00",
            "pred": {"usa_vmax_kt": 95.0, "usa_mslp_hpa": float("nan")},
            "target": {"usa_vmax_kt": 100.0, "usa_mslp_hpa": 950.0},
        },
    ]
    return build_long_rows(
        sample_id=_SAMPLE_ID,
        storm_id=_STORM_ID,
        season=_SEASON,
        basin=_BASIN,
        init_time_utc=_INIT_TIME,
        leads=leads,
        channels=_CHANNELS,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestIbtracsLongSchema:
    def test_schema_columns_match_canonical_order(self) -> None:
        schema = ibtracs_long_schema()
        assert schema.names == IBTRACS_LONG_COLUMNS

    def test_empty_frame_columns(self) -> None:
        empty = empty_long_frame()
        assert list(empty.columns) == IBTRACS_LONG_COLUMNS
        assert len(empty) == 0


# ---------------------------------------------------------------------------
# build_long_rows: row count, masking, NaN handling
# ---------------------------------------------------------------------------


class TestBuildLongRows:
    def test_row_count_equals_leads_times_channels(self) -> None:
        frame = _build_sample_block()
        # 3 leads x 2 channels = 6 rows.
        assert len(frame) == 6

    def test_columns_match_canonical(self) -> None:
        frame = _build_sample_block()
        assert list(frame.columns) == IBTRACS_LONG_COLUMNS

    def test_dtypes_match_canonical(self) -> None:
        frame = _build_sample_block()
        assert pd.api.types.is_integer_dtype(frame["lead_hour"])
        assert pd.api.types.is_integer_dtype(frame["season"])
        assert pd.api.types.is_float_dtype(frame["pred"])
        assert pd.api.types.is_float_dtype(frame["target"])
        assert pd.api.types.is_bool_dtype(frame["mask"])

    def test_mask_true_when_both_finite(self) -> None:
        frame = _build_sample_block()
        row_lead0_vmax = cast(
            pd.DataFrame,
            frame[(frame["lead_hour"] == 0) & (frame["channel"] == "usa_vmax_kt")],
        )
        assert bool(row_lead0_vmax["mask"].iloc[0])

    def test_mask_false_when_target_missing(self) -> None:
        frame = _build_sample_block()
        rows_lead6 = cast(pd.DataFrame, frame[frame["lead_hour"] == 6])
        # All channels at lead 6 have target=None, so mask must be False everywhere.
        assert not bool(rows_lead6["mask"].any())
        assert bool(rows_lead6["target"].isna().all())
        assert bool(rows_lead6["pred"].notna().all())

    def test_mask_false_when_pred_is_nan(self) -> None:
        frame = _build_sample_block()
        row = cast(
            pd.DataFrame,
            frame[(frame["lead_hour"] == 12) & (frame["channel"] == "usa_mslp_hpa")],
        )
        assert math.isnan(row["pred"].iloc[0])
        assert not bool(row["mask"].iloc[0])

    def test_carries_metadata_on_every_row(self) -> None:
        frame = _build_sample_block()
        assert (frame["sample_id"] == _SAMPLE_ID).all()
        assert (frame["storm_id"] == _STORM_ID).all()
        assert (frame["season"] == _SEASON).all()
        assert (frame["basin"] == _BASIN).all()
        assert (frame["init_time_utc"] == _INIT_TIME).all()

    def test_lead_hour_to_valid_time_pairing_unique(self) -> None:
        frame = _build_sample_block()
        pairs = frame[["lead_hour", "valid_time_utc"]].drop_duplicates()
        # Each lead corresponds to exactly one valid time.
        assert len(pairs) == 3

    def test_empty_leads_yields_empty_frame(self) -> None:
        frame = build_long_rows(
            sample_id=_SAMPLE_ID,
            storm_id=_STORM_ID,
            season=_SEASON,
            basin=_BASIN,
            init_time_utc=_INIT_TIME,
            leads=[],
            channels=_CHANNELS,
        )
        assert frame.empty
        assert list(frame.columns) == IBTRACS_LONG_COLUMNS

    def test_channels_preserved_in_caller_order(self) -> None:
        leads = [
            {
                "lead_hour": 0,
                "valid_time_utc": _INIT_TIME,
                "pred": {"usa_vmax_kt": 1.0, "usa_mslp_hpa": 2.0},
                "target": {"usa_vmax_kt": 1.0, "usa_mslp_hpa": 2.0},
            }
        ]
        frame = build_long_rows(
            sample_id=_SAMPLE_ID,
            storm_id=_STORM_ID,
            season=_SEASON,
            basin=_BASIN,
            init_time_utc=_INIT_TIME,
            leads=leads,
            channels=_CHANNELS,
        )
        assert frame["channel"].tolist() == _CHANNELS


# ---------------------------------------------------------------------------
# Pivot round-trip back to a wide build_splits-style frame
# ---------------------------------------------------------------------------


class TestLongToPivot:
    def test_pivot_produces_expected_columns(self) -> None:
        frame = _build_sample_block()
        wide = long_to_pivot(frame)
        # The sample_id row should have lead/channel/value columns for every valid combo.
        expected = {
            "lead_000h_usa_vmax_kt_pred",
            "lead_000h_usa_vmax_kt_target",
            "lead_000h_usa_mslp_hpa_pred",
            "lead_000h_usa_mslp_hpa_target",
            "lead_006h_usa_vmax_kt_pred",
            "lead_006h_usa_mslp_hpa_pred",
            "lead_012h_usa_vmax_kt_pred",
            "lead_012h_usa_vmax_kt_target",
            "lead_012h_usa_mslp_hpa_target",
        }
        assert expected.issubset(set(wide.columns))

    def test_pivot_values_round_trip(self) -> None:
        frame = _build_sample_block()
        wide = long_to_pivot(frame)
        row = wide.iloc[0]
        assert row["lead_000h_usa_vmax_kt_pred"] == 65.0
        assert row["lead_000h_usa_vmax_kt_target"] == 70.0
        assert row["lead_012h_usa_vmax_kt_target"] == 100.0

    def test_pivot_index_columns(self) -> None:
        frame = _build_sample_block()
        wide = long_to_pivot(frame)
        for column in ("sample_id", "storm_id", "season", "basin", "init_time_utc"):
            assert column in wide.columns
        assert wide["sample_id"].iloc[0] == _SAMPLE_ID

    def test_pivot_empty_frame_yields_empty(self) -> None:
        wide = long_to_pivot(empty_long_frame())
        assert wide.empty

    def test_missing_target_propagates_as_nan_in_wide(self) -> None:
        frame = _build_sample_block()
        wide = long_to_pivot(frame)
        # Lead-6 had no targets, so the wide target columns should be NaN.
        assert np.isnan(wide["lead_006h_usa_vmax_kt_target"].iloc[0])
        assert np.isnan(wide["lead_006h_usa_mslp_hpa_target"].iloc[0])
