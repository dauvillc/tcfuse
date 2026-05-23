"""Tests for tcfuse.utils.time helpers."""

from __future__ import annotations

import pandas as pd

from tcfuse.utils.time import lead_hours_rounded, to_compact_time

_TIME_ISO = "2016-09-12T01:09:42+00:00"
_TIME_COMPACT = "20160912T010942Z"


def test_to_compact_time_from_iso_string() -> None:
    assert to_compact_time(_TIME_ISO) == _TIME_COMPACT


def test_to_compact_time_from_timestamp() -> None:
    ts = pd.Timestamp(_TIME_ISO)
    assert isinstance(ts, pd.Timestamp)
    assert to_compact_time(ts) == _TIME_COMPACT


def test_to_compact_time_from_unix_seconds() -> None:
    ref = pd.Timestamp(_TIME_ISO)
    assert isinstance(ref, pd.Timestamp)
    assert to_compact_time(int(ref.timestamp()), unit="s") == _TIME_COMPACT


def test_lead_hours_rounded() -> None:
    init = "2016-09-12T00:00:00+00:00"
    snap = "2016-09-12T06:30:00+00:00"
    assert lead_hours_rounded(init, snap) == 6
