"""IBTrACS CSV loading and conversion to :class:`~tcfuse.data.sources.Source` objects."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from tcfuse.data.sources import Source, SourceKind

IBTRACS_SOURCE_NAME = "ibtracs_best_track"
IBTRACS_CHANNELS = [
    "usa_vmax_kt",
    "wmo_vmax_kt",
    "usa_mslp_hpa",
    "wmo_mslp_hpa",
    "usa_rmw_nm",
    "usa_r34_ne_nm",
    "usa_r34_se_nm",
    "usa_r34_sw_nm",
    "usa_r34_nw_nm",
]


def float_or_nan(value: Any) -> float:
    """Return a float value, preserving missing IBTrACS entries as NaN."""
    return float(value) if not pd.isna(value) else np.nan


def load_ibtracs(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load IBTrACS CSV and return rows indexed by SID plus a reverse ATCF lookup."""
    df = pd.read_csv(
        path,
        skiprows=[1],
        na_values=[" "],
        keep_default_na=True,
        low_memory=False,
    )

    track_type = cast(pd.Series, df["TRACK_TYPE"]).astype(str).str.strip().str.lower()
    df = cast(pd.DataFrame, df[track_type == "main"].copy())
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)
    df["SID"] = df["SID"].fillna("").str.strip()
    df["USA_ATCF_ID"] = df["USA_ATCF_ID"].fillna("").str.strip()

    ibtracs_by_sid: dict[str, pd.DataFrame] = {str(sid): grp for sid, grp in df.groupby("SID")}
    atcf_id_col = cast(pd.DataFrame, df[["SID", "USA_ATCF_ID"]]).dropna(subset=["USA_ATCF_ID"])
    atcf_id_col = atcf_id_col[atcf_id_col["USA_ATCF_ID"] != ""]
    atcf_to_sid: dict[str, str] = dict(zip(atcf_id_col["USA_ATCF_ID"], atcf_id_col["SID"]))
    return ibtracs_by_sid, atcf_to_sid


def ibtracs_rows_to_sources(
    storm_rows: pd.DataFrame,
    storm_id: str,
    basin: str,
) -> list[tuple[str, Source]]:
    """Convert IBTrACS rows for one storm into ``(snapshot_time_utc, Source)`` pairs."""
    storm_rows = storm_rows.sort_values("ISO_TIME")
    results: list[tuple[str, Source]] = []

    for _, row in storm_rows.iterrows():
        lat = cast(float, row["LAT"])
        lon = cast(float, row["LON"])
        iso_time = cast(pd.Timestamp, row["ISO_TIME"])

        if pd.isna(lat) or pd.isna(lon):
            warnings.warn(
                f"IBTrACS row for {storm_id} at {iso_time} has NaN lat/lon — skipped.",
                stacklevel=2,
            )
            continue

        usa_vmax_kt = float_or_nan(row.get("USA_WIND", np.nan))
        wmo_vmax_kt = float_or_nan(row.get("WMO_WIND", np.nan))
        usa_mslp_hpa = float_or_nan(row.get("USA_PRES", np.nan))
        wmo_mslp_hpa = float_or_nan(row.get("WMO_PRES", np.nan))
        usa_rmw_nm = float_or_nan(row.get("USA_RMW", np.nan))
        usa_r34_ne = float_or_nan(row.get("USA_R34_NE", np.nan))
        usa_r34_se = float_or_nan(row.get("USA_R34_SE", np.nan))
        usa_r34_sw = float_or_nan(row.get("USA_R34_SW", np.nan))
        usa_r34_nw = float_or_nan(row.get("USA_R34_NW", np.nan))

        values = torch.tensor(
            [
                usa_vmax_kt,
                wmo_vmax_kt,
                usa_mslp_hpa,
                wmo_mslp_hpa,
                usa_rmw_nm,
                usa_r34_ne,
                usa_r34_se,
                usa_r34_sw,
                usa_r34_nw,
            ],
            dtype=torch.float32,
        )
        time_unix_s = float(iso_time.timestamp())
        coords = torch.tensor([time_unix_s, float(lat), float(lon)], dtype=torch.float64)
        snapshot_time_utc = iso_time.replace(tzinfo=None).isoformat()

        source = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=coords,
            source_name=IBTRACS_SOURCE_NAME,
            channels=IBTRACS_CHANNELS,
            mask=torch.isfinite(values),
            meta={
                "storm_id": storm_id,
                "basin": basin,
                "snapshot_time_utc": snapshot_time_utc,
                "lat": float(lat),
                "lon": float(lon),
                "usa_vmax_kt": usa_vmax_kt,
                "wmo_vmax_kt": wmo_vmax_kt,
                "usa_mslp_hpa": usa_mslp_hpa,
                "wmo_mslp_hpa": wmo_mslp_hpa,
            },
        )
        results.append((snapshot_time_utc, source))

    return results
