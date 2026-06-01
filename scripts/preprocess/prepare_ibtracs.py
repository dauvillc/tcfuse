#!/usr/bin/env python3
"""Stage 0 — preprocess the raw IBTrACS CSV into the two artifacts consumed downstream.

Outputs (under ``${paths.preprocessed_sources}/ibtracs/``):

- ``ibtracs.parquet`` — one row per (SID, ISO_TIME) after the ``TRACK_TYPE == "MAIN"``
  filter, with all columns lowercased to their canonical IBTrACS names.
- ``atcf_to_sid.csv`` — translation table with columns
  ``sid, season, basin, subbasin, name, usa_atcf_id``. Used by every Stage 1
  preprocessor to translate ATCF storm identifiers into IBTrACS SIDs and to
  enrich the per-source index with ``season / basin / subbasin``.

Rows strictly before 1987-01-01 are excluded. When a SID maps to multiple
USA_ATCF_ID values, the mapping keeps the ATCF ID with the highest USA_WIND.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig

from scripts.preprocess.utils.runner import resolve_preproc_cfg

# Output directory under ${paths.preprocessed_sources}.
_IBTRACS_DIR_NAME = "ibtracs"

# IBTrACS wind-radius quadrants (per threshold) used to expand R34/R50/R64.
_RADIUS_QUADRANTS = ("NE", "SE", "SW", "NW")
_RADIUS_THRESHOLDS = ("USA_R34", "USA_R50", "USA_R64")

# Raw CSV column names read from the IBTrACS file (second row is units — skipped).
_RAW_COLUMNS = [
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
    *[f"{threshold}_{quad}" for threshold in _RADIUS_THRESHOLDS for quad in _RADIUS_QUADRANTS],
]

# Lowercased radius column names written to ibtracs.parquet.
_RADIUS_OUTPUT_COLUMNS = [
    f"{t.lower()}_{q.lower()}" for t in _RADIUS_THRESHOLDS for q in _RADIUS_QUADRANTS
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return ``df[name]`` typed as a Series (helper for stricter type checkers)."""
    return cast(pd.Series, df[name])


def _coerce_string(series: pd.Series) -> pd.Series:
    """Coerce a Series to stripped strings; missing entries become empty string."""
    return cast(pd.Series, series.fillna("").astype(str).str.strip())


def _coerce_nullable_int(series: pd.Series) -> pd.Series:
    """Coerce a Series to pandas nullable Int64; non-numeric entries become NA."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return cast(pd.Series, numeric.astype("Int64"))


def _coerce_float(series: pd.Series) -> pd.Series:
    """Coerce a Series to float64; non-numeric entries become NaN."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return cast(pd.Series, numeric.astype(float))


def preprocess_ibtracs(ibtracs_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the raw IBTrACS CSV into the two output DataFrames."""
    # Row 2 of the IBTrACS CSV is units — skip it on read.
    df = pd.read_csv(
        ibtracs_csv,
        skiprows=[1],
        usecols=_RAW_COLUMNS,
        na_values=[" "],
        keep_default_na=True,
        low_memory=False,
    )

    # Keep main track only (exclude spurs, merges, etc.).
    track_type = _col(df, "TRACK_TYPE").astype(str).str.strip().str.lower()
    df = cast(pd.DataFrame, df[track_type == "main"].copy())

    # Normalize string ids; empty string means missing ATCF id.
    df["SID"] = _coerce_string(_col(df, "SID"))
    df["USA_ATCF_ID"] = _coerce_string(_col(df, "USA_ATCF_ID"))

    # Filter to 1987+ and store iso_time as naive UTC strings.
    iso_time_utc = cast(pd.Series, pd.to_datetime(_col(df, "ISO_TIME"), utc=True))
    min_iso_time = pd.Timestamp("1987-01-01T00:00:00Z")
    df = cast(pd.DataFrame, df[iso_time_utc >= min_iso_time].copy())
    iso_time_utc = cast(pd.Series, iso_time_utc[iso_time_utc >= min_iso_time])
    iso_time_str = iso_time_utc.dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # SEASON is required for train/val/test split assignment — fail if any row lacks it.
    season_int = _coerce_nullable_int(_col(df, "SEASON"))
    if bool(season_int.isna().any()):
        n_missing = int(season_int.isna().sum())
        raise ValueError(
            f"IBTrACS CSV has {n_missing} main-track row(s) with missing SEASON; "
            "cannot derive a per-storm split season from these rows."
        )

    # Build the snapshots table (one row per sid x iso_time).
    snapshots_data: dict[str, Any] = {
        "sid": _col(df, "SID").to_numpy(),
        "season": season_int.astype(int).to_numpy(),
        "basin": _coerce_string(_col(df, "BASIN")).to_numpy(),
        "subbasin": _coerce_string(_col(df, "SUBBASIN")).to_numpy(),
        "name": _coerce_string(_col(df, "NAME")).to_numpy(),
        "number": _coerce_nullable_int(_col(df, "NUMBER")),
        "iso_time": iso_time_str.to_numpy(),
        "nature": _coerce_string(_col(df, "NATURE")).to_numpy(),
        "lat": _coerce_float(_col(df, "LAT")).to_numpy(),
        "lon": _coerce_float(_col(df, "LON")).to_numpy(),
        "usa_atcf_id": _col(df, "USA_ATCF_ID").to_numpy(),
        "usa_wind": _coerce_float(_col(df, "USA_WIND")).to_numpy(),
        "usa_pres": _coerce_float(_col(df, "USA_PRES")).to_numpy(),
        "usa_sshs": _coerce_nullable_int(_col(df, "USA_SSHS")),
    }
    for raw_col, out_col in zip(
        [f"{t}_{q}" for t in _RADIUS_THRESHOLDS for q in _RADIUS_QUADRANTS],
        _RADIUS_OUTPUT_COLUMNS,
        strict=True,
    ):
        snapshots_data[out_col] = _coerce_float(_col(df, raw_col)).to_numpy()

    snapshots = pd.DataFrame(snapshots_data)
    snapshots = cast(
        pd.DataFrame,
        snapshots.sort_values(["sid", "iso_time"]).reset_index(drop=True),
    )

    # Build ATCF→SID translation: one row per (sid, usa_atcf_id), resolving duplicates by max wind.
    pairs = cast(
        pd.DataFrame,
        snapshots[["sid", "season", "basin", "subbasin", "name", "usa_atcf_id", "usa_wind"]],
    )
    pairs = cast(pd.DataFrame, pairs[pairs["usa_atcf_id"] != ""])
    per_pair = cast(
        pd.DataFrame,
        pairs.groupby(["sid", "usa_atcf_id"], as_index=False).agg(
            season=("season", "first"),
            basin=("basin", "first"),
            subbasin=("subbasin", "first"),
            name=("name", "first"),
            usa_wind=("usa_wind", "max"),
        ),
    )
    # NaN usa_wind must not win the tie-break; treat missing as -inf.
    best_idx = cast(
        pd.Series,
        per_pair.assign(usa_wind_rank=per_pair["usa_wind"].fillna(float("-inf")))
        .groupby("sid")["usa_wind_rank"]
        .idxmax(),
    )
    atcf_to_sid = cast(
        pd.DataFrame,
        per_pair.loc[best_idx, ["sid", "season", "basin", "subbasin", "name", "usa_atcf_id"]]
        .sort_values("usa_atcf_id")
        .reset_index(drop=True),
    )

    return snapshots, atcf_to_sid


def write_outputs(
    snapshots: pd.DataFrame,
    atcf_to_sid: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write both Stage 0 artifacts under ``out_dir`` (created if missing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    atcf_path = out_dir / "atcf_to_sid.csv"
    parquet_path = out_dir / "ibtracs.parquet"
    atcf_to_sid.to_csv(atcf_path, index=False)
    snapshots.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(atcf_to_sid)} ATCF→SID pairs → {atcf_path}")
    print(
        f"Wrote {len(snapshots)} snapshots ({snapshots['sid'].nunique()} storms) → {parquet_path}"
    )


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Convert the raw IBTrACS CSV into the Stage 0 artifacts used downstream."""
    cfg: dict[str, Any] = resolve_preproc_cfg(raw_cfg)

    ibtracs_csv = Path(cfg["paths"]["raw_datasets"]["ibtracs"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    out_dir = sources_root / _IBTRACS_DIR_NAME

    print(f"Reading IBTrACS CSV from {ibtracs_csv} …")
    snapshots, atcf_to_sid = preprocess_ibtracs(ibtracs_csv)
    write_outputs(snapshots, atcf_to_sid, out_dir)


if __name__ == "__main__":
    main()
