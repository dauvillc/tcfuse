"""Stage 0 IBTrACS I/O.

Loads the preprocessed parquet / translation CSV and converts per-storm rows
into :class:`~tcfuse.data.sources.Source` objects.

The raw IBTrACS CSV is only read once, by ``scripts/preprocess/prepare_ibtracs.py``.
Every downstream consumer goes through this module instead.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from tcfuse.data.sources import Source, SourceKind

# Source identifier used everywhere the best-track is injected.
IBTRACS_SOURCE_NAME = "ibtracs_best_track"

# Channels written into the injected ibtracs_best_track Source.
# lat/lon appear here AND in Source.coords by design — the model can then
# treat storm position as a feature, not only as a coordinate.
IBTRACS_CHANNELS: list[str] = [
    "usa_wind",
    "usa_pres",
    "lat",
    "lon",
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
]

_IBTRACS_DIR_NAME = "ibtracs"


def ibtracs_paths(sources_root: Path) -> tuple[Path, Path]:
    """Return the canonical Stage 0 output paths under ``sources_root``."""
    base = sources_root / _IBTRACS_DIR_NAME
    return base / "ibtracs.parquet", base / "atcf_to_sid.csv"


def load_ibtracs_snapshots(sources_root: Path) -> pd.DataFrame:
    """Load the preprocessed IBTrACS snapshots parquet produced by Stage 0."""
    parquet_path, _ = ibtracs_paths(sources_root)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"IBTrACS snapshots parquet not found at {parquet_path}. "
            "Run scripts/preprocess/prepare_ibtracs.py first."
        )
    return pd.read_parquet(parquet_path)


def load_atcf_to_sid(sources_root: Path) -> pd.DataFrame:
    """Load the ATCF→SID translation table produced by Stage 0."""
    _, csv_path = ibtracs_paths(sources_root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ATCF→SID CSV not found at {csv_path}. "
            "Run scripts/preprocess/prepare_ibtracs.py first."
        )
    df = pd.read_csv(csv_path, dtype={"sid": str, "usa_atcf_id": str})
    df["season"] = cast(pd.Series, df["season"]).astype(int)
    return df


def load_atcf_to_sid_dict(sources_root: Path) -> dict[str, str]:
    """Convenience helper: return a plain ``{usa_atcf_id: sid}`` lookup."""
    df = load_atcf_to_sid(sources_root)
    return dict(zip(cast(pd.Series, df["usa_atcf_id"]), cast(pd.Series, df["sid"]), strict=True))


def ibtracs_rows_to_sources(
    storm_rows: pd.DataFrame,
    sid: str,
    basin: str,
) -> list[tuple[str, Source]]:
    """Convert per-storm IBTrACS rows into ``(time_utc, Source)`` pairs."""
    storm_rows = cast(pd.DataFrame, storm_rows.sort_values("iso_time"))
    results: list[tuple[str, Source]] = []

    for _, row in storm_rows.iterrows():
        # Read lat/lon — required to build valid SCALAR coords; skip row if missing.
        lat_val = row.get("lat")
        lon_val = row.get("lon")
        lat = (
            float("nan") if lat_val is None or bool(pd.isna(lat_val)) else float(cast(Any, lat_val))
        )
        lon = (
            float("nan") if lon_val is None or bool(pd.isna(lon_val)) else float(cast(Any, lon_val))
        )
        iso_time_raw = str(row["iso_time"])

        if np.isnan(lat) or np.isnan(lon):
            warnings.warn(
                f"IBTrACS row for {sid} at {iso_time_raw} has NaN lat/lon — skipped.",
                stacklevel=2,
            )
            continue

        # Normalize snapshot time to a tz-naive UTC Timestamp.
        iso_time = cast(pd.Timestamp, pd.Timestamp(iso_time_raw))
        if iso_time.tzinfo is None:
            iso_time_utc = iso_time.tz_localize("UTC")
        else:
            iso_time_utc = iso_time.tz_convert("UTC")
        # Tz-naive Timestamp for Source.time_utc; .isoformat() used as pipeline index key.
        time_utc = iso_time_utc.tz_localize(None)

        # Stack all 16 channels; missing numeric entries become NaN.
        channel_values: list[float] = []
        for channel in IBTRACS_CHANNELS:
            val = row.get(channel)
            if val is None or bool(pd.isna(val)):
                channel_values.append(float("nan"))
            else:
                channel_values.append(float(cast(Any, val)))

        values = np.array(channel_values, dtype=np.float32)
        # Spatial coords only: [lat, lon] (time stored separately in time_utc).
        coords = np.array([lat, lon], dtype=np.float64)

        source = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=coords,
            source_name=IBTRACS_SOURCE_NAME,
            channels=IBTRACS_CHANNELS,
            mask=np.isfinite(values),
            time_utc=time_utc,
            meta={
                "storm_id": sid,
                "basin": basin,
                "time_utc": time_utc.isoformat(),
            },
        )
        results.append((time_utc.isoformat(), source))

    return results


def group_ibtracs_by_sid(snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group the IBTrACS snapshots DataFrame by ``sid`` for fast per-storm access."""
    return {str(sid): cast(pd.DataFrame, grp) for sid, grp in snapshots.groupby("sid")}
