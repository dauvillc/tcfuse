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
import torch

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
    """Return the canonical Stage 0 output paths under ``sources_root``.

    Args:
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).

    Returns:
        ``(ibtracs_parquet, atcf_to_sid_csv)``.
    """
    base = sources_root / _IBTRACS_DIR_NAME
    return base / "ibtracs.parquet", base / "atcf_to_sid.csv"


def load_ibtracs_snapshots(sources_root: Path) -> pd.DataFrame:
    """Load the preprocessed IBTrACS snapshots parquet produced by Stage 0.

    Args:
        sources_root: Root directory for preprocessed sources.

    Returns:
        DataFrame with one row per (sid, iso_time). See
        ``scripts/preprocess/prepare_ibtracs.py`` for the full schema.

    Raises:
        FileNotFoundError: When the parquet does not exist.
    """
    parquet_path, _ = ibtracs_paths(sources_root)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"IBTrACS snapshots parquet not found at {parquet_path}. "
            "Run scripts/preprocess/prepare_ibtracs.py first."
        )
    return pd.read_parquet(parquet_path)


def load_atcf_to_sid(sources_root: Path) -> pd.DataFrame:
    """Load the ATCF→SID translation table produced by Stage 0.

    Args:
        sources_root: Root directory for preprocessed sources.

    Returns:
        DataFrame with columns ``sid, season, basin, subbasin, name,
        usa_atcf_id``.

    Raises:
        FileNotFoundError: When the CSV does not exist.
    """
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


def _row_value(row: pd.Series, key: str) -> float:
    """Read a numeric column from an IBTrACS row, mapping missing entries to NaN."""
    if key not in row.index:
        return float("nan")
    value = row[key]
    if value is None or bool(pd.isna(value)):
        return float("nan")
    return float(cast(Any, value))


def ibtracs_rows_to_sources(
    storm_rows: pd.DataFrame,
    sid: str,
    basin: str,
) -> list[tuple[str, Source]]:
    """Convert per-storm IBTrACS rows into ``(snapshot_time_utc, Source)`` pairs.

    Each row becomes one SCALAR Source with the 16 channels listed in
    :data:`IBTRACS_CHANNELS`. ``coords = [time_unix_s, lat, lon]``; ``lat`` and
    ``lon`` are intentionally duplicated as values so the embedding layer can
    treat storm position as a feature.

    Rows with NaN ``lat`` or ``lon`` are skipped with a warning — those
    coordinates are required to build a valid SCALAR Source.

    Args:
        storm_rows: All IBTrACS rows for a single SID (any row order).
        sid: IBTrACS SID for the storm; written to ``Source.meta["storm_id"]``.
        basin: 2-letter basin code; written to ``Source.meta["basin"]``.

    Returns:
        List of ``(snapshot_time_utc, Source)`` tuples sorted by snapshot time.
    """
    storm_rows = cast(pd.DataFrame, storm_rows.sort_values("iso_time"))
    results: list[tuple[str, Source]] = []

    for _, row in storm_rows.iterrows():
        lat = _row_value(row, "lat")
        lon = _row_value(row, "lon")
        iso_time_raw = str(row["iso_time"])

        if np.isnan(lat) or np.isnan(lon):
            warnings.warn(
                f"IBTrACS row for {sid} at {iso_time_raw} has NaN lat/lon — skipped.",
                stacklevel=2,
            )
            continue

        iso_time = cast(pd.Timestamp, pd.Timestamp(iso_time_raw))
        if iso_time.tzinfo is None:
            iso_time_utc = iso_time.tz_localize("UTC")
        else:
            iso_time_utc = iso_time.tz_convert("UTC")
        time_unix_s = float(iso_time_utc.timestamp())
        snapshot_time_utc = iso_time_utc.tz_localize(None).isoformat()

        channel_values = [_row_value(row, channel) for channel in IBTRACS_CHANNELS]
        values = torch.tensor(channel_values, dtype=torch.float32)
        coords = torch.tensor([time_unix_s, lat, lon], dtype=torch.float64)

        source = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=coords,
            source_name=IBTRACS_SOURCE_NAME,
            channels=IBTRACS_CHANNELS,
            mask=torch.isfinite(values),
            meta={
                "storm_id": sid,
                "basin": basin,
                "snapshot_time_utc": snapshot_time_utc,
            },
        )
        results.append((snapshot_time_utc, source))

    return results


def group_ibtracs_by_sid(snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group the IBTrACS snapshots DataFrame by ``sid`` for fast per-storm access."""
    return {str(sid): cast(pd.DataFrame, grp) for sid, grp in snapshots.groupby("sid")}
