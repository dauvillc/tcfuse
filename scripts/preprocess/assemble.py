#!/usr/bin/env python3
"""Assemble individually preprocessed sources into one HDF5 file per storm.

Reads all per-source snapshot files produced by the source-specific preprocessing
scripts (prepare_pmw, prepare_infrared, prepare_radar, …) and merges them into a
single ``{assembled_data}/{storm_id}.h5`` file per tropical cyclone.  The assembled
files are consumed by the ML dataloader, which opens exactly one file per sample.

Additionally, IBTrACS best-track data (``ibtracs_best_track`` source) is injected
into every assembled storm for which a matching ATCF ID can be found in IBTrACS.
A global ``index.parquet`` is written to ``preprocessed_data`` after assembly.

Run from the project root:
    python scripts/preprocess/assemble.py [paths=jz] [submitit=false] [num_workers=4]
        [skip_existing=false]
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.data.sources import MultisourceMetadata, Source, SourceKind, StormData
from tcfuse.utils.archive import submit_archive_job

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IBTRACS_SOURCE_NAME = "ibtracs_best_track"
_IBTRACS_CHANNELS = [
    "vmax_kt",
    "mslp_hpa",
    "rmw_nm",
    "r34_ne_nm",
    "r34_se_nm",
    "r34_sw_nm",
    "r34_nw_nm",
]
_ASSEMBLED_INDEX_COLUMNS = [
    "storm_id",
    "basin",
    "season",
    "atcf_id",
    "source_name",
    "snapshot_time_utc",
    "lat",
    "lon",
    "vmax_kt",
]

# ---------------------------------------------------------------------------
# IBTrACS helpers
# ---------------------------------------------------------------------------


def load_ibtracs(
    path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load IBTrACS CSV and return rows indexed by SID plus a reverse ATCF lookup.

    Reads the IBTrACS v04r01 CSV, skipping the units row (row index 1 in the
    file), filters to MAIN tracks only, parses ISO_TIME to UTC-aware timestamps,
    and groups rows by IBTrACS SID.  An auxiliary reverse mapping from
    USA_ATCF_ID to SID is also built for fast storm matching.

    Args:
        path: Path to ``ibtracs.ALL.list.v04r01.csv``.

    Returns:
        Tuple of:
        - ``ibtracs_by_sid``: dict mapping IBTrACS SID to its DataFrame of
          MAIN-track rows.
        - ``atcf_to_sid``: dict mapping USA_ATCF_ID to the corresponding SID
          (only entries with a non-blank USA_ATCF_ID are included).
    """
    # Row 1 in the CSV (0-indexed after the header) is a units row — skip it.
    df = pd.read_csv(
        path,
        skiprows=[1],
        na_values=[" "],
        keep_default_na=True,
        low_memory=False,
    )

    # Restrict to primary tracks; spurs and provisional tracks are excluded.
    df = df[df["TRACK_TYPE"] == "main"].copy()

    # Parse timestamps to UTC-aware Timestamps.
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)

    # Strip leading/trailing whitespace from string identifier columns.
    # USA_ATCF_ID may be NaN (blank fields converted by na_values=[" "]) for
    # non-US-tracked storms, so fill NaN with "" before stripping.
    df["SID"] = df["SID"].fillna("").str.strip()
    df["USA_ATCF_ID"] = df["USA_ATCF_ID"].fillna("").str.strip()

    # Group rows by IBTrACS SID (primary key).
    ibtracs_by_sid: dict[str, pd.DataFrame] = {str(sid): grp for sid, grp in df.groupby("SID")}

    # Build reverse mapping: USA_ATCF_ID → SID (skip blank/NaN ATCF IDs).
    atcf_id_col = df[["SID", "USA_ATCF_ID"]].dropna(subset=["USA_ATCF_ID"])
    atcf_id_col = atcf_id_col[atcf_id_col["USA_ATCF_ID"] != ""]
    # Each ATCF ID should map to exactly one SID; take the first occurrence.
    atcf_to_sid: dict[str, str] = dict(zip(atcf_id_col["USA_ATCF_ID"], atcf_id_col["SID"]))

    return ibtracs_by_sid, atcf_to_sid


def ibtracs_rows_to_sources(
    storm_rows: pd.DataFrame,
    storm_id: str,
    basin: str,
) -> list[tuple[str, Source]]:
    """Convert IBTrACS rows for one storm into (snapshot_time_utc, Source) pairs.

    Each row becomes one SCALAR :class:`~tcfuse.data.sources.Source` with
    :data:`_IBTRACS_CHANNELS` as channels.  Missing numeric values (NaN in the
    CSV) are preserved as ``float32`` NaN in the values tensor.  Rows with NaN
    latitude or longitude are skipped with a warning.

    Args:
        storm_rows: DataFrame of MAIN-track rows for one storm, with a parsed
            UTC-aware ``ISO_TIME`` column and numeric intensity/size columns.
        storm_id: Project storm identifier, e.g. ``"2016AL10"``.
        basin: Ocean basin code, e.g. ``"AL"``.

    Returns:
        List of ``(snapshot_time_utc_isoformat, Source)`` pairs in
        chronological order.
    """
    # Sort rows chronologically before processing.
    storm_rows = storm_rows.sort_values("ISO_TIME")

    results: list[tuple[str, Source]] = []
    for _, row in storm_rows.iterrows():
        lat = row["LAT"]
        lon = row["LON"]

        # Skip rows where the storm centre position is missing.
        if pd.isna(lat) or pd.isna(lon):
            warnings.warn(
                f"IBTrACS row for {storm_id} at {row['ISO_TIME']} has NaN lat/lon — skipped.",
                stacklevel=2,
            )
            continue

        # Derive vmax_kt: prefer USA 1-min wind, fall back to WMO wind.
        usa_wind = row.get("USA_WIND", np.nan)
        wmo_wind = row.get("WMO_WIND", np.nan)
        vmax_kt = (
            float(usa_wind)
            if not pd.isna(usa_wind)
            else float(wmo_wind)
            if not pd.isna(wmo_wind)
            else np.nan
        )

        # Derive mslp_hpa: prefer USA pressure, fall back to WMO pressure.
        usa_pres = row.get("USA_PRES", np.nan)
        wmo_pres = row.get("WMO_PRES", np.nan)
        mslp_hpa = (
            float(usa_pres)
            if not pd.isna(usa_pres)
            else float(wmo_pres)
            if not pd.isna(wmo_pres)
            else np.nan
        )

        # Wind-structure parameters (NaN when not reported).
        rmw_nm = float(row["USA_RMW"]) if not pd.isna(row.get("USA_RMW", np.nan)) else np.nan
        r34_ne = float(row["USA_R34_NE"]) if not pd.isna(row.get("USA_R34_NE", np.nan)) else np.nan
        r34_se = float(row["USA_R34_SE"]) if not pd.isna(row.get("USA_R34_SE", np.nan)) else np.nan
        r34_sw = float(row["USA_R34_SW"]) if not pd.isna(row.get("USA_R34_SW", np.nan)) else np.nan
        r34_nw = float(row["USA_R34_NW"]) if not pd.isna(row.get("USA_R34_NW", np.nan)) else np.nan

        # Build tensors.
        values = torch.tensor(
            [vmax_kt, mslp_hpa, rmw_nm, r34_ne, r34_se, r34_sw, r34_nw],
            dtype=torch.float32,
        )  # (7,)
        time_unix_s = float(row["ISO_TIME"].timestamp())
        coords = torch.tensor(
            [time_unix_s, float(lat), float(lon)],
            dtype=torch.float64,
        )  # (3,)

        # ISO 8601 key compatible with the rest of the pipeline.
        snapshot_time_utc = row["ISO_TIME"].isoformat()

        source = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=coords,
            source_name=_IBTRACS_SOURCE_NAME,
            channels=_IBTRACS_CHANNELS,
            meta={
                "storm_id": storm_id,
                "basin": basin,
                "snapshot_time_utc": snapshot_time_utc,
                "lat": float(lat),
                "lon": float(lon),
                "vmax_kt": vmax_kt,
            },
        )
        results.append((snapshot_time_utc, source))

    return results


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_storm(
    storm_id: str,
    rows: pd.DataFrame,
    assembled_root: Path,
    skip_existing: bool,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    atcf_to_sid: dict[str, str],
) -> str | None:
    """Load all available sources for one storm and write the assembled HDF5 file.

    Iterates over every index row for ``storm_id`` (one row per source snapshot),
    loads the corresponding HDF5 snapshot from disk, and writes a single
    ``{assembled_root}/{storm_id}.h5`` containing all sources.  IBTrACS
    best-track snapshots are injected automatically when a matching ATCF ID is
    found.

    Missing snapshot files (present in the index but absent on disk) are silently
    skipped with a warning so that a partially downloaded dataset does not block
    assembly of the remaining storms.

    Args:
        storm_id: Storm identifier, e.g. ``"2016AL10"``.
        rows: All index rows for this storm (one row per available snapshot,
            across all sources).  Columns must include ``source_name``,
            ``snapshot_time_utc``, and ``file_path``.
        assembled_root: Root directory for assembled storm files
            (``cfg.paths.preprocessed_data``).
        skip_existing: If True, return immediately when the assembled file
            already exists on disk.
        ibtracs_by_sid: IBTrACS rows grouped by SID (primary key).
        atcf_to_sid: Reverse mapping USA_ATCF_ID → SID for storm matching.

    Returns:
        ``storm_id`` on success, ``None`` when skipped (already exists) or
        when no source snapshots could be loaded.
    """
    # storm_id is the ATCF ID (BBCCYYYY, e.g. "AL102016"); resolve to the
    # IBTrACS SID which becomes the authoritative assembled file name.
    sid = atcf_to_sid.get(storm_id)
    final_storm_id = sid if sid is not None else storm_id
    dest_path = StormData.path(assembled_root, final_storm_id)

    # Skip storms whose assembled file already exists when requested.
    if skip_existing and dest_path.exists():
        return storm_id

    # Parse basin and season from ATCF ID format (BBCCYYYY).
    basin = storm_id[:2]
    season = int(storm_id[-4:])

    # Load each snapshot from its individual HDF5 file.
    sources: dict[tuple[str, str], Source] = {}
    for _, row in rows.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.exists():
            warnings.warn(
                f"Snapshot file missing, skipping: {file_path}",
                stacklevel=2,
            )
            continue
        source = Source.from_disk(file_path)
        key = (str(row["source_name"]), str(row["snapshot_time_utc"]))
        sources[key] = source

    # Inject IBTrACS best-track sources when a matching SID was found.
    atcf_id: str | None = None
    if sid is not None:
        storm_rows = ibtracs_by_sid[sid]
        # Read atcf_id from the table (storm_id already equals USA_ATCF_ID).
        atcf_id = str(storm_rows["USA_ATCF_ID"].iloc[0]).strip()
        ibtracs_sources = ibtracs_rows_to_sources(storm_rows, final_storm_id, basin)
        for snapshot_time_utc, source in ibtracs_sources:
            sources[(_IBTRACS_SOURCE_NAME, snapshot_time_utc)] = source
    else:
        logging.warning(
            "IBTrACS: no ATCF match for %s; ibtracs_best_track sources will be absent.",
            storm_id,
        )

    # Nothing to write if no source snapshots at all.
    if not sources:
        return None

    storm_data = StormData(
        storm_id=final_storm_id,
        basin=basin,
        season=season,
        sources=sources,
        atcf_id=atcf_id,
    )
    storm_data.write(assembled_root)
    return storm_id


def _assemble_storms_batch(
    storm_ids: list[str],
    index: pd.DataFrame,
    assembled_root: Path,
    skip_existing: bool,
    num_workers: int,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    atcf_to_sid: dict[str, str],
) -> list[str | None]:
    """Assemble a batch of storms, optionally in parallel.

    Designed to run either locally or as a single submitit SLURM job.  Within
    the job, individual storms are processed in parallel using ProcessPoolExecutor.

    Args:
        storm_ids: Ordered list of storm identifiers to assemble.
        index: Merged source index (all sources), used to look up each storm's rows.
        assembled_root: Root directory for assembled storm files
            (``cfg.paths.preprocessed_data``).
        skip_existing: Forwarded to :func:`assemble_storm`.
        num_workers: Number of parallel worker processes.  Use 1 for sequential.
        ibtracs_by_sid: IBTrACS rows grouped by SID; forwarded to :func:`assemble_storm`.
        atcf_to_sid: Reverse ATCF lookup; forwarded to :func:`assemble_storm`.

    Returns:
        List of results from :func:`assemble_storm` in the same order as
        ``storm_ids`` (``storm_id`` on success, ``None`` on skip/failure).
    """
    # Pre-group the index by storm_id for fast per-storm lookup.
    grouped = {sid: grp for sid, grp in index.groupby("storm_id") if sid in set(storm_ids)}

    # Sequential fallback for debugging or when num_workers == 1.
    if num_workers <= 1:
        return [
            assemble_storm(
                sid,
                grouped[sid],
                assembled_root,
                skip_existing,
                ibtracs_by_sid,
                atcf_to_sid,
            )
            for sid in tqdm(storm_ids, desc="assemble")
        ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    assemble_storm,
                    storm_ids,
                    [grouped[sid] for sid in storm_ids],
                    repeat(assembled_root),
                    repeat(skip_existing),
                    repeat(ibtracs_by_sid),
                    repeat(atcf_to_sid),
                    chunksize=max(1, len(storm_ids) // (num_workers * 4)),
                ),
                total=len(storm_ids),
                desc="assemble",
            )
        )


# ---------------------------------------------------------------------------
# Assembled index
# ---------------------------------------------------------------------------


def build_assembled_index(
    multi_meta_index: pd.DataFrame,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    atcf_to_sid: dict[str, str],
    assembled_storm_ids: list[str],
) -> pd.DataFrame:
    """Build a dataset-wide index with one row per (storm_id, source_name, snapshot).

    Columns: ``storm_id``, ``basin``, ``season``, ``atcf_id``, ``source_name``,
    ``snapshot_time_utc``, ``lat``, ``lon``, ``vmax_kt``.

    For ``ibtracs_best_track`` rows, ``lat``, ``lon``, and ``vmax_kt`` come
    directly from IBTrACS.  For all other source rows, ``vmax_kt`` is left as
    the value from the per-source index (NaN when absent — no join is performed).

    Args:
        multi_meta_index: Merged per-source snapshot index from
            :class:`~tcfuse.data.sources.MultisourceMetadata` (one row per
            non-IBTrACS snapshot across all sources).
        ibtracs_by_sid: IBTrACS rows grouped by SID.
        atcf_to_sid: Reverse mapping USA_ATCF_ID → SID.
        assembled_storm_ids: Storm IDs that were successfully assembled; used to
            filter rows from ``multi_meta_index``.

    Returns:
        DataFrame with schema ``_ASSEMBLED_INDEX_COLUMNS``, sorted by
        ``(storm_id, snapshot_time_utc)``.
    """
    assembled_set = set(assembled_storm_ids)

    # --- Part A: non-IBTrACS rows from the per-source index -------------------
    non_ibt = multi_meta_index[multi_meta_index["storm_id"].isin(assembled_set)].copy()

    # Add storm-level columns derived from ATCF ID format (BBCCYYYY).
    non_ibt["basin"] = non_ibt["storm_id"].str[:2]
    non_ibt["season"] = non_ibt["storm_id"].str[-4:].astype(int)

    # atcf_id = the ATCF ID itself when matched in IBTrACS, else None.
    non_ibt["atcf_id"] = non_ibt["storm_id"].where(
        non_ibt["storm_id"].isin(atcf_to_sid), other=None
    )
    # Replace ATCF ID with IBTrACS SID as the final storm_id.
    non_ibt["storm_id"] = non_ibt["storm_id"].map(
        lambda atcf: atcf_to_sid.get(str(atcf), str(atcf))
    )

    # Ensure vmax_kt column exists (may be absent in some source indexes).
    if "vmax_kt" not in non_ibt.columns:
        non_ibt["vmax_kt"] = np.nan

    # Ensure lat/lon columns exist.
    for col in ("lat", "lon"):
        if col not in non_ibt.columns:
            non_ibt[col] = np.nan

    # --- Part B: ibtracs_best_track rows from the IBTrACS data ----------------
    # assembled_storm_ids are ATCF IDs (BBCCYYYY); look them up directly.
    ibt_rows_list: list[dict[str, Any]] = []
    for storm_id in assembled_storm_ids:
        ibtracs_sid = atcf_to_sid.get(storm_id)
        if ibtracs_sid is None:
            continue
        storm_df = ibtracs_by_sid[ibtracs_sid]
        atcf_id_val = str(storm_df["USA_ATCF_ID"].iloc[0]).strip()
        basin = storm_id[:2]
        season = int(storm_id[-4:])
        for _, row in storm_df.sort_values("ISO_TIME").iterrows():
            lat = row["LAT"]
            lon = row["LON"]
            if pd.isna(lat) or pd.isna(lon):
                continue
            usa_wind = row.get("USA_WIND", np.nan)
            wmo_wind = row.get("WMO_WIND", np.nan)
            vmax_kt = (
                float(usa_wind)
                if not pd.isna(usa_wind)
                else float(wmo_wind)
                if not pd.isna(wmo_wind)
                else np.nan
            )
            ibt_rows_list.append(
                {
                    "storm_id": ibtracs_sid,
                    "basin": basin,
                    "season": season,
                    "atcf_id": atcf_id_val,
                    "source_name": _IBTRACS_SOURCE_NAME,
                    "snapshot_time_utc": row["ISO_TIME"].isoformat(),
                    "lat": float(lat),
                    "lon": float(lon),
                    "vmax_kt": vmax_kt,
                }
            )

    ibt_df = pd.DataFrame(ibt_rows_list, columns=_ASSEMBLED_INDEX_COLUMNS)

    # --- Combine and finalise -------------------------------------------------
    index_df = pd.concat(
        [non_ibt[_ASSEMBLED_INDEX_COLUMNS], ibt_df],
        ignore_index=True,
    )
    index_df = index_df.sort_values(["storm_id", "snapshot_time_utc"]).reset_index(drop=True)
    return index_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Assemble all available preprocessed sources into per-storm HDF5 files."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    assembled_root.mkdir(parents=True, exist_ok=True)

    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))
    launch_local = not bool(cfg.get("submitit", False))

    # Load IBTrACS before assembly.
    ibtracs_path = Path(cfg["paths"]["raw_datasets"]["ibtracs"])
    if not ibtracs_path.exists():
        print(
            f"WARNING: IBTrACS CSV not found at {ibtracs_path}. "
            "ibtracs_best_track sources will be absent from all storms."
        )
        ibtracs_by_sid: dict[str, pd.DataFrame] = {}
        atcf_to_sid: dict[str, str] = {}
    else:
        print(f"Loading IBTrACS from {ibtracs_path} …")
        ibtracs_by_sid, atcf_to_sid = load_ibtracs(ibtracs_path)
        print(
            f"Loaded {len(ibtracs_by_sid)} storms from IBTrACS ({len(atcf_to_sid)} ATCF-tracked)."
        )

    # Scan all source directories and build a merged index across sources.
    print(f"Loading source metadata from {sources_root} …")
    multi_meta = MultisourceMetadata.from_disk(sources_root)
    if not multi_meta.sources:
        print("No preprocessed sources found. Nothing to assemble.")
        return
    index = multi_meta.index
    storm_ids = sorted(index["storm_id"].unique())
    print(
        f"Found {len(storm_ids)} storms across {len(multi_meta)} sources "
        f"({len(index)} total snapshots)."
    )

    if launch_local:
        results: list[str | None] = _assemble_storms_batch(
            storm_ids,
            index,
            assembled_root,
            skip_existing,
            num_workers,
            ibtracs_by_sid,
            atcf_to_sid,
        )
    else:
        from tcfuse.utils.submitit_utils import make_executor as make_submitit_executor

        chunk_size = int(cfg.get("chunk_size", 200))
        chunks = [storm_ids[i : i + chunk_size] for i in range(0, len(storm_ids), chunk_size)]

        slurm_executor = make_submitit_executor(cfg, "assemble")
        print(
            f"Submitting {len(chunks)} SLURM jobs ({len(storm_ids)} storms,\
            chunk_size={chunk_size}) …"
        )
        jobs = [
            slurm_executor.submit(
                _assemble_storms_batch,
                chunk,
                index,
                assembled_root,
                skip_existing,
                num_workers,
                ibtracs_by_sid,
                atcf_to_sid,
            )
            for chunk in chunks
        ]
        results = []
        for job in tqdm(jobs, desc="collecting"):
            results.extend(job.result())

    written = [r for r in results if r is not None]
    skipped = len(results) - len(written)
    print(f"Assembled {len(written)}/{len(storm_ids)} storms → {assembled_root}")
    if skipped:
        print(f"Skipped / failed: {skipped}")

    # Write global assembled index.
    print("Building assembled index …")
    index_df = build_assembled_index(index, ibtracs_by_sid, atcf_to_sid, written)
    index_path = assembled_root / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"Wrote assembled index: {len(index_df)} rows → {index_path}")

    # Archive the entire assembled directory to STORE as a single tarball.
    submit_archive_job(
        assembled_root,
        Path(cfg["paths"]["archives"]["preprocessed_data"]),
        cfg,
        job_name="archive_assembled",
    )


if __name__ == "__main__":
    main()
