#!/usr/bin/env python3
"""Assemble individually preprocessed sources into one HDF5 file per storm.

Reads all per-source snapshot files produced by the source-specific preprocessing
scripts (prepare_pmw, prepare_infrared, prepare_radar, …) and merges them into a
single ``{assembled_data}/storm_data/{storm_id}.h5`` file per tropical cyclone.
The assembled files are consumed by the ML dataloader, which opens exactly one
file per sample.

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
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, cast

import h5py
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
_ASSEMBLED_INDEX_COLUMNS = [
    "storm_id",
    "basin",
    "season",
    "atcf_id",
    "source_name",
    "snapshot_time_utc",
    "lat",
    "lon",
    "usa_vmax_kt",
    "wmo_vmax_kt",
]
_SOURCE_KIND_GROUPS: dict[str, SourceKind] = {
    "scalar": SourceKind.SCALAR,
    "profile": SourceKind.PROFILE,
    "field": SourceKind.FIELD,
}
_ROOT_ATTRS = {"storm_id", "basin", "season", "atcf_id"}

# ---------------------------------------------------------------------------
# IBTrACS helpers
# ---------------------------------------------------------------------------


def _float_or_nan(value: Any) -> float:
    """Return a float value, preserving missing IBTrACS entries as NaN."""
    return float(value) if not pd.isna(value) else np.nan


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
    track_type = cast(pd.Series, df["TRACK_TYPE"]).astype(str).str.strip().str.lower()
    df = cast(pd.DataFrame, df[track_type == "main"].copy())

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
    atcf_id_col = cast(pd.DataFrame, df[["SID", "USA_ATCF_ID"]]).dropna(subset=["USA_ATCF_ID"])
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
        lat = cast(float, row["LAT"])
        lon = cast(float, row["LON"])
        iso_time = cast(pd.Timestamp, row["ISO_TIME"])

        # Skip rows where the storm centre position is missing.
        if pd.isna(lat) or pd.isna(lon):
            warnings.warn(
                f"IBTrACS row for {storm_id} at {iso_time} has NaN lat/lon — skipped.",
                stacklevel=2,
            )
            continue

        # Keep USA and WMO best-track definitions independent; never fall back
        # from one provider to the other.
        usa_vmax_kt = _float_or_nan(row.get("USA_WIND", np.nan))
        wmo_vmax_kt = _float_or_nan(row.get("WMO_WIND", np.nan))
        usa_mslp_hpa = _float_or_nan(row.get("USA_PRES", np.nan))
        wmo_mslp_hpa = _float_or_nan(row.get("WMO_PRES", np.nan))

        # USA wind-structure parameters are provider-specific and remain NaN
        # when not reported.
        usa_rmw_nm = _float_or_nan(row.get("USA_RMW", np.nan))
        usa_r34_ne = _float_or_nan(row.get("USA_R34_NE", np.nan))
        usa_r34_se = _float_or_nan(row.get("USA_R34_SE", np.nan))
        usa_r34_sw = _float_or_nan(row.get("USA_R34_SW", np.nan))
        usa_r34_nw = _float_or_nan(row.get("USA_R34_NW", np.nan))

        # Build tensors.
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
        )  # (9,)
        time_unix_s = float(iso_time.timestamp())
        coords = torch.tensor(
            [time_unix_s, float(lat), float(lon)],
            dtype=torch.float64,
        )  # (3,)

        # ISO 8601 key compatible with the rest of the pipeline.
        # Strip tzinfo to match the naive-UTC format used by other sources.
        snapshot_time_utc = iso_time.replace(tzinfo=None).isoformat()

        source = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=coords,
            source_name=_IBTRACS_SOURCE_NAME,
            channels=_IBTRACS_CHANNELS,
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


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _to_compact_time(snapshot_time_utc: str) -> str:
    """Convert an isoformat timestamp to the compact HDF5 group name."""
    return pd.Timestamp(snapshot_time_utc).strftime("%Y%m%dT%H%M%SZ")


def _find_single_source_group(source_file: h5py.File) -> tuple[h5py.Group, SourceKind]:
    """Return the single source group and kind from a per-source HDF5 file."""
    for kind_group_name, kind in _SOURCE_KIND_GROUPS.items():
        if kind_group_name not in source_file:
            continue
        kind_group = source_file[kind_group_name]
        if not isinstance(kind_group, h5py.Group):
            continue
        for group in kind_group.values():
            if isinstance(group, h5py.Group):
                return group, kind
    raise ValueError("No source group found in HDF5 file.")


def _set_snapshot_attrs(
    snap_group: h5py.Group,
    kind: SourceKind,
    snapshot_time_utc: str,
    meta: dict[str, Any],
) -> None:
    """Write assembled snapshot attributes shared by copied and injected sources."""
    snap_group.attrs["kind"] = kind.name
    snap_group.attrs["snapshot_time_utc"] = snapshot_time_utc
    for key, value in meta.items():
        if key in _ROOT_ATTRS or key == "snapshot_time_utc":
            continue
        try:
            snap_group.attrs[key] = value
        except TypeError:
            warnings.warn(
                f"Could not write meta key '{key}' as HDF5 attr: {type(value)}",
                stacklevel=2,
            )


def _copy_snapshot_to_assembled(
    source_path: Path,
    dest_file: h5py.File,
    source_name: str,
    snapshot_time_utc: str,
) -> None:
    """Copy one per-source snapshot into an open assembled storm HDF5 file."""
    compact_time = _to_compact_time(snapshot_time_utc)
    source_group = dest_file.require_group(source_name)
    if compact_time in source_group:
        del source_group[compact_time]

    with h5py.File(source_path, "r") as source_file:
        src_group, kind = _find_single_source_group(source_file)
        source_file.copy(src_group, source_group, name=compact_time)
        snap_group = source_group[compact_time]
        if not isinstance(snap_group, h5py.Group):
            raise TypeError(f"Copied snapshot is not an HDF5 group: {source_path}")
        _set_snapshot_attrs(snap_group, kind, snapshot_time_utc, dict(source_file.attrs))


def _write_source_to_assembled(
    source: Source,
    dest_file: h5py.File,
    source_name: str,
    snapshot_time_utc: str,
) -> None:
    """Write one in-memory Source into an open assembled storm HDF5 file."""
    compact_time = _to_compact_time(snapshot_time_utc)
    source_group = dest_file.require_group(source_name)
    if compact_time in source_group:
        del source_group[compact_time]
    snap_group = source_group.create_group(compact_time)
    source.to_hdf5_group(snap_group)
    _set_snapshot_attrs(snap_group, source.kind, snapshot_time_utc, source.meta)


def assemble_storm(
    storm_id: str,
    rows: pd.DataFrame,
    assembled_root: Path,
    skip_existing: bool,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    atcf_to_sid: dict[str, str],
) -> str | None:
    """Stream all available sources for one storm into an assembled HDF5 file.

    Iterates over every index row for ``storm_id`` (one row per source snapshot),
    copies the corresponding HDF5 snapshot from disk, and writes a single
    ``{assembled_root}/storm_data/{storm_id}.h5`` containing all sources.
    Disk-backed snapshots are copied directly to avoid holding a full storm's
    tensors in memory.  IBTrACS best-track snapshots are injected automatically
    when a matching ATCF ID is found.

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

    # Prepare tiny in-memory IBTrACS sources, if available.
    atcf_id: str | None = None
    ibtracs_sources: list[tuple[str, Source]] = []
    if sid is not None:
        storm_rows = ibtracs_by_sid[sid]
        # Read atcf_id from the table (storm_id already equals USA_ATCF_ID).
        atcf_id = str(storm_rows["USA_ATCF_ID"].iloc[0]).strip()
        ibtracs_sources = ibtracs_rows_to_sources(storm_rows, final_storm_id, basin)
    else:
        logging.warning(
            "IBTrACS: no ATCF match for %s; ibtracs_best_track sources will be absent.",
            storm_id,
        )

    # Avoid creating empty assembled files when every source snapshot is missing.
    has_disk_snapshot = any(Path(str(row["file_path"])).exists() for _, row in rows.iterrows())
    if not has_disk_snapshot and not ibtracs_sources:
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written_snapshots = 0
    with h5py.File(dest_path, "w") as dest_file:
        # Write storm-level constants as root attributes.
        dest_file.attrs["storm_id"] = final_storm_id
        dest_file.attrs["basin"] = basin
        dest_file.attrs["season"] = season
        if atcf_id is not None:
            dest_file.attrs["atcf_id"] = atcf_id

        # Copy source snapshots one at a time without materializing the storm.
        for _, row in rows.iterrows():
            file_path = Path(str(row["file_path"]))
            if not file_path.exists():
                warnings.warn(
                    f"Snapshot file missing, skipping: {file_path}",
                    stacklevel=2,
                )
                continue
            _copy_snapshot_to_assembled(
                file_path,
                dest_file,
                str(row["source_name"]),
                str(row["snapshot_time_utc"]),
            )
            written_snapshots += 1

        # Inject IBTrACS best-track sources after disk-backed snapshots.
        for snapshot_time_utc, source in ibtracs_sources:
            _write_source_to_assembled(
                source,
                dest_file,
                _IBTRACS_SOURCE_NAME,
                snapshot_time_utc,
            )
            written_snapshots += 1

    if written_snapshots == 0:
        dest_path.unlink(missing_ok=True)
        return None
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
    storm_id_set = set(storm_ids)
    grouped = {sid: grp for sid, grp in index.groupby("storm_id") if sid in storm_id_set}

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

    results: list[str | None] = [None] * len(storm_ids)
    pending: dict[Future[str | None], int] = {}
    storm_iter = iter(enumerate(storm_ids))
    max_pending = max(1, num_workers * 2)

    def submit_next(pool: ProcessPoolExecutor) -> bool:
        """Submit one storm to the pool if any remain."""
        try:
            idx, sid = next(storm_iter)
        except StopIteration:
            return False
        pending[
            pool.submit(
                assemble_storm,
                sid,
                grouped[sid],
                assembled_root,
                skip_existing,
                ibtracs_by_sid,
                atcf_to_sid,
            )
        ] = idx
        return True

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in range(min(max_pending, len(storm_ids))):
            submit_next(pool)

        with tqdm(total=len(storm_ids), desc="assemble") as progress:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    results[idx] = future.result()
                    progress.update(1)
                    submit_next(pool)

    return results


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
    ``snapshot_time_utc``, ``lat``, ``lon``, ``usa_vmax_kt``, ``wmo_vmax_kt``.

    For ``ibtracs_best_track`` rows, ``lat``, ``lon``, ``usa_vmax_kt``, and
    ``wmo_vmax_kt`` come directly from IBTrACS.  For all other source rows,
    best-track wind columns are left as NaN; no join or provider fallback is
    performed.

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
    assembled_set = list(assembled_storm_ids)

    # --- Part A: non-IBTrACS rows from the per-source index -------------------
    non_ibt = multi_meta_index[multi_meta_index["storm_id"].isin(assembled_set)].copy()

    # Add storm-level columns derived from ATCF ID format (BBCCYYYY).
    non_ibt["basin"] = cast(pd.Series, non_ibt["storm_id"]).str[:2]
    non_ibt["season"] = cast(pd.Series, non_ibt["storm_id"]).str[-4:].astype(int)

    # atcf_id = the ATCF ID itself when matched in IBTrACS, else None.
    storm_id_series = cast(pd.Series, non_ibt["storm_id"])
    non_ibt["atcf_id"] = storm_id_series.where(storm_id_series.isin(atcf_to_sid), other=None)
    # Replace ATCF ID with IBTrACS SID as the final storm_id.
    non_ibt["storm_id"] = storm_id_series.map(lambda atcf: atcf_to_sid.get(str(atcf), str(atcf)))

    # Non-IBTrACS rows must not inherit generic per-source intensity metadata
    # because USA and WMO best-track winds have distinct definitions.
    non_ibt["usa_vmax_kt"] = np.nan
    non_ibt["wmo_vmax_kt"] = np.nan

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
            lat = cast(float, row["LAT"])
            lon = cast(float, row["LON"])
            iso_time = cast(pd.Timestamp, row["ISO_TIME"])
            if pd.isna(lat) or pd.isna(lon):
                continue
            usa_vmax_kt = _float_or_nan(row.get("USA_WIND", np.nan))
            wmo_vmax_kt = _float_or_nan(row.get("WMO_WIND", np.nan))
            ibt_rows_list.append(
                {
                    "storm_id": ibtracs_sid,
                    "basin": basin,
                    "season": season,
                    "atcf_id": atcf_id_val,
                    "source_name": _IBTRACS_SOURCE_NAME,
                    "snapshot_time_utc": iso_time.replace(tzinfo=None).isoformat(),
                    "lat": float(lat),
                    "lon": float(lon),
                    "usa_vmax_kt": usa_vmax_kt,
                    "wmo_vmax_kt": wmo_vmax_kt,
                }
            )

    # --- Combine and finalise -------------------------------------------------
    non_ibt_df = cast(pd.DataFrame, non_ibt[_ASSEMBLED_INDEX_COLUMNS])
    ibt_df = pd.DataFrame(ibt_rows_list, columns=_ASSEMBLED_INDEX_COLUMNS)
    index_df = pd.concat(
        [non_ibt_df, ibt_df],
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
