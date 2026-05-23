#!/usr/bin/env python3
"""Assemble individually preprocessed sources into one HDF5 file per storm."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import h5py
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.preprocess.utils.runner import resolve_preproc_cfg
from tcfuse.data.ibtracs import (
    IBTRACS_SOURCE_NAME,
    float_or_nan,
    ibtracs_rows_to_sources,
    load_ibtracs,
)
from tcfuse.data.sources import MultisourceMetadata, Source, SourceKind, StormData
from tcfuse.utils.archive import submit_archive_job
from tcfuse.utils.time import to_compact_time

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
        snap_group.attrs[key] = value


def _copy_snapshot_to_assembled(
    source_path: Path,
    dest_file: h5py.File,
    source_name: str,
    snapshot_time_utc: str,
) -> None:
    """Copy one per-source snapshot into an open assembled storm HDF5 file."""
    compact_time = to_compact_time(snapshot_time_utc)
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
    compact_time = to_compact_time(snapshot_time_utc)
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
    """Stream all available sources for one storm into an assembled HDF5 file."""
    sid = atcf_to_sid.get(storm_id)
    final_storm_id = sid if sid is not None else storm_id
    dest_path = StormData.path(assembled_root, final_storm_id)

    if skip_existing and dest_path.exists():
        return storm_id

    basin = storm_id[:2]
    season = int(storm_id[-4:])

    atcf_id: str | None = None
    ibtracs_sources: list[tuple[str, Source]] = []
    if sid is not None:
        storm_rows = ibtracs_by_sid[sid]
        atcf_id = str(storm_rows["USA_ATCF_ID"].iloc[0]).strip()
        ibtracs_sources = ibtracs_rows_to_sources(storm_rows, final_storm_id, basin)
    else:
        logging.warning(
            "IBTrACS: no ATCF match for %s; ibtracs_best_track sources will be absent.",
            storm_id,
        )

    if rows.empty and not ibtracs_sources:
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written_snapshots = 0
    with h5py.File(dest_path, "w") as dest_file:
        dest_file.attrs["storm_id"] = final_storm_id
        dest_file.attrs["basin"] = basin
        dest_file.attrs["season"] = season
        if atcf_id is not None:
            dest_file.attrs["atcf_id"] = atcf_id

        for _, row in rows.iterrows():
            file_path = Path(str(row["file_path"]))
            if not file_path.exists():
                raise FileNotFoundError(f"Snapshot file missing: {file_path}")
            _copy_snapshot_to_assembled(
                file_path,
                dest_file,
                str(row["source_name"]),
                str(row["snapshot_time_utc"]),
            )
            written_snapshots += 1

        for snapshot_time_utc, source in ibtracs_sources:
            _write_source_to_assembled(
                source,
                dest_file,
                IBTRACS_SOURCE_NAME,
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
    """Assemble a batch of storms, optionally in parallel."""
    storm_id_set = set(storm_ids)
    grouped = {sid: grp for sid, grp in index.groupby("storm_id") if sid in storm_id_set}

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


def _ibtracs_rows_from_assembled(
    assembled_root: Path,
    atcf_storm_id: str,
    ibtracs_sid: str,
) -> list[dict[str, Any]]:
    """Read injected IBTrACS snapshot index rows from an assembled storm file."""
    storm_path = StormData.path(assembled_root, ibtracs_sid)
    if not storm_path.exists():
        return []

    atcf_id_val: str | None = None
    basin = atcf_storm_id[:2]
    season = int(atcf_storm_id[-4:])
    rows: list[dict[str, Any]] = []

    with h5py.File(storm_path, "r") as storm_file:
        if IBTRACS_SOURCE_NAME not in storm_file:
            return []
        source_group = cast(h5py.Group, storm_file[IBTRACS_SOURCE_NAME])
        if "atcf_id" in storm_file.attrs:
            atcf_id_val = str(storm_file.attrs["atcf_id"])

        for compact_time in source_group.keys():
            snap_group = cast(h5py.Group, source_group[compact_time])
            snapshot_time_utc = str(snap_group.attrs["snapshot_time_utc"])
            lat = float(np.asarray(snap_group.attrs["lat"]).item())
            lon = float(np.asarray(snap_group.attrs["lon"]).item())
            usa_vmax_kt = float_or_nan(snap_group.attrs.get("usa_vmax_kt", np.nan))
            wmo_vmax_kt = float_or_nan(snap_group.attrs.get("wmo_vmax_kt", np.nan))
            rows.append(
                {
                    "storm_id": ibtracs_sid,
                    "basin": basin,
                    "season": season,
                    "atcf_id": atcf_id_val,
                    "source_name": IBTRACS_SOURCE_NAME,
                    "snapshot_time_utc": snapshot_time_utc,
                    "lat": lat,
                    "lon": lon,
                    "usa_vmax_kt": usa_vmax_kt,
                    "wmo_vmax_kt": wmo_vmax_kt,
                }
            )

    return rows


def build_assembled_index(
    multi_meta_index: pd.DataFrame,
    atcf_to_sid: dict[str, str],
    assembled_storm_ids: list[str],
    assembled_root: Path,
) -> pd.DataFrame:
    """Build a dataset-wide index with one row per (storm_id, source_name, snapshot)."""
    assembled_set = list(assembled_storm_ids)
    non_ibt = multi_meta_index[multi_meta_index["storm_id"].isin(assembled_set)].copy()

    non_ibt["basin"] = cast(pd.Series, non_ibt["storm_id"]).str[:2]
    non_ibt["season"] = cast(pd.Series, non_ibt["storm_id"]).str[-4:].astype(int)
    storm_id_series = cast(pd.Series, non_ibt["storm_id"])
    non_ibt["atcf_id"] = storm_id_series.where(storm_id_series.isin(atcf_to_sid), other=None)
    non_ibt["storm_id"] = storm_id_series.map(lambda atcf: atcf_to_sid.get(str(atcf), str(atcf)))
    non_ibt["usa_vmax_kt"] = np.nan
    non_ibt["wmo_vmax_kt"] = np.nan

    for col in ("lat", "lon"):
        if col not in non_ibt.columns:
            non_ibt[col] = np.nan

    ibt_rows_list: list[dict[str, Any]] = []
    for storm_id in assembled_storm_ids:
        ibtracs_sid = atcf_to_sid.get(storm_id)
        if ibtracs_sid is None:
            continue
        ibt_rows_list.extend(
            _ibtracs_rows_from_assembled(assembled_root, storm_id, ibtracs_sid)
        )

    non_ibt_df = cast(pd.DataFrame, non_ibt[_ASSEMBLED_INDEX_COLUMNS])
    ibt_df = pd.DataFrame(ibt_rows_list, columns=_ASSEMBLED_INDEX_COLUMNS)
    index_df = pd.concat([non_ibt_df, ibt_df], ignore_index=True)
    return index_df.sort_values(["storm_id", "snapshot_time_utc"]).reset_index(drop=True)


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Assemble all available preprocessed sources into per-storm HDF5 files."""
    cfg = resolve_preproc_cfg(raw_cfg)
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    assembled_root.mkdir(parents=True, exist_ok=True)

    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

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

    if cfg.get("submitit", False):
        from tcfuse.utils.submitit_utils import make_executor

        chunk_size = int(cfg.get("chunk_size", 200))
        chunks = [storm_ids[i : i + chunk_size] for i in range(0, len(storm_ids), chunk_size)]
        slurm_executor = make_executor(cfg, "assemble")
        print(
            f"Submitting {len(chunks)} SLURM jobs ({len(storm_ids)} storms, "
            f"chunk_size={chunk_size}) …"
        )
        results: list[str | None] = []
        for job in tqdm(
            [
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
            ],
            desc="collecting",
        ):
            results.extend(job.result())
    else:
        results = _assemble_storms_batch(
            storm_ids,
            index,
            assembled_root,
            skip_existing,
            num_workers,
            ibtracs_by_sid,
            atcf_to_sid,
        )

    written = [r for r in results if r is not None]
    skipped = len(results) - len(written)
    print(f"Assembled {len(written)}/{len(storm_ids)} storms → {assembled_root}")
    if skipped:
        print(f"Skipped / failed: {skipped}")

    print("Building assembled index …")
    index_df = build_assembled_index(index, atcf_to_sid, written, assembled_root)
    index_path = assembled_root / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"Wrote assembled index: {len(index_df)} rows → {index_path}")

    submit_archive_job(
        assembled_root,
        Path(cfg["paths"]["archives"]["preprocessed_data"]),
        cfg,
        job_name="archive_assembled",
    )


if __name__ == "__main__":
    main()
