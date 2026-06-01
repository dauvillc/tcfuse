#!/usr/bin/env python3
"""Stage 2 — assemble per-source HDF5 snapshots into one HDF5 file per storm.

Reads:
- Stage 1 outputs under ``${paths.preprocessed_sources}/<source>/``.
- Stage 0 IBTrACS artifacts under ``${paths.preprocessed_sources}/ibtracs/``.

Writes:
- ``${paths.preprocessed_data}/storm_data/{sid}.h5`` — one assembled HDF5 file
  per IBTrACS SID. Contains every available Stage 1 source plus an injected
  ``ibtracs_best_track`` SCALAR Source (16 channels).
- ``${paths.preprocessed_data}/index.parquet`` — concatenated index of
  satellite-source snapshot rows and full IBTrACS rows. Satellite rows leave
  IBTrACS-specific columns NaN; IBTrACS rows leave nothing extra.
- ``${paths.preprocessed_data}/sources_metadata.yaml`` — merged source
  descriptors (channels, shape, kind) for downstream ML pipeline use.

Storms not present in IBTrACS are simply not assembled — the SID set comes
straight from the IBTrACS parquet, after the ``TRACK_TYPE == "MAIN"`` filter
already applied at Stage 0.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import h5py
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.preprocess.utils.runner import resolve_preproc_cfg
from tcfuse.data.ibtracs import (
    IBTRACS_SOURCE_NAME,
    group_ibtracs_by_sid,
    ibtracs_rows_to_sources,
    load_atcf_to_sid,
    load_ibtracs_snapshots,
)
from tcfuse.data.sources import MultisourceMetadata, Source, SourceKind, StormData
from tcfuse.utils.archive import submit_archive_job
from tcfuse.utils.time import to_compact_time

_SOURCE_KIND_GROUPS: dict[str, SourceKind] = {
    "scalar": SourceKind.SCALAR,
    "profile": SourceKind.PROFILE,
    "field": SourceKind.FIELD,
}
_ROOT_ATTRS = {"storm_id", "basin", "subbasin", "season", "atcf_id"}

# Satellite-row columns kept in the concatenated assembled index.
_SAT_INDEX_COLUMNS: list[str] = [
    "sid",
    "source_name",
    "snapshot_time_utc",
    "season",
    "basin",
    "subbasin",
]

# Stage 0 outputs live here; not a Stage 1 measurement source.
_IBTRACS_DIR_NAME = "ibtracs"


def _is_stage1_source_dir(entry: Path) -> bool:
    """Return True when a directory holds both Stage-1 metadata and index files."""
    if not entry.is_dir() or entry.name == _IBTRACS_DIR_NAME:
        return False
    return (entry / "metadata.yaml").is_file() and (entry / "index.parquet").is_file()


def _discover_stage1_metadata_yaml_paths(sources_root: Path) -> list[Path]:
    """Collect per-source ``metadata.yaml`` paths under ``sources_root``."""
    return [
        entry / "metadata.yaml"
        for entry in sorted(sources_root.iterdir())
        if _is_stage1_source_dir(entry)
    ]


def _load_stage1_snapshot_index(sources_root: Path) -> pd.DataFrame:
    """Concatenate every Stage-1 ``index.parquet`` under ``sources_root``."""
    frames: list[pd.DataFrame] = []
    for entry in sorted(sources_root.iterdir()):
        if not _is_stage1_source_dir(entry):
            continue
        frames.append(pd.read_parquet(entry / "index.parquet"))
    if not frames:
        return pd.DataFrame(columns=list(_SAT_INDEX_COLUMNS))
    return pd.concat(frames, ignore_index=True)


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
    sid: str,
    rows: pd.DataFrame,
    sources_root: Path,
    assembled_root: Path,
    skip_existing: bool,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    sid_attrs: dict[str, dict[str, Any]],
    atcf_for_sid: dict[str, str],
) -> str | None:
    """Stream all available sources for one storm into an assembled HDF5 file.

    Returns the SID on success, ``None`` when nothing was written.
    """
    info = sid_attrs.get(sid)
    if info is None:
        return None

    basin = info["basin"]
    subbasin = info["subbasin"]
    season = info["season"]
    atcf_id = atcf_for_sid.get(sid)

    dest_path = StormData.path(assembled_root, sid)
    if skip_existing and dest_path.exists():
        return sid

    storm_rows = ibtracs_by_sid.get(sid)
    ibtracs_sources: list[tuple[str, Source]] = []
    if storm_rows is not None and not storm_rows.empty:
        ibtracs_sources = ibtracs_rows_to_sources(storm_rows, sid, basin)

    if rows.empty and not ibtracs_sources:
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written_snapshots = 0
    with h5py.File(dest_path, "w") as dest_file:
        dest_file.attrs["storm_id"] = sid
        dest_file.attrs["basin"] = basin
        dest_file.attrs["subbasin"] = subbasin
        dest_file.attrs["season"] = season
        if atcf_id is not None:
            dest_file.attrs["atcf_id"] = atcf_id

        for _, row in rows.iterrows():
            source_name = str(row["source_name"])
            snapshot_time_utc = str(row["snapshot_time_utc"])
            file_path = Source.path(
                sources_root,
                source_name,
                sid,
                to_compact_time(snapshot_time_utc),
            )
            if not file_path.exists():
                raise FileNotFoundError(f"Snapshot file missing: {file_path}")
            _copy_snapshot_to_assembled(file_path, dest_file, source_name, snapshot_time_utc)
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
    return sid


def _assemble_storms_batch(
    sids: list[str],
    index: pd.DataFrame,
    sources_root: Path,
    assembled_root: Path,
    skip_existing: bool,
    num_workers: int,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    sid_attrs: dict[str, dict[str, Any]],
    atcf_for_sid: dict[str, str],
) -> list[str | None]:
    """Assemble a batch of storms, optionally in parallel."""
    sid_set = set(sids)
    grouped = {sid: grp for sid, grp in index.groupby("sid") if sid in sid_set}
    empty = pd.DataFrame(columns=index.columns)
    rows_per_sid = [grouped.get(sid, empty) for sid in sids]

    if num_workers <= 1:
        return [
            assemble_storm(
                sid,
                rows,
                sources_root,
                assembled_root,
                skip_existing,
                ibtracs_by_sid,
                sid_attrs,
                atcf_for_sid,
            )
            for sid, rows in zip(tqdm(sids, desc="assemble"), rows_per_sid, strict=True)
        ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    assemble_storm,
                    sids,
                    rows_per_sid,
                    repeat(sources_root),
                    repeat(assembled_root),
                    repeat(skip_existing),
                    repeat(ibtracs_by_sid),
                    repeat(sid_attrs),
                    repeat(atcf_for_sid),
                    chunksize=max(1, len(sids) // (num_workers * 4)),
                ),
                total=len(sids),
                desc="assemble",
            )
        )


def _scan_storm_satellite_index(
    assembled_root: Path,
    sids: list[str],
    sid_attrs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Browse every assembled storm file and collect one row per non-IBTrACS snapshot."""
    rows: list[dict[str, Any]] = []
    for sid in sids:
        path = StormData.path(assembled_root, sid)
        if not path.exists():
            continue
        info = sid_attrs.get(sid)
        if info is None:
            continue

        with h5py.File(path, "r") as storm_file:
            for source_name, source_group in storm_file.items():
                if source_name == IBTRACS_SOURCE_NAME:
                    continue
                if not isinstance(source_group, h5py.Group):
                    continue
                for snap_group in source_group.values():
                    if not isinstance(snap_group, h5py.Group):
                        continue
                    snapshot_time_utc = str(snap_group.attrs["snapshot_time_utc"])
                    rows.append(
                        {
                            "sid": sid,
                            "source_name": source_name,
                            "snapshot_time_utc": snapshot_time_utc,
                            "season": info["season"],
                            "basin": info["basin"],
                            "subbasin": info["subbasin"],
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=_SAT_INDEX_COLUMNS)
    return pd.DataFrame(rows, columns=_SAT_INDEX_COLUMNS)


def build_assembled_index(
    ibtracs_snapshots: pd.DataFrame,
    assembled_root: Path,
    assembled_sids: list[str],
    sid_attrs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build the concatenated index of satellite rows + IBTrACS rows.

    Satellite rows carry the trimmed schema in :data:`_SAT_INDEX_COLUMNS`
    (IBTrACS-specific columns are NaN). IBTrACS rows carry the full Stage 0
    schema with ``source_name = "ibtracs_best_track"`` and the IBTrACS
    ``iso_time`` column renamed to ``snapshot_time_utc`` for the union schema.
    """
    sat_index = _scan_storm_satellite_index(assembled_root, assembled_sids, sid_attrs)

    ibt_rows = cast(
        pd.DataFrame,
        ibtracs_snapshots[ibtracs_snapshots["sid"].isin(assembled_sids)].copy(),
    )
    ibt_rows = cast(pd.DataFrame, ibt_rows.rename(columns={"iso_time": "snapshot_time_utc"}))
    ibt_rows["source_name"] = IBTRACS_SOURCE_NAME

    combined = cast(
        pd.DataFrame,
        pd.concat([sat_index, ibt_rows], ignore_index=True),
    )

    # Establish a stable column order: trimmed schema first, then the IBTrACS-
    # specific columns that only exist on best-track rows.
    extra_columns = [c for c in combined.columns if c not in _SAT_INDEX_COLUMNS]
    final_columns = [*_SAT_INDEX_COLUMNS, *extra_columns]
    combined = cast(pd.DataFrame, combined.reindex(columns=final_columns))

    # Fill IBTrACS-specific columns with NaN on satellite rows (already implicit
    # via concat, but make it explicit for downstream consumers reading dtypes).
    for col in extra_columns:
        if combined[col].dtype == object:
            continue
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return cast(
        pd.DataFrame,
        combined.sort_values(["sid", "snapshot_time_utc"]).reset_index(drop=True),
    )


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Assemble all available preprocessed sources into per-storm HDF5 files."""
    cfg = resolve_preproc_cfg(raw_cfg)
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    assembled_root.mkdir(parents=True, exist_ok=True)

    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    print(f"Loading IBTrACS Stage 0 artifacts from {sources_root / 'ibtracs'} …")
    ibtracs_snapshots = load_ibtracs_snapshots(sources_root)
    ibtracs_by_sid = group_ibtracs_by_sid(ibtracs_snapshots)
    translation = load_atcf_to_sid(sources_root)
    print(f"Loaded {len(ibtracs_by_sid)} IBTrACS storms; {len(translation)} ATCF↔SID pairings.")

    # Storm-level constants keyed by SID, derived from the translation table.
    subset_df = cast(pd.DataFrame, translation[["sid", "season", "basin", "subbasin"]])
    keep = cast(pd.DataFrame, subset_df.drop_duplicates(subset=["sid"]))
    sid_attrs: dict[str, dict[str, Any]] = {
        str(rec["sid"]): {
            "season": int(rec["season"]),
            "basin": str(rec["basin"]),
            "subbasin": str(rec["subbasin"]),
        }
        for rec in cast(list[dict[str, Any]], keep.to_dict(orient="records"))
    }
    atcf_for_sid: dict[str, str] = {
        str(rec["sid"]): str(rec["usa_atcf_id"])
        for rec in cast(list[dict[str, Any]], translation.to_dict(orient="records"))
        if str(rec["usa_atcf_id"]).strip() != ""
    }

    print(f"Loading per-source metadata from {sources_root} …")
    yaml_paths = _discover_stage1_metadata_yaml_paths(sources_root)
    multi_meta = MultisourceMetadata.from_multiple_yaml(yaml_paths)
    index = _load_stage1_snapshot_index(sources_root)
    if multi_meta.sources:
        print(
            f"Found {len(multi_meta)} source(s), {len(index)} total snapshots, "
            f"{index['sid'].nunique() if not index.empty else 0} unique SIDs."
        )
    else:
        print("No Stage 1 source indices found; will still write IBTrACS-only storm files.")

    sids = sorted(ibtracs_by_sid.keys())
    if not sids:
        print("No IBTrACS storms to assemble. Nothing to do.")
        return

    if cfg.get("submitit", False):
        from tcfuse.utils.submitit_utils import make_executor

        chunk_size = int(cfg.get("chunk_size", 200))
        chunks = [sids[i : i + chunk_size] for i in range(0, len(sids), chunk_size)]
        slurm_executor = make_executor(cfg, "assemble")
        print(
            f"Submitting {len(chunks)} SLURM jobs ({len(sids)} storms, chunk_size={chunk_size}) …"
        )
        results: list[str | None] = []
        for job in tqdm(
            [
                slurm_executor.submit(
                    _assemble_storms_batch,
                    chunk,
                    index,
                    sources_root,
                    assembled_root,
                    skip_existing,
                    num_workers,
                    ibtracs_by_sid,
                    sid_attrs,
                    atcf_for_sid,
                )
                for chunk in chunks
            ],
            desc="collecting",
        ):
            results.extend(job.result())
    else:
        results = _assemble_storms_batch(
            sids,
            index,
            sources_root,
            assembled_root,
            skip_existing,
            num_workers,
            ibtracs_by_sid,
            sid_attrs,
            atcf_for_sid,
        )

    written = [r for r in results if r is not None]
    skipped = len(results) - len(written)
    print(f"Assembled {len(written)}/{len(sids)} storms → {assembled_root}")
    if skipped:
        print(f"Skipped / empty: {skipped}")

    print("Building assembled index …")
    index_df = build_assembled_index(ibtracs_snapshots, assembled_root, written, sid_attrs)
    index_path = assembled_root / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"Wrote assembled index: {len(index_df)} rows → {index_path}")

    if multi_meta.sources:
        metadata_path = assembled_root / "sources_metadata.yaml"
        multi_meta.to_yaml(metadata_path)
        print(f"Wrote sources metadata: {len(multi_meta)} source(s) → {metadata_path}")

    submit_archive_job(
        assembled_root,
        Path(cfg["paths"]["archives"]["preprocessed_data"]),
        cfg,
        job_name="archive_assembled",
    )


if __name__ == "__main__":
    main()
