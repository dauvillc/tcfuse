"""Worker functions for the assemble stage.

Kept in a proper importable module (not __main__) so that ProcessPoolExecutor
can pickle assemble_storm by module-qualified name when called from SLURM jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from tcfuse.data.ibtracs import (
    IBTRACS_SOURCE_NAME,
    ibtracs_rows_to_sources,
)
from tcfuse.data.sources import Source, SourceKind, StormData
from tcfuse.data.sources.storm_data import ASSEMBLED_ROOT_ATTRS
from tcfuse.utils.time import to_compact_time

_SOURCE_KIND_GROUPS: dict[str, SourceKind] = {
    "scalar": SourceKind.SCALAR,
    "profile": SourceKind.PROFILE,
    "field": SourceKind.FIELD,
}


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
    time_utc: str,
    meta: dict[str, Any],
) -> None:
    """Write assembled snapshot attributes shared by copied and injected sources."""
    snap_group.attrs["kind"] = kind.name
    snap_group.attrs["time_utc"] = time_utc
    for key, value in meta.items():
        # Storm-level root attrs live on the file root, not per snapshot.
        if key in ASSEMBLED_ROOT_ATTRS or key == "time_utc":
            continue
        snap_group.attrs[key] = value


def _copy_snapshot_to_assembled(
    source_path: Path,
    dest_file: h5py.File,
    source_name: str,
    time_utc: str,
) -> None:
    """Copy one per-source snapshot into an open assembled storm HDF5 file."""
    compact_time = to_compact_time(time_utc)
    source_group = dest_file.require_group(source_name)
    if compact_time in source_group:
        del source_group[compact_time]

    with h5py.File(source_path, "r") as source_file:
        src_group, kind = _find_single_source_group(source_file)
        source_file.copy(src_group, source_group, name=compact_time)
        snap_group = source_group[compact_time]
        if not isinstance(snap_group, h5py.Group):
            raise TypeError(f"Copied snapshot is not an HDF5 group: {source_path}")
        _set_snapshot_attrs(snap_group, kind, time_utc, dict(source_file.attrs))


def _write_source_to_assembled(
    source: Source,
    dest_file: h5py.File,
    source_name: str,
    time_utc: str,
) -> None:
    """Write one in-memory Source into an open assembled storm HDF5 file."""
    compact_time = to_compact_time(time_utc)
    source_group = dest_file.require_group(source_name)
    if compact_time in source_group:
        del source_group[compact_time]
    snap_group = source_group.create_group(compact_time)
    source.to_hdf5_group(snap_group)
    _set_snapshot_attrs(snap_group, source.kind, time_utc, source.meta)


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
            time_utc = str(row["time_utc"])
            file_path = Source.path(
                sources_root,
                source_name,
                sid,
                to_compact_time(time_utc),
            )
            if not file_path.exists():
                raise FileNotFoundError(f"Snapshot file missing: {file_path}")
            _copy_snapshot_to_assembled(file_path, dest_file, source_name, time_utc)
            written_snapshots += 1

        for time_utc, source in ibtracs_sources:
            _write_source_to_assembled(
                source,
                dest_file,
                IBTRACS_SOURCE_NAME,
                time_utc,
            )
            written_snapshots += 1

    if written_snapshots == 0:
        # Remove empty partial writes when no satellite or IBTrACS snapshots land.
        dest_path.unlink(missing_ok=True)
        return None
    return sid
