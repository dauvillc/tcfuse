"""NetCDF / HDF5 I/O helpers.

On-disk format
--------------
Preprocessed data is organised **source-first**: each source has its own
directory under ``cfg.paths.preprocessed_sources/<source_name>/``, which
contains an ``index.parquet`` and a ``snapshots/`` sub-directory.

Each snapshot is a single HDF5 file named
``{storm_id}_{snapshot_time_utc}.h5`` (e.g. ``AL012020_20200801T120000Z.h5``)
and holds **exactly one source**.

Root-level attributes carry snapshot metadata (storm_id, basin, vmax_kt, …).
The source is stored under one of three top-level groups — ``/scalar``,
``/profile``, ``/field`` — matching the three SourceKind values.  Within each
group, the source is a sub-group named by its ``source_name``.  A source
sub-group contains:

* ``values``  — float32 array; shape (C,) / (L, C) / (H, W, C)
* ``coords``  — float64 array; shape (3,) / (L, 4) / (H, W, 3)
                (float32 for FIELD coords to save space)
* ``mask``    — bool array (same leading shape as values); present only when
                missing data exists.
* Attribute ``source_name`` — string identifier.
* Attribute ``channels``    — JSON-encoded list of channel name strings.

The ``index.parquet`` file alongside ``snapshots/`` provides fast lookup
without opening individual HDF5 files.  Use :func:`source_snapshot_path` to
compute the canonical path for any source snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from tcfuse.data.sources.base import MultisourceMetadata, Source, SourceKind, SourceMetadata

# Map SourceKind to top-level HDF5 group name.
_KIND_TO_GROUP: dict[SourceKind, str] = {
    SourceKind.SCALAR: "scalar",
    SourceKind.PROFILE: "profile",
    SourceKind.FIELD: "field",
}
_GROUP_TO_KIND: dict[str, SourceKind] = {v: k for k, v in _KIND_TO_GROUP.items()}

_FLOAT_COMPRESSION = {"compression": "gzip", "compression_opts": 4}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def source_snapshot_path(
    sources_root: Path,
    source_name: str,
    storm_id: str,
    snapshot_time_utc: str,
) -> Path:
    """Return the canonical path for a single-source snapshot file.

    Args:
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        source_name: Source identifier, e.g. ``"pmw_amsr2_gcomw1"``.
        storm_id: Storm identifier, e.g. ``"2016AL10"``.
        snapshot_time_utc: Compact UTC timestamp string,
            e.g. ``"20160912T010942Z"``.

    Returns:
        Absolute path to the snapshot HDF5 file.
    """
    return sources_root / source_name / "snapshots" / f"{storm_id}_{snapshot_time_utc}.h5"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def write_source(group: h5py.Group, source: Source) -> None:
    """Write a Source into an HDF5 group.

    The group should already be a sub-group named by the source's source_name
    (created by the caller).  Datasets ``values`` and ``coords`` are always
    written; ``mask`` is written only when ``source.mask is not None``.

    Args:
        group: Open, writable h5py Group for this source.
        source: Source object to serialise.
    """
    group.create_dataset(
        "values",
        data=source.values.detach().cpu().numpy().astype(np.float32),
        **_FLOAT_COMPRESSION,
    )
    # FIELD coords stored as float32 (lat/lon precision sufficient); others float64.
    coord_dtype = np.float32 if source.kind is SourceKind.FIELD else np.float64
    group.create_dataset(
        "coords",
        data=source.coords.detach().cpu().numpy().astype(coord_dtype),
        **_FLOAT_COMPRESSION,
    )
    if source.mask is not None:
        group.create_dataset(
            "mask",
            data=source.mask.detach().cpu().numpy().astype(bool),
        )
    group.attrs["source_name"] = source.source_name
    group.attrs["channels"] = json.dumps(source.channels)


def read_source(group: h5py.Group, kind: SourceKind) -> Source:
    """Read a Source from an HDF5 group previously written by :func:`write_source`.

    Args:
        group: Open h5py Group for this source.
        kind: SourceKind of this source (determined by parent group name).

    Returns:
        Reconstructed Source with tensors on CPU.
    """
    values = torch.from_numpy(np.array(group["values"], dtype=np.float32))
    coord_dtype = np.float32 if kind is SourceKind.FIELD else np.float64
    coords = torch.from_numpy(np.array(group["coords"], dtype=coord_dtype))
    mask: torch.Tensor | None = None
    if "mask" in group:
        mask = torch.from_numpy(np.array(group["mask"], dtype=bool))
    source_name = str(group.attrs["source_name"])
    channels: list[str] = json.loads(str(group.attrs["channels"]))
    return Source(
        kind=kind,
        values=values,
        coords=coords,
        source_name=source_name,
        channels=channels,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Snapshot-level helpers
# ---------------------------------------------------------------------------


def write_snapshot(
    path: Path,
    meta: dict[str, Any],
    sources: dict[str, Source],
) -> None:
    """Write a complete TC snapshot to an HDF5 file.

    Args:
        path: Destination file path (parent directory must exist).
        meta: Scalar metadata stored as root attributes.  Recommended keys:
            storm_id, basin, snapshot_time_utc (ISO-8601 str), lat, lon,
            vmax_kt, mslp_hpa.
        sources: Mapping from source_name to Source object.  Sources absent
            from this dict are simply not written (missing-source convention).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in meta.items():
            f.attrs[key] = value
        for source_name, source in sources.items():
            kind_group_name = _KIND_TO_GROUP[source.kind]
            if kind_group_name not in f:
                f.create_group(kind_group_name)
            src_group = f[kind_group_name].create_group(source_name)  # type: ignore[union-attr]
            write_source(src_group, source)


def read_snapshot(
    path: Path,
    source_names: list[str] | None = None,
) -> dict[str, Source]:
    """Read sources from a snapshot HDF5 file.

    Args:
        path: Path to the snapshot ``.h5`` file.
        source_names: If provided, only these sources are loaded.  If None,
            all sources present in the file are loaded.

    Returns:
        Dict mapping source_name to Source.  Sources requested but absent from
        the file are silently skipped (missing-source convention).
    """
    result: dict[str, Source] = {}
    with h5py.File(path, "r") as f:
        for kind_group_name, kind in _GROUP_TO_KIND.items():
            if kind_group_name not in f:
                continue
            for name, group in f[kind_group_name].items():  # type: ignore[union-attr]
                if source_names is not None and name not in source_names:
                    continue
                result[name] = read_source(group, kind)
    return result


def read_snapshot_meta(path: Path) -> dict[str, Any]:
    """Read only root-level metadata attributes from a snapshot file.

    Useful for building or refreshing an index without loading source tensors.

    Args:
        path: Path to the snapshot ``.h5`` file.

    Returns:
        Dict of root-level HDF5 attributes.
    """
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


# ---------------------------------------------------------------------------
# Source directory helpers
# ---------------------------------------------------------------------------


def write_source_metadata(
    sources_root: Path,
    source_name: str,
    source_type: str,
    source_kind: SourceKind,
    channels: list[str],
) -> None:
    """Write a metadata.yaml file describing a preprocessed source directory.

    The file is written to ``{sources_root}/{source_name}/metadata.yaml`` and
    contains the information needed by downstream consumers (e.g. the ML
    pipeline) to understand a source without opening individual HDF5 snapshots.

    Args:
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        source_name: Source directory name, e.g. ``"radar_gmi"``.
        source_type: Physical category of the source, e.g. ``"radar"``,
            ``"microwave"``, ``"infrared"``.
        source_kind: Dimensionality class (SCALAR, PROFILE, or FIELD).
        channels: Ordered list of channel names matching the last axis of
            the ``values`` array in each snapshot.
    """
    meta = {
        "name": source_name,
        "type": source_type,
        "kind": source_kind.name.lower(),
        "channels": channels,
        "num_channels": len(channels),
    }
    meta_path = sources_root / source_name / "metadata.yaml"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


def read_source_metadata(
    sources_root: Path,
    source_name: str,
) -> SourceMetadata:
    """Load source-level metadata and snapshot index from disk.

    Reads ``{sources_root}/{source_name}/metadata.yaml`` and
    ``{sources_root}/{source_name}/index.parquet``.

    Args:
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).
        source_name: Source directory name, e.g. ``"pmw_amsr2_gcomw1"``.

    Returns:
        :class:`SourceMetadata` with the snapshot index fully loaded into memory.
    """
    meta_path = sources_root / source_name / "metadata.yaml"
    with open(meta_path) as f:
        raw = yaml.safe_load(f)

    source_kind = SourceKind[raw["kind"].upper()]
    index = pd.read_parquet(sources_root / source_name / "index.parquet")

    return SourceMetadata(
        name=raw["name"],
        type=raw["type"],
        kind=source_kind,
        channels=raw["channels"],
        index=index,
    )


def read_multisource_metadata(sources_root: str | Path) -> MultisourceMetadata:
    """Load metadata for all sources found under sources_root.

    Scans for sub-directories that contain both ``metadata.yaml`` and
    ``index.parquet``, skipping any that are missing either file.

    Args:
        sources_root: Root directory for preprocessed sources
            (``cfg.paths.preprocessed_sources``).

    Returns:
        A :class:`MultisourceMetadata` containing one entry per valid
        source directory found, with a merged snapshot index.
    """
    sources_root = Path(sources_root)
    sources: dict[str, SourceMetadata] = {}
    for entry in sorted(sources_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "metadata.yaml").exists():
            continue
        if not (entry / "index.parquet").exists():
            continue
        sources[entry.name] = read_source_metadata(sources_root, entry.name)
    return MultisourceMetadata(sources=sources)
