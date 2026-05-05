"""Snapshot abstraction: a collection of sources at a single storm observation time."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py

from tcfuse.data.sources.source import _GROUP_TO_KIND, _KIND_TO_GROUP, Source


@dataclasses.dataclass
class Snapshot:
    """A TC snapshot: one or more sources observed at the same nominal time.

    Wraps a mapping from source name to :class:`Source` together with
    storm-level scalar metadata.  Provides dict-like access to sources and
    handles all snapshot-level HDF5 I/O.

    Args:
        sources: Mapping from source_name to Source object.
        meta: Scalar storm metadata (storm_id, basin, snapshot_time_utc,
            lat, lon, vmax_kt, mslp_hpa, …).
    """

    sources: dict[str, Source]
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, source_name: str) -> Source:
        """Return the Source for the given source name."""
        return self.sources[source_name]

    def __contains__(self, source_name: object) -> bool:
        """Return True if the given source name is present in this snapshot."""
        return source_name in self.sources

    def __iter__(self) -> Iterator[str]:
        """Iterate over source names in this snapshot."""
        return iter(self.sources)

    def __len__(self) -> int:
        """Return the number of sources in this snapshot."""
        return len(self.sources)

    def keys(self) -> list[str]:
        """Return an iterator over source names in this snapshot."""
        return list(self.sources.keys())

    def values(self) -> list[Source]:
        """Return an iterator over Source objects in this snapshot."""
        return list(self.sources.values())

    def items(self) -> list[tuple[str, Source]]:
        """Return an iterator over (source_name, Source) pairs in this snapshot."""
        return list(self.sources.items())

    # ------------------------------------------------------------------
    # Snapshot-level HDF5 I/O
    # ------------------------------------------------------------------

    def write(self, path: Path) -> None:
        """Write this snapshot to an HDF5 file.

        Creates parent directories as needed.  Sources absent from
        ``self.sources`` are simply not written (missing-source convention).

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for key, value in self.meta.items():
                f.attrs[key] = value
            for source_name, source in self.sources.items():
                kind_group_name = _KIND_TO_GROUP[source.kind]
                if kind_group_name not in f:
                    f.create_group(kind_group_name)
                src_group = f[kind_group_name].create_group(source_name)  # type: ignore[union-attr]
                source.to_hdf5_group(src_group)

    @classmethod
    def from_disk(
        cls,
        path: Path,
        source_names: list[str] | None = None,
    ) -> Snapshot:
        """Load a snapshot from an HDF5 file.

        Args:
            path: Path to the snapshot ``.h5`` file.
            source_names: If provided, only these sources are loaded.  If
                None, all sources present in the file are loaded.  Sources
                requested but absent from the file are silently skipped
                (missing-source convention).

        Returns:
            :class:`Snapshot` with sources on CPU and root attributes in
            ``meta``.
        """
        sources: dict[str, Source] = {}
        with h5py.File(path, "r") as f:
            meta = dict(f.attrs)
            for kind_group_name, kind in _GROUP_TO_KIND.items():
                if kind_group_name not in f:
                    continue
                for name, group in f[kind_group_name].items():  # type: ignore[union-attr]
                    if source_names is not None and name not in source_names:
                        continue
                    sources[name] = Source.from_hdf5_group(group, kind)
        return cls(sources=sources, meta=meta)

    @staticmethod
    def read_meta(path: Path) -> dict[str, Any]:
        """Read only root-level metadata attributes from a snapshot file.

        Useful for building or refreshing an index without loading source
        tensors.

        Args:
            path: Path to the snapshot ``.h5`` file.

        Returns:
            Dict of root-level HDF5 attributes.
        """
        with h5py.File(path, "r") as f:
            return dict(f.attrs)

    @staticmethod
    def path(
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
