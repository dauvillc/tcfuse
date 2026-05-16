"""StormData: assembled multi-source container for a single tropical cyclone."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from tcfuse.data.sources.source import Source, SourceKind

# HDF5 attribute keys written at the root level of each assembled file.
_ROOT_ATTRS = ("storm_id", "basin", "season", "atcf_id")
_STORM_DATA_DIR = "storm_data"


def _to_compact_time(snapshot_time_utc: str) -> str:
    """Convert an isoformat timestamp to a compact, HDF5-safe group name.

    Args:
        snapshot_time_utc: ISO 8601 timestamp, e.g. ``"2016-09-12T01:09:42+00:00"``.

    Returns:
        Compact string without separators or timezone offset,
        e.g. ``"20160912T010942Z"``.
    """
    return pd.Timestamp(snapshot_time_utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class StormData:
    """All preprocessed sources for a single tropical cyclone.

    Sources are indexed by ``(source_name, snapshot_time_utc)`` because the same
    instrument can produce multiple overpasses for a storm at different times.
    The ``snapshot_time_utc`` key is the isoformat string as it appears in the
    per-source ``index.parquet`` files.

    ``season`` is the TC season year (e.g. 2016).  It is the primary axis used
    for train/val/test splits.

    Args:
        storm_id: Storm identifier, e.g. ``"2016AL10"``.
        basin: Ocean basin code, e.g. ``"AL"``.
        season: TC season year, e.g. ``2016``.  Derived from ``storm_id[:4]``.
        sources: Mapping from ``(source_name, snapshot_time_utc)`` to the
            corresponding :class:`~tcfuse.data.sources.source.Source`.
            Both FIELD, PROFILE, and SCALAR sources may coexist.
    """

    storm_id: str
    basin: str
    season: int
    sources: dict[tuple[str, str], Source] = dataclasses.field(default_factory=dict)
    atcf_id: str | None = None

    # ------------------------------------------------------------------
    # Canonical path helper
    # ------------------------------------------------------------------

    @staticmethod
    def path(assembled_root: Path, storm_id: str) -> Path:
        """Return the canonical path for a storm's assembled HDF5 file.

        Args:
            assembled_root: Root directory for assembled storm files
                (``cfg.paths.preprocessed_data``).
            storm_id: Storm identifier, e.g. ``"2016AL10"``.

        Returns:
            ``{assembled_root}/storm_data/{storm_id}.h5``
        """
        return assembled_root / _STORM_DATA_DIR / f"{storm_id}.h5"

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def write(self, assembled_root: Path) -> None:
        """Write all sources to a single assembled HDF5 file.

        Creates ``{assembled_root}/storm_data/{storm_id}.h5`` with the following layout::

            /
            ├── attrs: {storm_id, basin, season}
            └── {source_name}/
                └── {compact_snapshot_time}/
                    ├── values      float32, gzip-4
                    ├── coords      float32 (FIELD) or float64 (others), gzip-4
                    ├── [mask]      bool (only when Source.mask is not None)
                    └── attrs:
                        ├── source_name        str
                        ├── channels           JSON list
                        ├── kind               "SCALAR" | "PROFILE" | "FIELD"
                        ├── snapshot_time_utc  isoformat str (for round-trip key recovery)
                        └── [other meta]       lat, lon, vmax_kt, … from Source.meta

        Args:
            assembled_root: Root directory for assembled storm files
                (``cfg.paths.preprocessed_data``).
        """
        dest = StormData.path(assembled_root, self.storm_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(dest, "w") as f:
            # Write storm-level constants as root attributes.
            f.attrs["storm_id"] = self.storm_id
            f.attrs["basin"] = self.basin
            f.attrs["season"] = self.season
            if self.atcf_id is not None:
                f.attrs["atcf_id"] = self.atcf_id

            # Write each (source_name, snapshot_time_utc) snapshot.
            for (source_name, snapshot_time_utc), source in self.sources.items():
                compact_time = _to_compact_time(snapshot_time_utc)
                snap_group = f.require_group(f"{source_name}/{compact_time}")

                # Delegate tensor serialization to the existing Source method.
                source.to_hdf5_group(snap_group)

                # Add kind so from_disk can reconstruct SourceKind without extra structure.
                snap_group.attrs["kind"] = source.kind.name

                # Store the original isoformat timestamp for exact key round-trip.
                snap_group.attrs["snapshot_time_utc"] = snapshot_time_utc

                # Write any remaining Source.meta entries as snapshot-level attrs.
                for key, value in source.meta.items():
                    if key not in _ROOT_ATTRS and key != "snapshot_time_utc":
                        try:
                            snap_group.attrs[key] = value
                        except TypeError:
                            # Skip values that h5py cannot serialise as attributes.
                            warnings.warn(
                                f"Could not write meta key '{key}' as HDF5 attr: {type(value)}",
                                stacklevel=2,
                            )

    @classmethod
    def from_disk(cls, assembled_root: Path, storm_id: str) -> StormData:
        """Load all sources for a storm from its assembled HDF5 file.

        Reconstructs the ``sources`` dict with ``(source_name, snapshot_time_utc)``
        keys matching the original isoformat strings, so the keys are compatible
        with the per-source ``index.parquet`` index.

        Args:
            assembled_root: Root directory for assembled storm files
                (``cfg.paths.preprocessed_data``).
            storm_id: Storm identifier, e.g. ``"2016AL10"``.

        Returns:
            :class:`StormData` with all sources loaded into CPU tensors.
        """
        path = StormData.path(assembled_root, storm_id)
        sources: dict[tuple[str, str], Source] = {}

        with h5py.File(path, "r") as f:
            # Read storm-level constants from root attrs.
            loaded_storm_id = str(f.attrs["storm_id"])
            basin = str(f.attrs["basin"])
            season = int(f.attrs["season"])
            atcf_id: str | None = str(f.attrs["atcf_id"]) if "atcf_id" in f.attrs else None

            # Iterate source_name groups, then compact_time sub-groups.
            for source_name, source_group in f.items():
                if not isinstance(source_group, h5py.Group):
                    continue
                for _compact_time, snap_group in source_group.items():
                    if not isinstance(snap_group, h5py.Group):
                        continue

                    # Recover SourceKind from the stored attr.
                    kind = SourceKind[str(snap_group.attrs["kind"])]

                    # Reconstruct tensors from the snapshot group.
                    source = Source.from_hdf5_group(snap_group, kind)

                    # Recover snapshot_time_utc in its original isoformat.
                    snapshot_time_utc = str(snap_group.attrs["snapshot_time_utc"])

                    # Populate Source.meta from any remaining snapshot attrs.
                    meta: dict[str, Any] = {
                        "storm_id": loaded_storm_id,
                        "basin": basin,
                        "snapshot_time_utc": snapshot_time_utc,
                    }
                    skip_keys = {"source_name", "channels", "kind", "snapshot_time_utc"}
                    for key in snap_group.attrs:
                        if key not in skip_keys:
                            meta[key] = snap_group.attrs[key]
                    source.meta = meta

                    sources[(source_name, snapshot_time_utc)] = source

        return cls(
            storm_id=loaded_storm_id,
            basin=basin,
            season=season,
            sources=sources,
            atcf_id=atcf_id,
        )

    @staticmethod
    def read_meta(assembled_root: Path, storm_id: str) -> dict[str, Any]:
        """Read only root-level attributes without loading any tensors.

        Returns ``{storm_id, basin, season}`` and, when present, ``atcf_id`` —
        the storm-level constants stored at the root of the assembled HDF5 file.

        Args:
            assembled_root: Root directory for assembled storm files
                (``cfg.paths.preprocessed_data``).
            storm_id: Storm identifier, e.g. ``"2016AL10"``.

        Returns:
            Dict with keys ``"storm_id"``, ``"basin"``, and ``"season"``.
        """
        path = StormData.path(assembled_root, storm_id)
        with h5py.File(path, "r") as f:
            return dict(f.attrs)
