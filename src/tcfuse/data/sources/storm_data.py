"""StormData: assembled multi-source container for a single tropical cyclone."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from tcfuse.data.sources.source import Source, SourceKind
from tcfuse.data.window_index import snapshot_in_window
from tcfuse.utils.time import to_compact_time

# HDF5 attribute keys written at the root level of each assembled file.
_ROOT_ATTRS = ("storm_id", "basin", "subbasin", "season", "atcf_id")
_STORM_DATA_DIR = "storm_data"


@dataclass
class StormData:
    """All preprocessed sources for a single tropical cyclone.

    Sources are indexed by ``(source_name, snapshot_time_utc)`` because the same
    instrument can produce multiple overpasses for a storm at different times.
    The ``snapshot_time_utc`` key is the isoformat string as it appears in the
    per-source ``index.parquet`` files.

    ``season`` is the TC season year (e.g. 2016). It is the primary axis used
    for train/val/test splits.

    Args:
        storm_id: IBTrACS SID, e.g. ``"2016292N14270"``.
        basin: Ocean basin code, e.g. ``"AL"``.
        subbasin: IBTrACS sub-basin code, e.g. ``"GM"``.
        season: TC season year, e.g. 2016.
        sources: Mapping from ``(source_name, snapshot_time_utc)`` to the
            corresponding :class:`~tcfuse.data.sources.source.Source`.
            FIELD, PROFILE, and SCALAR sources may coexist.
        atcf_id: USA ATCF identifier when available (e.g. ``"AL102016"``).
    """

    storm_id: str
    basin: str
    subbasin: str
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
            storm_id: IBTrACS SID, e.g. ``"2016292N14270"``.

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
            ├── attrs: {storm_id, basin, subbasin, season, atcf_id?}
            └── {source_name}/
                └── {compact_snapshot_time}/
                    ├── values      float32, gzip-4
                    ├── coords      float32 (FIELD) or float64 (others), gzip-4
                    ├── mask        bool, same shape as values (per-value availability)
                    └── attrs:
                        ├── source_name        str
                        ├── channels           JSON list
                        ├── kind               "SCALAR" | "PROFILE" | "FIELD"
                        ├── snapshot_time_utc  isoformat str (for round-trip key recovery)
                        └── [other meta]       lat, lon, … from Source.meta

        Args:
            assembled_root: Root directory for assembled storm files
                (``cfg.paths.preprocessed_data``).
        """
        dest = StormData.path(assembled_root, self.storm_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(dest, "w") as f:
            f.attrs["storm_id"] = self.storm_id
            f.attrs["basin"] = self.basin
            f.attrs["subbasin"] = self.subbasin
            f.attrs["season"] = self.season
            if self.atcf_id is not None:
                f.attrs["atcf_id"] = self.atcf_id

            for (source_name, snapshot_time_utc), source in self.sources.items():
                compact_time = to_compact_time(snapshot_time_utc)
                snap_group = f.require_group(f"{source_name}/{compact_time}")

                source.to_hdf5_group(snap_group)
                snap_group.attrs["kind"] = source.kind.name
                snap_group.attrs["snapshot_time_utc"] = snapshot_time_utc

                for key, value in source.meta.items():
                    if key not in _ROOT_ATTRS and key != "snapshot_time_utc":
                        try:
                            snap_group.attrs[key] = value
                        except TypeError:
                            warnings.warn(
                                f"Could not write meta key '{key}' as HDF5 attr: {type(value)}",
                                stacklevel=2,
                            )

    @classmethod
    def from_disk(
        cls,
        assembled_root: Path,
        storm_id: str,
        *,
        window_start_utc: str | None = None,
        window_end_utc: str | None = None,
    ) -> StormData:
        """Load sources for a storm from its assembled HDF5 file.

        Reconstructs the ``sources`` dict with ``(source_name, snapshot_time_utc)``
        keys matching the original isoformat strings, so the keys are compatible
        with the per-source ``index.parquet`` index.

        When ``window_start_utc`` and ``window_end_utc`` are both provided, only
        snapshots whose ``snapshot_time_utc`` falls in the closed interval
        ``[window_start_utc, window_end_utc]`` are loaded.

        Args:
            assembled_root: Root directory for assembled storm files.
            storm_id: IBTrACS SID, e.g. ``"2016292N14270"``.
            window_start_utc: Optional inclusive window lower bound.
            window_end_utc: Optional inclusive window upper bound.

        Returns:
            :class:`StormData` with matching sources loaded into CPU tensors.

        Raises:
            KeyError: When the file is missing any of the mandatory root attrs
                (``storm_id``, ``basin``, ``subbasin``, ``season``).
            ValueError: When only one of ``window_start_utc`` / ``window_end_utc``
                is provided.
        """
        if (window_start_utc is None) ^ (window_end_utc is None):
            raise ValueError(
                "window_start_utc and window_end_utc must both be set or both be None."
            )

        path = StormData.path(assembled_root, storm_id)
        sources: dict[tuple[str, str], Source] = {}

        with h5py.File(path, "r") as f:
            loaded_storm_id = str(f.attrs["storm_id"])
            basin = str(f.attrs["basin"])
            subbasin = str(f.attrs["subbasin"])
            season = int(f.attrs["season"])
            atcf_id: str | None = str(f.attrs["atcf_id"]) if "atcf_id" in f.attrs else None

            for source_name, source_group in f.items():
                if not isinstance(source_group, h5py.Group):
                    continue
                for _compact_time, snap_group in source_group.items():
                    if not isinstance(snap_group, h5py.Group):
                        continue

                    snapshot_time_utc = str(snap_group.attrs["snapshot_time_utc"])
                    if window_start_utc is not None and window_end_utc is not None:
                        if not snapshot_in_window(
                            snapshot_time_utc,
                            window_start_utc,
                            window_end_utc,
                        ):
                            continue

                    kind = SourceKind[str(snap_group.attrs["kind"])]
                    source = Source.from_hdf5_group(snap_group, kind)

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
            subbasin=subbasin,
            season=season,
            sources=sources,
            atcf_id=atcf_id,
        )

    @staticmethod
    def read_meta(assembled_root: Path, storm_id: str) -> dict[str, Any]:
        """Read only root-level attributes without loading any tensors.

        Returns ``{storm_id, basin, subbasin, season}`` and, when present,
        ``atcf_id`` — the storm-level constants stored at the root of the
        assembled HDF5 file.

        Args:
            assembled_root: Root directory for assembled storm files.
            storm_id: IBTrACS SID, e.g. ``"2016292N14270"``.

        Returns:
            Dict of root-level HDF5 attributes.
        """
        path = StormData.path(assembled_root, storm_id)
        with h5py.File(path, "r") as f:
            return dict(f.attrs)
