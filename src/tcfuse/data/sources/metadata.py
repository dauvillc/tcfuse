"""Source-level and multi-source metadata, with disk I/O."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tcfuse.data.sources.source import SourceKind


@dataclasses.dataclass
class SourceMetadata:
    """Source-level metadata: physical description + full snapshot index.

    This object holds everything that is known about a source across all its
    snapshots.  It does NOT contain any measurement tensors — those live in
    individual :class:`~tcfuse.data.sources.source.Source` objects loaded on demand.

    Args:
        name: Source directory name, e.g. ``"pmw_amsr2_gcomw1"``.
        type: Physical category, e.g. ``"microwave"``, ``"radar"``.
        kind: Dimensionality class (SCALAR, PROFILE, or FIELD).
        channels: Ordered list of channel names matching the last axis of
            the ``values`` array in each snapshot.
        index: Source index loaded from ``index.parquet``.  Each row
            corresponds to one HDF5 snapshot file; columns include at least
            ``storm_id``, ``snapshot_time_utc``, ``lat``, ``lon``,
            ``source_name``, and ``file_path``.
        char_vars: Instrument-level descriptor variables constant across all snapshots
            of this source (e.g. ``{"ifov": {"tb_89.0h": [7.2, 4.4, 7.2, 4.4]}}``).
            Values must be JSON-serialisable (lists, dicts, scalars).
    """

    name: str
    type: str
    kind: SourceKind
    channels: list[str]
    index: pd.DataFrame = dataclasses.field(compare=False)
    char_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def num_channels(self) -> int:
        """Number of channels (last axis of ``values`` in each snapshot)."""
        return len(self.channels)

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def write(self, sources_root: Path) -> None:
        """Write ``metadata.yaml`` and ``index.parquet`` for this source.

        Both files are written to ``{sources_root}/{self.name}/``.  The
        directory is created if it does not exist.

        Args:
            sources_root: Root directory for preprocessed sources
                (``cfg.paths.preprocessed_sources``).
        """
        dest = sources_root / self.name
        dest.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "type": self.type,
            "kind": self.kind.name.lower(),
            "channels": self.channels,
            "num_channels": self.num_channels,
            "char_vars": self.char_vars,
        }
        with open(dest / "metadata.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        self.index.to_parquet(dest / "index.parquet", index=False)

    @classmethod
    def from_disk(cls, sources_root: Path, source_name: str) -> SourceMetadata:
        """Load source-level metadata and snapshot index from disk.

        Reads ``{sources_root}/{source_name}/metadata.yaml`` and
        ``{sources_root}/{source_name}/index.parquet``.

        Args:
            sources_root: Root directory for preprocessed sources
                (``cfg.paths.preprocessed_sources``).
            source_name: Source directory name, e.g. ``"pmw_amsr2_gcomw1"``.

        Returns:
            :class:`SourceMetadata` with the snapshot index fully loaded into
            memory.
        """
        meta_path = sources_root / source_name / "metadata.yaml"
        with open(meta_path) as f:
            raw = yaml.safe_load(f)

        source_kind = SourceKind[raw["kind"].upper()]
        index = pd.read_parquet(sources_root / source_name / "index.parquet")
        # Backward-compatible: older metadata.yaml files written before char_vars was introduced.
        char_vars: dict[str, Any] = raw.get("char_vars") or {}

        return cls(
            name=raw["name"],
            type=raw["type"],
            kind=source_kind,
            channels=raw["channels"],
            index=index,
            char_vars=char_vars,
        )


@dataclasses.dataclass
class MultisourceMetadata:
    """Grouped metadata for a collection of sources.

    Wraps multiple :class:`SourceMetadata` objects and exposes a merged
    snapshot index across all sources.

    Args:
        sources: Mapping from source name to its :class:`SourceMetadata`.
    """

    sources: dict[str, SourceMetadata]

    def __post_init__(self) -> None:
        """Merge individual source indices into a single DataFrame for easy querying."""
        self._index: pd.DataFrame = (
            pd.concat(
                [meta.index for meta in self.sources.values()],
                ignore_index=True,
            )
            if self.sources
            else pd.DataFrame()
        )

    @property
    def index(self) -> pd.DataFrame:
        """Merged snapshot index across all sources (one row per snapshot per source)."""
        return self._index

    def __getitem__(self, source_name: str) -> SourceMetadata:
        """Return the SourceMetadata for the given source name."""
        return self.sources[source_name]

    def __len__(self) -> int:
        """Return the number of sources."""
        return len(self.sources)

    def __iter__(self) -> Iterator[str]:
        """Iterate over source names."""
        return iter(self.sources)

    def __contains__(self, source_name: object) -> bool:
        """Return True if source_name is present."""
        return source_name in self.sources

    @property
    def names(self) -> list[str]:
        """Ordered list of source names."""
        return list(self.sources.keys())

    def filter_by_source_type(self, source_types: str | list[str]) -> MultisourceMetadata:
        """Return a new MultisourceMetadata restricted to sources whose type matches.

        Args:
            source_types: A single type string (e.g. ``"microwave"``) or a list of
                type strings. Only sources whose :attr:`SourceMetadata.type` appears
                in this set are included in the returned object.

        Returns:
            A new :class:`MultisourceMetadata` containing only the matching sources.
        """
        if isinstance(source_types, str):
            source_types = [source_types]
        allowed = set(source_types)
        filtered = {name: meta for name, meta in self.sources.items() if meta.type in allowed}
        return MultisourceMetadata(sources=filtered)

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(cls, sources_root: Path) -> MultisourceMetadata:
        """Load metadata for all sources found under ``sources_root``.

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
            sources[entry.name] = SourceMetadata.from_disk(sources_root, entry.name)
        return cls(sources=sources)
