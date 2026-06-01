"""Source-level and multi-source metadata, with YAML I/O."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from tcfuse.data.sources.source import SourceKind


@dataclasses.dataclass
class SourceMetadata:
    """Source-level metadata: physical description of a measurement source.

    This object holds static descriptors shared by every snapshot of a source.
    It does NOT contain measurement tensors (see :class:`~tcfuse.data.sources.source.Source`)
    or snapshot indices (see per-source ``index.parquet`` or the assembled index).

    Args:
        name: Source directory name, e.g. ``"pmw_amsr2_gcomw1"``.
        type: Physical category, e.g. ``"microwave"``, ``"radar"``.
        kind: Dimensionality class (SCALAR, PROFILE, or FIELD).
        channels: Ordered list of channel names matching the last axis of
            the ``values`` array in each snapshot.
        shape: Spatial shape shared by every snapshot of this source (excluding channels).
            - SCALAR:  ``()``
            - PROFILE: ``(L,)``
            - FIELD:   ``(H, W)``
            All snapshots within a source are guaranteed to share this shape.
        char_vars: Instrument-level descriptor variables constant across all snapshots
            of this source (e.g. ``{"ifov": {"tb_89.0h": [7.2, 4.4, 7.2, 4.4]}}``).
            Values must be JSON-serialisable (lists, dicts, scalars).
    """

    name: str
    type: str
    kind: SourceKind
    channels: list[str]
    shape: tuple[int, ...]
    char_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def num_channels(self) -> int:
        """Number of channels (last axis of ``values`` in each snapshot)."""
        return len(self.channels)

    @property
    def num_tokens(self) -> int:
        """Number of spatial tokens per snapshot (flattened spatial dims).

        Returns 1 for SCALAR sources (empty shape).
        """
        # math.prod(()) == 1, which is correct for SCALAR.
        return max(1, math.prod(self.shape))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serialisable dict for this source."""
        return {
            "name": self.name,
            "type": self.type,
            "kind": self.kind.name.lower(),
            "channels": self.channels,
            "num_channels": self.num_channels,
            "shape": list(self.shape),
            "char_vars": self.char_vars,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceMetadata:
        """Build a :class:`SourceMetadata` from a dict (one source entry)."""
        source_kind = SourceKind[str(raw["kind"]).upper()]
        shape = tuple(int(d) for d in raw["shape"])
        return cls(
            name=str(raw["name"]),
            type=str(raw["type"]),
            kind=source_kind,
            channels=list(raw["channels"]),
            shape=shape,
            char_vars=dict(raw.get("char_vars") or {}),
        )

    def to_yaml(self, yaml_path: Path) -> None:
        """Write this source's metadata to a YAML file.

        Args:
            yaml_path: Full path to the output ``metadata.yaml`` file.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> SourceMetadata:
        """Load source metadata from a YAML file.

        Args:
            yaml_path: Full path to a per-source ``metadata.yaml`` file.

        Returns:
            :class:`SourceMetadata` with static descriptors only (no index).
        """
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a mapping in {yaml_path}, got {type(raw).__name__}")
        return cls.from_dict(raw)


@dataclasses.dataclass
class MultisourceMetadata:
    """Grouped metadata for a collection of sources.

    Wraps multiple :class:`SourceMetadata` objects. Snapshot indices live in
    ``index.parquet`` files on disk, not in this class.

    Args:
        sources: Mapping from source name to its :class:`SourceMetadata`.
    """

    sources: dict[str, SourceMetadata]

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
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return ``{source_name: source_dict}`` suitable for YAML export."""
        return {name: meta.to_dict() for name, meta in self.sources.items()}

    @classmethod
    def from_dict(cls, raw: dict[str, dict[str, Any]]) -> MultisourceMetadata:
        """Build from ``{source_name: source_metadata_dict}``."""
        sources = {name: SourceMetadata.from_dict(entry) for name, entry in raw.items()}
        return cls(sources=sources)

    def to_yaml(self, yaml_path: Path) -> None:
        """Write merged source metadata to a YAML file.

        Args:
            yaml_path: Full path to the output file (e.g. ``sources_metadata.yaml``).
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> MultisourceMetadata:
        """Load multi-source metadata from a single YAML file.

        The file must contain a mapping ``{source_name: source_metadata_dict}``.

        Args:
            yaml_path: Full path to a multi-source metadata YAML file.

        Returns:
            :class:`MultisourceMetadata` with one entry per source key.
        """
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a mapping in {yaml_path}, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def from_multiple_yaml(cls, yaml_paths: list[Path]) -> MultisourceMetadata:
        """Load and union metadata from several per-source YAML files.

        Each path is loaded with :meth:`SourceMetadata.from_yaml`. Later paths
        overwrite earlier ones when source names collide.

        Args:
            yaml_paths: List of per-source ``metadata.yaml`` file paths.

        Returns:
            :class:`MultisourceMetadata` containing the union of all sources.
        """
        merged: dict[str, dict[str, Any]] = {}
        for path in yaml_paths:
            meta = SourceMetadata.from_yaml(path)
            merged[meta.name] = meta.to_dict()
        return cls.from_dict(merged)
