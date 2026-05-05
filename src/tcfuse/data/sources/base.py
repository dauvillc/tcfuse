"""Base source abstraction: a collection of (value, coordinate) pairs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd
import torch
from torch import Tensor


class SourceKind(Enum):
    """Dimensionality class of a source."""

    SCALAR = auto()  # 0D: single point measurement
    PROFILE = auto()  # 1D: vertical profile (L levels)
    FIELD = auto()  # 2D: image or gridded field (H x W)


@dataclass
class Source:
    """A single observation source: values paired with explicit spatio-temporal coordinates.

    All coordinates are continuous and physical — no learned bin embeddings.
    Missing values are represented as NaN and must propagate correctly through
    any embedding or loss computation.

    Args:
        kind: Dimensionality class of this source.
        values: Observed measurements.
            - SCALAR:  (C,)
            - PROFILE: (L, C)   — L levels, C channels
            - FIELD:   (H, W, C)
        coords: Spatio-temporal coordinates paired with each measurement.
            - SCALAR:  (3,)         — [time, lat, lon]
            - PROFILE: (L, 4)       — [time, lat, lon, alt] per level
            - FIELD:   (H, W, 3)    — [time (scalar broadcast), lat, lon] per pixel
        source_name: Human-readable source identifier, e.g. "pmw_amsr2" or "era5_surface".
        channels: Names of each channel in the last axis of ``values``, in order.
            Length must equal ``values.shape[-1]``.
        mask: Boolean mask of valid (non-missing) entries; True = valid.
            Same leading shape as values. If None, all entries are assumed valid.
    """

    kind: SourceKind
    values: Tensor
    coords: Tensor
    source_name: str
    channels: list[str]
    mask: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate shape consistency between values, coords, and mask."""
        self._validate()

    def _validate(self) -> None:
        """Check shape consistency between values, coords, and mask."""
        v, c = self.values, self.coords

        if self.kind is SourceKind.SCALAR:
            if v.ndim != 1:
                raise ValueError(f"SCALAR values must be 1-D, got shape {v.shape}")
            if c.shape != (3,):
                raise ValueError(f"SCALAR coords must be (3,), got {c.shape}")

        elif self.kind is SourceKind.PROFILE:
            if v.ndim != 2:
                raise ValueError(f"PROFILE values must be 2-D (L, C), got {v.shape}")
            if c.ndim != 2 or c.shape[0] != v.shape[0] or c.shape[1] != 4:
                raise ValueError(
                    f"PROFILE coords must be (L, 4), got {c.shape} for values {v.shape}"
                )

        elif self.kind is SourceKind.FIELD:
            if v.ndim != 3:
                raise ValueError(f"FIELD values must be 3-D (H, W, C), got {v.shape}")
            if c.shape != (*v.shape[:2], 3):
                raise ValueError(
                    f"FIELD coords must be (H, W, 3), got {c.shape} for values {v.shape}"
                )

        if self.mask is not None and self.mask.shape != v.shape[: self.mask.ndim]:
            raise ValueError(
                f"mask leading shape {self.mask.shape} incompatible with values {v.shape}"
            )

    @property
    def n_tokens(self) -> int:
        """Number of (value, coord) pairs in this source (flattened spatial dims)."""
        if self.kind is SourceKind.SCALAR:
            return 1
        elif self.kind is SourceKind.PROFILE:
            return self.values.shape[0]  # L
        else:  # FIELD
            h, w = self.values.shape[:2]
            return h * w

    def to(self, device: torch.device | str) -> Source:
        """Move tensors to device, returning a new Source."""
        return Source(
            kind=self.kind,
            values=self.values.to(device),
            coords=self.coords.to(device),
            source_name=self.source_name,
            channels=self.channels,
            mask=self.mask.to(device) if self.mask is not None else None,
        )


@dataclasses.dataclass
class SourceMetadata:
    """Source-level metadata: physical description + full snapshot index.

    This object holds everything that is known about a source across all its
    snapshots.  It does NOT contain any measurement tensors — those live in
    individual :class:`Source` objects loaded on demand.

    Args:
        name: Source directory name, e.g. ``"pmw_amsr2_gcomw1"``.
        type: Physical category, e.g. ``"microwave"``, ``"radar"``.
        kind: Dimensionality class (SCALAR, PROFILE, or FIELD).
        channels: Ordered list of channel names matching the last axis of
            the ``values`` array in each snapshot.
        index: Snapshot index loaded from ``index.parquet``.  Each row
            corresponds to one HDF5 snapshot file; columns include at least
            ``storm_id``, ``snapshot_time_utc``, ``lat``, ``lon``,
            ``source_name``, and ``file_path``.
    """

    name: str
    type: str
    kind: SourceKind
    channels: list[str]
    index: pd.DataFrame = dataclasses.field(compare=False)

    @property
    def num_channels(self) -> int:
        """Number of channels (last axis of ``values`` in each snapshot)."""
        return len(self.channels)


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
