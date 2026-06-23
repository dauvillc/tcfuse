"""Decoded data containers: per-source value tensors after the un-embedding layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tcfuse.data.sources.source import SourceKind


@dataclass
class DecodedSource:
    """A single source after un-embedding a token tensor back into raw values.

    Produced by a :class:`~tcfuse.models.decoders.base.SourceDecoder` from an
    :class:`~tcfuse.models.encoders.embedded.EmbeddedSource`. Mirrors the value
    layout of :class:`~tcfuse.data.sources.torch_source.TorchSource` (not the
    embedded token layout): it carries no coords / mask / time, since the
    decoder reconstructs values only.

    Args:
        kind: Dimensionality class of the originating source.
        values: Decoded measurements, always batched.
            - SCALAR:  (B, C)
            - PROFILE: (B, L, C)
            - FIELD:   (B, H, W, C)
        source_name: Human-readable source identifier, e.g. "pmw_amsr2".
    """

    kind: SourceKind
    values: Tensor
    source_name: str

    def __post_init__(self) -> None:
        """Validate that ``values`` has the rank expected for its kind."""
        self._validate()

    def _validate(self) -> None:
        """Check that ``values.ndim`` matches the TorchSource layout for this kind."""
        v = self.values
        if self.kind is SourceKind.SCALAR:
            # SCALAR decodes to one value vector per sample: (B, C).
            if v.ndim != 2:
                raise ValueError(f"SCALAR values must be 2-D (B, C), got {v.shape}")
        elif self.kind is SourceKind.PROFILE:
            # PROFILE decodes to one level axis: (B, L, C).
            if v.ndim != 3:
                raise ValueError(f"PROFILE values must be 3-D (B, L, C), got {v.shape}")
        elif self.kind is SourceKind.FIELD:
            # FIELD decodes to two spatial axes: (B, H, W, C).
            if v.ndim != 4:
                raise ValueError(f"FIELD values must be 4-D (B, H, W, C), got {v.shape}")

    @property
    def batch_size(self) -> int:
        """Number of samples in this decoded source."""
        return int(self.values.shape[0])

    @property
    def num_channels(self) -> int:
        """Number of channels C (last axis of ``values``)."""
        return int(self.values.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape (excluding batch and channel dims).

        Returns:
            - SCALAR:  ``()``
            - PROFILE: ``(L,)``
            - FIELD:   ``(H, W)``
        """
        if self.kind is SourceKind.SCALAR:
            return ()
        elif self.kind is SourceKind.PROFILE:
            # values: (B, L, C) → spatial shape is (L,)
            return (self.values.shape[1],)
        else:  # FIELD
            # values: (B, H, W, C) → spatial shape is (H, W)
            return (self.values.shape[1], self.values.shape[2])

    @property
    def n_tokens(self) -> int:
        """Number of value positions per sample (flattened spatial dims)."""
        # math.prod(()) == 1, which is correct for SCALAR (a single position).
        return max(1, math.prod(self.shape))

    def to(self, device: torch.device | str) -> DecodedSource:
        """Move ``values`` to ``device``, returning a new DecodedSource."""
        return DecodedSource(
            kind=self.kind,
            values=self.values.to(device),
            source_name=self.source_name,
        )


@dataclass
class DecodedBatch:
    """Batched collection of decoded sources, the un-embedding counterpart of EmbeddedBatch.

    Args:
        sources: Dict from ``(source_name, source_index)`` to a DecodedSource.
            Keys mirror those of the originating
            :class:`~tcfuse.models.encoders.embedded.EmbeddedBatch`.
        is_target: Dict from ``(source_name, source_index)`` to a ``(B,)`` bool
            tensor flagging which samples have that slot as a target. Passed
            through unchanged from the originating EmbeddedBatch.
    """

    sources: dict[tuple[str, int], DecodedSource]
    is_target: dict[tuple[str, int], torch.Tensor]

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch (read off any decoded source)."""
        # Every source shares the same leading batch dim; read it off the first.
        return next(iter(self.sources.values())).batch_size
