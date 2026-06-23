"""Embedded data containers: per-source token tensors after the embedding layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tcfuse.data.sources.source import SourceKind


@dataclass
class EmbeddedSource:
    """A single source after patch-embedding into a token tensor.

    Produced by a :class:`~tcfuse.models.encoders.base.SourceEncoder` from a
    :class:`~tcfuse.data.sources.torch_source.TorchSource`. Unlike ``TorchSource``,
    this carries no coords / mask / time: the embedding layer has either dropped
    that metadata (downstream backbone does not read it) or, in a later iteration,
    folded it into ``features``.

    Args:
        kind: Dimensionality class of the originating source.
        features: Embedded tokens, always batched.
            - SCALAR:  (B, D)
            - PROFILE: (B, El, D)       — El = L // patch_size
            - FIELD:   (B, Eh, Ew, D)   — Eh = H // patch_size, Ew = W // patch_size
        source_name: Human-readable source identifier, e.g. "pmw_amsr2".
    """

    kind: SourceKind
    features: Tensor
    source_name: str

    def __post_init__(self) -> None:
        """Validate that ``features`` has the rank expected for its kind."""
        self._validate()

    def _validate(self) -> None:
        """Check that ``features.ndim`` matches the embedded layout for this kind."""
        f = self.features
        if self.kind is SourceKind.SCALAR:
            # SCALAR embeds to a single token per sample: (B, D).
            if f.ndim != 2:
                raise ValueError(f"SCALAR features must be 2-D (B, D), got {f.shape}")
        elif self.kind is SourceKind.PROFILE:
            # PROFILE keeps one embedded axis: (B, El, D).
            if f.ndim != 3:
                raise ValueError(f"PROFILE features must be 3-D (B, El, D), got {f.shape}")
        elif self.kind is SourceKind.FIELD:
            # FIELD keeps two embedded spatial axes: (B, Eh, Ew, D).
            if f.ndim != 4:
                raise ValueError(f"FIELD features must be 4-D (B, Eh, Ew, D), got {f.shape}")

    @property
    def batch_size(self) -> int:
        """Number of samples in this embedded source."""
        return int(self.features.shape[0])

    @property
    def embed_dim(self) -> int:
        """Embedding dimension D (last axis of ``features``)."""
        return int(self.features.shape[-1])

    @property
    def embedded_shape(self) -> tuple[int, ...]:
        """Embedded spatial shape (excluding batch and embedding dims).

        Returns:
            - SCALAR:  ``()``
            - PROFILE: ``(El,)``
            - FIELD:   ``(Eh, Ew)``
        """
        if self.kind is SourceKind.SCALAR:
            return ()
        elif self.kind is SourceKind.PROFILE:
            # features: (B, El, D) → embedded spatial shape is (El,)
            return (self.features.shape[1],)
        else:  # FIELD
            # features: (B, Eh, Ew, D) → embedded spatial shape is (Eh, Ew)
            return (self.features.shape[1], self.features.shape[2])

    @property
    def n_tokens(self) -> int:
        """Number of tokens per sample (flattened embedded spatial dims)."""
        # math.prod(()) == 1, which is correct for SCALAR (a single token).
        return max(1, math.prod(self.embedded_shape))

    def to(self, device: torch.device | str) -> EmbeddedSource:
        """Move ``features`` to ``device``, returning a new EmbeddedSource."""
        return EmbeddedSource(
            kind=self.kind,
            features=self.features.to(device),
            source_name=self.source_name,
        )


@dataclass
class EmbeddedBatch:
    """Batched collection of embedded sources for the downstream backbone.

    The minimal counterpart of :class:`~tcfuse.data.collate.WindowBatch`: it keeps
    only what a transformer / perceiver backbone needs — the embedded tokens per
    source and the per-source target flags. All window/storm metadata is dropped.

    Args:
        sources: Dict from ``(source_name, source_index)`` to an EmbeddedSource.
            Keys mirror those of the originating :class:`WindowBatch`.
        is_target: Dict from ``(source_name, source_index)`` to a ``(B,)`` bool
            tensor flagging which samples have that slot as a target. Keys match
            those in ``sources``.
    """

    sources: dict[tuple[str, int], EmbeddedSource]
    is_target: dict[tuple[str, int], torch.Tensor]

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch (read off any embedded source)."""
        # Every source shares the same leading batch dim; read it off the first.
        return next(iter(self.sources.values())).batch_size
