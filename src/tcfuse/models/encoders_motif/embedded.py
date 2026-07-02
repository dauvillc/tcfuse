"""Embedded data containers for MoTiF: dual value/coordinate token tensors per source."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tcfuse.data.sources.source import SourceKind


@dataclass
class MotifEmbeddedSource:
    """A single source after embedding into separate value and coordinate tensors.

    Produced by a :class:`~tcfuse.models.encoders_motif.base.MotifSourceEncoder`
    from a :class:`~tcfuse.data.sources.torch_source.TorchSource`. Unlike the
    single-tensor :class:`~tcfuse.models.encoders.embedded.EmbeddedSource`, the
    coordinate embedding is kept **separate** from the value embedding so the
    MoTiF backbone can inject it as positional conditioning at every layer.
    Both tensors share all dims except the last (embedding) one.

    Args:
        kind: Dimensionality class of the originating source.
        values: Embedded value tokens, always batched.
            - SCALAR:  (B, Dv)
            - PROFILE: (B, El, Dv)       — El = ceil(L / patch_size)
            - FIELD:   (B, Eh, Ew, Dv)   — Eh = ceil(H / patch_size), Ew = ceil(W / patch_size)
        coords: Embedded coordinate tokens, same layout as ``values`` but with
            last dim Dc (possibly != Dv).
            - SCALAR:  (B, Dc)
            - PROFILE: (B, El, Dc)
            - FIELD:   (B, Eh, Ew, Dc)
        source_name: Human-readable source identifier, e.g. "pmw_amsr2".
        input_shape: Original spatial shape of the source before any patch-size padding.
            Stored by the encoder so a paired decoder can crop the output back to
            the exact original spatial dimensions.
            - SCALAR:  ``()``
            - PROFILE: ``(L,)``
            - FIELD:   ``(H, W)``
            Defaults to ``()`` (no cropping applied downstream).
    """

    kind: SourceKind
    values: Tensor
    coords: Tensor
    source_name: str
    input_shape: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate tensor ranks and the shared-spatial-dims contract."""
        self._validate()

    def _validate(self) -> None:
        """Check ranks per kind and that values/coords share all non-embedding dims."""
        # Both tensors must have the rank expected for this kind's embedded layout.
        for name, t in (("values", self.values), ("coords", self.coords)):
            if self.kind is SourceKind.SCALAR:
                # SCALAR embeds to a single token per sample: (B, D).
                if t.ndim != 2:
                    raise ValueError(f"SCALAR {name} must be 2-D (B, D), got {t.shape}")
            elif self.kind is SourceKind.PROFILE:
                # PROFILE keeps one embedded axis: (B, El, D).
                if t.ndim != 3:
                    raise ValueError(f"PROFILE {name} must be 3-D (B, El, D), got {t.shape}")
            elif self.kind is SourceKind.FIELD:
                # FIELD keeps two embedded spatial axes: (B, Eh, Ew, D).
                if t.ndim != 4:
                    raise ValueError(f"FIELD {name} must be 4-D (B, Eh, Ew, D), got {t.shape}")
        # Core MoTiF contract: values and coords share batch + spatial dims (last dim may differ).
        if self.values.shape[:-1] != self.coords.shape[:-1]:
            raise ValueError(
                "values and coords must share all dims except the last, got "
                f"values {self.values.shape} vs coords {self.coords.shape}"
            )

    @property
    def batch_size(self) -> int:
        """Number of samples in this embedded source."""
        return int(self.values.shape[0])

    @property
    def value_dim(self) -> int:
        """Value-embedding dimension Dv (last axis of ``values``)."""
        return int(self.values.shape[-1])

    @property
    def coord_dim(self) -> int:
        """Coordinate-embedding dimension Dc (last axis of ``coords``)."""
        return int(self.coords.shape[-1])

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
            # values: (B, El, Dv) → embedded spatial shape is (El,)
            return (self.values.shape[1],)
        else:  # FIELD
            # values: (B, Eh, Ew, Dv) → embedded spatial shape is (Eh, Ew)
            return (self.values.shape[1], self.values.shape[2])

    @property
    def n_tokens(self) -> int:
        """Number of tokens per sample (flattened embedded spatial dims)."""
        # math.prod(()) == 1, which is correct for SCALAR (a single token).
        return max(1, math.prod(self.embedded_shape))

    def to(self, device: torch.device | str) -> MotifEmbeddedSource:
        """Move both tensors to ``device``, returning a new MotifEmbeddedSource."""
        return MotifEmbeddedSource(
            kind=self.kind,
            values=self.values.to(device),
            coords=self.coords.to(device),
            source_name=self.source_name,
            input_shape=self.input_shape,
        )


@dataclass
class MotifEmbeddedBatch:
    """Batched collection of MoTiF-embedded sources for the downstream backbone.

    The minimal counterpart of :class:`~tcfuse.data.collate.WindowBatch`: it keeps
    only what the MoTiF backbone needs — the embedded value/coordinate tokens per
    source and the per-source target flags. All window/storm metadata is dropped.

    Args:
        sources: Dict from ``(source_name, source_index)`` to a MotifEmbeddedSource.
            Keys mirror those of the originating :class:`WindowBatch`.
        is_target: Dict from ``(source_name, source_index)`` to a ``(B,)`` bool
            tensor flagging which samples have that slot as a target. Keys match
            those in ``sources``.
    """

    sources: dict[tuple[str, int], MotifEmbeddedSource]
    is_target: dict[tuple[str, int], torch.Tensor]

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch (read off any embedded source)."""
        # Every source shares the same leading batch dim; read it off the first.
        return next(iter(self.sources.values())).batch_size
