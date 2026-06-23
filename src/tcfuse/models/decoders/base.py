"""Abstract interface for per-source un-embedding decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from tcfuse.models.decoders.decoded import DecodedSource
from tcfuse.models.encoders.embedded import EmbeddedSource


class SourceDecoder(nn.Module, ABC):
    """Un-embed one source's token tensor back into a raw value tensor.

    Concrete subclasses (one per :class:`~tcfuse.data.sources.source.SourceKind`)
    invert the corresponding :class:`~tcfuse.models.encoders.base.SourceEncoder`,
    mapping an :class:`~tcfuse.models.encoders.embedded.EmbeddedSource` back to a
    :class:`~tcfuse.models.decoders.decoded.DecodedSource` whose ``values`` match
    the originating :class:`~tcfuse.data.sources.torch_source.TorchSource` layout.
    The common interface lets
    :class:`~tcfuse.models.decoders.multisource.MultiSourceDecoder` treat every
    source uniformly.

    Args:
        source_name: Source identifier this decoder is bound to, e.g. "pmw_amsr2".
        num_channels: Number of output channels C (last axis of the decoded values).
        embed_dim: Input embedding dimension D.
        patch_size: Patch size p along each spatial axis. Used by PROFILE / FIELD
            decoders; SCALAR decoders ignore it (a scalar has no spatial axis).
            Part of the common signature so the dispatcher can construct any
            decoder uniformly.
    """

    def __init__(
        self, *, source_name: str, num_channels: int, embed_dim: int, patch_size: int
    ) -> None:
        super().__init__()
        self.source_name = source_name
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    @abstractmethod
    def forward(self, embedded: EmbeddedSource) -> DecodedSource:
        """Un-embed a single batched EmbeddedSource into a :class:`DecodedSource`."""
