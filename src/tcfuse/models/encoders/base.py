"""Abstract interface for per-source embedding encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.embedded import EmbeddedSource


class SourceEncoder(nn.Module, ABC):
    """Embed one source's raw tensor into a token tensor.

    Concrete subclasses (one per :class:`~tcfuse.data.sources.source.SourceKind`)
    patch-embed a :class:`~tcfuse.data.sources.torch_source.TorchSource` into an
    :class:`~tcfuse.models.encoders.embedded.EmbeddedSource`. The common interface
    lets :class:`~tcfuse.models.encoders.multisource.MultiSourceEncoder` treat every
    source uniformly.

    Args:
        source_name: Source identifier this encoder is bound to, e.g. "pmw_amsr2".
        num_channels: Number of input channels C (last axis of ``values``).
        embed_dim: Output embedding dimension D.
        patch_size: Patch size p along each spatial axis. Used by PROFILE / FIELD
            encoders; SCALAR encoders ignore it (a scalar has no spatial axis).
            Part of the common signature so the dispatcher can construct any
            encoder uniformly.
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
    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Embed a single batched source into an :class:`EmbeddedSource`."""
