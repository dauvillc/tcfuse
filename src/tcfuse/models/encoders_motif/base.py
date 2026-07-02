"""Abstract interface for per-source MoTiF embedding encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedSource
from tcfuse.models.encoders_motif.positional import CoordEmbedding, MotifCoordEncodingConfig


class MotifSourceEncoder(nn.Module, ABC):
    """Embed one source's raw tensor into separate value and coordinate tokens.

    Concrete subclasses (one per :class:`~tcfuse.data.sources.source.SourceKind`)
    embed a :class:`~tcfuse.data.sources.torch_source.TorchSource` into a
    :class:`~tcfuse.models.encoders_motif.embedded.MotifEmbeddedSource`. The common
    interface lets :class:`~tcfuse.models.encoders_motif.multisource.MotifMultiSourceEncoder`
    treat every source uniformly.

    Unlike :class:`~tcfuse.models.encoders.base.SourceEncoder`, the coordinate
    embedding is never added to the value tokens: subclasses produce it as a
    standalone tensor (see :class:`CoordEmbedding`) so the MoTiF backbone can
    inject it as positional conditioning at every layer.

    Args:
        source_name: Source identifier this encoder is bound to, e.g. "pmw_amsr2".
        num_channels: Number of input channels C (last axis of ``values``).
        value_dim: Output value-embedding dimension Dv.
        coord_dim: Output coordinate-embedding dimension Dc (may differ from Dv).
        patch_size: Patch size p along each spatial axis. Used by PROFILE / FIELD
            encoders; SCALAR encoders ignore it (a scalar has no spatial axis).
            Part of the common signature so the dispatcher can construct any
            encoder uniformly.
        coord_encoding: Fourier coordinate-embedding hyperparameters shared by all
            sources.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        value_dim: int,
        coord_dim: int,
        patch_size: int,
        coord_encoding: MotifCoordEncodingConfig,
    ) -> None:
        super().__init__()
        self.source_name = source_name
        self.num_channels = num_channels
        self.value_dim = value_dim
        self.coord_dim = coord_dim
        self.patch_size = patch_size
        self.coord_encoding_config = coord_encoding

    def _make_coord_embedding(self, spatial_axis_groups: list[str]) -> CoordEmbedding:
        """Build this source's standalone coordinate embedder.

        Args:
            spatial_axis_groups: Group id per spatial coordinate axis (see
                :class:`CoordEmbedding`); the temporal axes are appended
                internally.
        """
        # The coord tensor is mandatory in MoTiF, so a module is always allocated.
        return CoordEmbedding(
            coord_dim=self.coord_dim,
            spatial_axis_groups=spatial_axis_groups,
            config=self.coord_encoding_config,
        )

    @abstractmethod
    def forward(self, source: TorchSource) -> MotifEmbeddedSource:
        """Embed a single batched source into a :class:`MotifEmbeddedSource`."""
