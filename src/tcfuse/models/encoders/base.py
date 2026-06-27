"""Abstract interface for per-source embedding encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.embedded import EmbeddedSource
from tcfuse.models.encoders.positional import CoordEncodingConfig, SpatioTemporalEncoding


class SourceEncoder(nn.Module, ABC):
    """Embed one source's raw tensor into a token tensor.

    Concrete subclasses (one per :class:`~tcfuse.data.sources.source.SourceKind`)
    patch-embed a :class:`~tcfuse.data.sources.torch_source.TorchSource` into an
    :class:`~tcfuse.models.encoders.embedded.EmbeddedSource`. The common interface
    lets :class:`~tcfuse.models.encoders.multisource.MultiSourceEncoder` treat every
    source uniformly.

    Subclasses also fold a spatio-temporal Fourier positional encoding (see
    :class:`~tcfuse.models.encoders.positional.SpatioTemporalEncoding`) additively
    into the token features, unless ``coord_encoding.enabled`` is ``False``.

    Args:
        source_name: Source identifier this encoder is bound to, e.g. "pmw_amsr2".
        num_channels: Number of input channels C (last axis of ``values``).
        embed_dim: Output embedding dimension D.
        patch_size: Patch size p along each spatial axis. Used by PROFILE / FIELD
            encoders; SCALAR encoders ignore it (a scalar has no spatial axis).
            Part of the common signature so the dispatcher can construct any
            encoder uniformly.
        coord_encoding: Fourier positional-encoding hyperparameters shared by all
            sources. ``enabled=False`` disables coordinate encoding for this run.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        embed_dim: int,
        patch_size: int,
        coord_encoding: CoordEncodingConfig,
    ) -> None:
        super().__init__()
        self.source_name = source_name
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.coord_encoding_config = coord_encoding

    def _make_coord_encoding(self, spatial_axis_groups: list[str]) -> SpatioTemporalEncoding | None:
        """Build this source's positional encoder, or ``None`` when disabled.

        Args:
            spatial_axis_groups: Group id per spatial coordinate axis (see
                :class:`SpatioTemporalEncoding`); the temporal axes are appended
                internally.
        """
        # Respect the master switch: no module allocated when coordinate encoding is off.
        if not self.coord_encoding_config.enabled:
            return None
        return SpatioTemporalEncoding(
            embed_dim=self.embed_dim,
            spatial_axis_groups=spatial_axis_groups,
            config=self.coord_encoding_config,
        )

    @abstractmethod
    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Embed a single batched source into an :class:`EmbeddedSource`."""
