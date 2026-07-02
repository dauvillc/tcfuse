"""Batch-level dispatcher mapping a WindowBatch to a MotifEmbeddedBatch."""

from __future__ import annotations

from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.encoders_motif.base import MotifSourceEncoder
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedBatch, MotifEmbeddedSource
from tcfuse.models.encoders_motif.patch_embed import (
    MotifFieldEncoder,
    MotifProfileEncoder,
    MotifScalarEncoder,
)
from tcfuse.models.encoders_motif.positional import MotifCoordEncodingConfig

# Map each source kind to the encoder class that embeds it.
_KIND_TO_ENCODER: dict[SourceKind, type[MotifSourceEncoder]] = {
    SourceKind.SCALAR: MotifScalarEncoder,
    SourceKind.PROFILE: MotifProfileEncoder,
    SourceKind.FIELD: MotifFieldEncoder,
}


class MotifMultiSourceEncoder(nn.Module):
    """Embed every source in a batch with a per-source-name MoTiF encoder.

    Allocates one :class:`~tcfuse.models.encoders_motif.base.MotifSourceEncoder`
    per source name at construction time, choosing the encoder class from the
    source's :class:`~tcfuse.data.sources.source.SourceKind` and reading its
    channel count from ``sources_metadata``. Meant to be owned by the future
    MoTiF backbone, which threads ``sources_metadata`` through at runtime (same
    Hydra-partial pattern as the existing backbones).

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
            Provides each source's kind and channel count.
        value_dim: Output value-embedding dimension Dv, shared across all sources.
        coord_dim: Output coordinate-embedding dimension Dc, shared across all
            sources (may differ from ``value_dim``).
        patch_size: Patch size p used by PROFILE / FIELD encoders.
        coord_encoding: Fourier coordinate-embedding hyperparameters shared by
            all per-source encoders. Defaults to :class:`MotifCoordEncodingConfig`
            defaults when ``None``.
    """

    def __init__(
        self,
        sources_metadata: MultisourceMetadata,
        *,
        value_dim: int,
        coord_dim: int,
        patch_size: int,
        coord_encoding: MotifCoordEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        # Fall back to default coordinate-embedding hyperparameters when unset.
        coord_encoding = (
            coord_encoding if coord_encoding is not None else MotifCoordEncodingConfig()
        )
        encoders: dict[str, MotifSourceEncoder] = {}
        # Map original source name -> sanitized ModuleDict key.
        # ModuleDict does not allow '.' or '-' in keys.
        self._key_map: dict[str, str] = {}
        for name in sources_metadata.names:
            meta = sources_metadata[name]
            # Pick the encoder class for this source's dimensionality class.
            encoder_cls = _KIND_TO_ENCODER[meta.kind]
            # Allocate a per-source encoder from its channel count.
            encoders[name.replace(".", "_").replace("-", "_")] = encoder_cls(
                source_name=name,
                num_channels=meta.num_channels,
                value_dim=value_dim,
                coord_dim=coord_dim,
                patch_size=patch_size,
                coord_encoding=coord_encoding,
            )
            self._key_map[name] = name.replace(".", "_").replace("-", "_")
        self._encoders = nn.ModuleDict(encoders)

    def forward(self, batch: WindowBatch) -> MotifEmbeddedBatch:
        """Embed each source in the batch, returning a MotifEmbeddedBatch.

        Args:
            batch: Collated, normalized window batch (NaN-fill zeroed inside each encoder).

        Returns:
            A :class:`MotifEmbeddedBatch` with one MotifEmbeddedSource per input
            source and ``is_target`` carried through unchanged.
        """
        # Embed every (source_name, index) slot with its source-name encoder.
        embedded_sources: dict[tuple[str, int], MotifEmbeddedSource] = {}
        for key, source in batch.sources.items():
            source_name, _idx = key
            encoder = self._encoders[self._key_map[source_name]]
            embedded_sources[key] = encoder(source)
        # Target flags are independent of the embedding; pass them straight through.
        return MotifEmbeddedBatch(sources=embedded_sources, is_target=dict(batch.is_target))
