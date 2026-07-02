"""Batch-level dispatcher mapping a MotifEmbeddedBatch to a MotifDecodedBatch."""

from __future__ import annotations

from torch import nn

from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.decoders_motif.base import MotifSourceDecoder
from tcfuse.models.decoders_motif.decoded import MotifDecodedBatch, MotifDecodedSource
from tcfuse.models.decoders_motif.patch_unembed import (
    MotifFieldDecoder,
    MotifProfileDecoder,
    MotifScalarDecoder,
)
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedBatch

# Map each source kind to the decoder class that un-embeds it.
_KIND_TO_DECODER: dict[SourceKind, type[MotifSourceDecoder]] = {
    SourceKind.SCALAR: MotifScalarDecoder,
    SourceKind.PROFILE: MotifProfileDecoder,
    SourceKind.FIELD: MotifFieldDecoder,
}


class MotifMultiSourceDecoder(nn.Module):
    """Un-embed every source in a MoTiF batch with a per-source-name decoder.

    Allocates one :class:`~tcfuse.models.decoders_motif.base.MotifSourceDecoder`
    per source name at construction time, choosing the decoder class from the
    source's :class:`~tcfuse.data.sources.source.SourceKind` and reading its
    channel count from ``sources_metadata``. Mirrors
    :class:`~tcfuse.models.encoders_motif.multisource.MotifMultiSourceEncoder`;
    meant to be a Hydra partial (``_partial_: true``) so the MoTiF backbone can
    pass ``sources_metadata`` at runtime.

    ``embed_dim`` and ``patch_size`` must match the
    :class:`~tcfuse.models.encoders_motif.multisource.MotifMultiSourceEncoder`
    (or backbone) that produced the input
    :class:`~tcfuse.models.encoders_motif.embedded.MotifEmbeddedBatch`.

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
            Provides each source's kind and channel count.
        embed_dim: Input value-embedding dimension Dv, shared across all sources.
        patch_size: Patch size p used by PROFILE / FIELD decoders.
    """

    def __init__(
        self,
        sources_metadata: MultisourceMetadata,
        *,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        decoders: dict[str, MotifSourceDecoder] = {}
        # Map original source name -> sanitized ModuleDict key.
        # ModuleDict does not allow '.' or '-' in keys.
        self._key_map: dict[str, str] = {}
        for name in sources_metadata.names:
            meta = sources_metadata[name]
            # Pick the decoder class for this source's dimensionality class.
            decoder_cls = _KIND_TO_DECODER[meta.kind]
            # Allocate a per-source decoder from its channel count.
            decoders[name.replace(".", "_").replace("-", "_")] = decoder_cls(
                source_name=name,
                num_channels=meta.num_channels,
                embed_dim=embed_dim,
                patch_size=patch_size,
            )
            self._key_map[name] = name.replace(".", "_").replace("-", "_")
        self._decoders = nn.ModuleDict(decoders)

    def forward(self, batch: MotifEmbeddedBatch) -> MotifDecodedBatch:
        """Un-embed each source in the batch, returning a MotifDecodedBatch.

        Args:
            batch: MoTiF embedded batch produced by a MotifMultiSourceEncoder (or a
                backbone consuming/producing the same MotifEmbeddedBatch interface).

        Returns:
            A :class:`MotifDecodedBatch` with one MotifDecodedSource per input
            source and ``is_target`` carried through unchanged.
        """
        # Un-embed every (source_name, index) slot with its source-name decoder.
        decoded_sources: dict[tuple[str, int], MotifDecodedSource] = {}
        for key, source in batch.sources.items():
            source_name, _idx = key
            decoder = self._decoders[self._key_map[source_name]]
            decoded_sources[key] = decoder(source)
        # Target flags are independent of decoding; pass them straight through.
        return MotifDecodedBatch(sources=decoded_sources, is_target=dict(batch.is_target))
