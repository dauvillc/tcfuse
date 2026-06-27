"""Batch-level dispatcher mapping a WindowBatch to an EmbeddedBatch."""

from __future__ import annotations

from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.encoders.base import SourceEncoder
from tcfuse.models.encoders.embedded import EmbeddedBatch, EmbeddedSource
from tcfuse.models.encoders.patch_embed import FieldEncoder, ProfileEncoder, ScalarEncoder
from tcfuse.models.encoders.positional import CoordEncodingConfig

# Map each source kind to the encoder class that embeds it.
_KIND_TO_ENCODER: dict[SourceKind, type[SourceEncoder]] = {
    SourceKind.SCALAR: ScalarEncoder,
    SourceKind.PROFILE: ProfileEncoder,
    SourceKind.FIELD: FieldEncoder,
}


class MultiSourceEncoder(nn.Module):
    """Embed every source in a batch with a per-source-name encoder.

    Allocates one :class:`~tcfuse.models.encoders.base.SourceEncoder` per source
    name at construction time, choosing the encoder class from the source's
    :class:`~tcfuse.data.sources.source.SourceKind` and reading its channel count
    from ``sources_metadata``. Meant to be a Hydra partial (``_partial_: true``)
    so ``BaseLightningModule`` can pass ``sources_metadata`` at runtime.

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
            Provides each source's kind and channel count.
        embed_dim: Output embedding dimension D, shared across all sources.
        patch_size: Patch size p used by PROFILE / FIELD encoders.
        coord_encoding: Fourier positional-encoding hyperparameters shared by all
            per-source encoders. Defaults to :class:`CoordEncodingConfig` defaults
            (encoding enabled) when ``None``.
    """

    def __init__(
        self,
        sources_metadata: MultisourceMetadata,
        *,
        embed_dim: int,
        patch_size: int,
        coord_encoding: CoordEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        # Fall back to default coordinate-encoding hyperparameters when unset.
        coord_encoding = coord_encoding if coord_encoding is not None else CoordEncodingConfig()
        encoders: dict[str, SourceEncoder] = {}
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
                embed_dim=embed_dim,
                patch_size=patch_size,
                coord_encoding=coord_encoding,
            )
            self._key_map[name] = name.replace(".", "_").replace("-", "_")
        self._encoders = nn.ModuleDict(encoders)

    def forward(self, batch: WindowBatch) -> EmbeddedBatch:
        """Embed each source in the batch, returning an EmbeddedBatch.

        Args:
            batch: Collated, normalized window batch (NaN-fill already zeroed).

        Returns:
            An :class:`EmbeddedBatch` with one EmbeddedSource per input source and
            ``is_target`` carried through unchanged.
        """
        # Embed every (source_name, index) slot with its source-name encoder.
        embedded_sources: dict[tuple[str, int], EmbeddedSource] = {}
        for key, source in batch.sources.items():
            source_name, _idx = key
            encoder = self._encoders[self._key_map[source_name]]
            embedded_sources[key] = encoder(source)
        # Target flags are independent of the embedding; pass them straight through.
        return EmbeddedBatch(sources=embedded_sources, is_target=dict(batch.is_target))
