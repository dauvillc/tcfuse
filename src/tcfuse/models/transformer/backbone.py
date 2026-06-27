"""Plain single-sequence transformer backbone.

Concatenates every source's embedded tokens into one multi-source sequence,
runs a stack of standard transformer blocks over it, then splits the result
back into per-source tokens for decoding.
"""

from __future__ import annotations

import dataclasses

import torch
from einops import rearrange
from torch import Tensor, nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.decoders.multisource import MultiSourceDecoder
from tcfuse.models.encoders.embedded import EmbeddedBatch, EmbeddedSource
from tcfuse.models.encoders.multisource import MultiSourceEncoder
from tcfuse.models.encoders.positional import CoordEncodingConfig
from tcfuse.models.transformer.block import TransformerBlock


class SingleSequenceTransformerBackbone(nn.Module):
    """Encode, transform, and decode a WindowBatch through one shared sequence.

    Owns its own :class:`MultiSourceEncoder` / :class:`MultiSourceDecoder`
    (built from ``sources_metadata``, ``embed_dim``, ``patch_size``), so it can
    be dropped in wherever a plain ``WindowBatch -> WindowBatch`` backbone is
    expected (e.g. :class:`~tcfuse.lightning.base_module.BaseLightningModule`),
    while internally doing attention over a token-level representation.

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
        embed_dim: Shared token embedding dimension D.
        patch_size: Patch size used by the PROFILE / FIELD encoder and decoder.
        num_layers: Number of stacked transformer blocks.
        num_heads: Number of attention heads per block; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability used inside every transformer block.
        coord_encoding: Fourier positional-encoding hyperparameters forwarded to
            the :class:`MultiSourceEncoder` (defaults applied when ``None``).
    """

    def __init__(
        self,
        sources_metadata: MultisourceMetadata,
        *,
        embed_dim: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        coord_encoding: CoordEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(
            sources_metadata,
            embed_dim=embed_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        self.decoder = MultiSourceDecoder(
            sources_metadata, embed_dim=embed_dim, patch_size=patch_size
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Final pre-decoder norm: a pre-LN stack leaves the residual stream
        # unnormalized, so normalize once before the decoder reads it.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, batch: WindowBatch) -> WindowBatch:
        """Run the full encode -> transform -> decode pipeline on a WindowBatch.

        Args:
            batch: Collated window batch (values expected in normalized space).

        Returns:
            A new :class:`WindowBatch` with transformed source values; masks,
            coords, and all other fields are unchanged.
        """
        # Patch-embed every source into per-source token tensors.
        embedded = self.encoder(batch)
        # Flatten and concatenate every source's tokens into one (B, L, D) sequence.
        sequence = self._flatten_sequence(embedded)
        # Run the shared transformer stack over the whole multi-source sequence.
        for block in self.blocks:
            sequence = block(sequence)
        # Normalize the pre-LN residual stream before decoding.
        sequence = self.norm(sequence)
        # Split the processed sequence back into per-source token tensors.
        new_embedded = self._unflatten_sequence(sequence, embedded)
        # Un-embed tokens back into raw per-source values.
        decoded = self.decoder(new_embedded)
        # Rebuild the output WindowBatch, replacing only each source's values.
        new_sources = dict(batch.sources)
        for key, decoded_source in decoded.sources.items():
            new_sources[key] = dataclasses.replace(batch.sources[key], values=decoded_source.values)
        return dataclasses.replace(batch, sources=new_sources)

    def _flatten_sequence(self, embedded: EmbeddedBatch) -> Tensor:
        """Flatten and concatenate every source's tokens into one sequence.

        Args:
            embedded: Per-source token tensors produced by ``self.encoder``.

        Returns:
            The concatenated ``(B, L_total, D)`` sequence, with sources laid
            out in the iteration order of ``embedded.sources`` (the same dict,
            so :meth:`_unflatten_sequence` can split it back apart by
            re-iterating that same order).
        """
        flat_per_source: list[Tensor] = []
        for source in embedded.sources.values():
            if source.kind is SourceKind.SCALAR:
                # (B, D) -> (B, 1, D): a scalar source is a single token.
                seq = source.features.unsqueeze(1)
            elif source.kind is SourceKind.PROFILE:
                # Already (B, El, D): one embedded axis to flatten.
                seq = source.features
            else:  # FIELD
                # (B, Eh, Ew, D) -> (B, Eh * Ew, D): flatten the spatial grid.
                seq = rearrange(source.features, "b h w d -> b (h w) d")
            flat_per_source.append(seq)
        # Concatenate every source's tokens along the sequence axis.
        return torch.cat(flat_per_source, dim=1)

    def _unflatten_sequence(self, sequence: Tensor, embedded: EmbeddedBatch) -> EmbeddedBatch:
        """Split a processed sequence back into per-source token tensors.

        Args:
            sequence: Transformer output, shape (B, L_total, D).
            embedded: Original EmbeddedBatch — its iteration order matches the
                concatenation order produced by :meth:`_flatten_sequence`, and
                it provides each source's kind, source_name, token count, and
                embedded spatial shape needed to reshape back.

        Returns:
            An :class:`EmbeddedBatch` with the same keys as ``embedded`` and
            ``is_target`` carried through unchanged.
        """
        # Split the sequence back into one chunk per source, in concatenation order.
        sizes = [source.n_tokens for source in embedded.sources.values()]
        chunks = torch.split(sequence, sizes, dim=1)
        new_sources: dict[tuple[str, int], EmbeddedSource] = {}
        for (key, source), chunk in zip(embedded.sources.items(), chunks):
            if source.kind is SourceKind.SCALAR:
                # (B, 1, D) -> (B, D): drop the singleton token axis.
                features = chunk.squeeze(1)
            elif source.kind is SourceKind.PROFILE:
                # Already (B, El, D): no reshape needed.
                features = chunk
            else:  # FIELD
                # (B, Eh * Ew, D) -> (B, Eh, Ew, D): restore the spatial grid.
                Eh, Ew = source.embedded_shape
                features = rearrange(chunk, "b (h w) d -> b h w d", h=Eh, w=Ew)
            new_sources[key] = EmbeddedSource(
                kind=source.kind,
                features=features,
                source_name=source.source_name,
                input_shape=source.input_shape,
            )
        return EmbeddedBatch(sources=new_sources, is_target=dict(embedded.is_target))
