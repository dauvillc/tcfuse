"""Perceiver IO backbone.

Encodes every source into tokens, flattens them into one multi-source sequence
``X``, then routes information through a small learned latent array ``Z``: an
encode cross-attention writes ``X`` into ``Z``, a stack of self-attention blocks
processes ``Z``, and a decode cross-attention reads ``Z`` back out using ``X`` as
queries. The decode queries for masked sources carry only their coordinate
information (their values were zeroed before embedding), so reconstructing them
from ``Z`` leaks no value information.
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
from tcfuse.models.perceiver.block import CrossAttentionBlock, LatentBlock


class PerceiverIOBackbone(nn.Module):
    """Encode, route through a latent array, and decode a WindowBatch.

    Owns its own :class:`MultiSourceEncoder` / :class:`MultiSourceDecoder` (built
    from ``sources_metadata``, ``embed_dim``, ``patch_size``), so it can be dropped
    in wherever a plain ``WindowBatch -> WindowBatch`` backbone is expected (e.g.
    :class:`~tcfuse.lightning.base_module.BaseLightningModule`), while internally
    doing most of the work on a small latent sequence of length ``num_latents``.

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
        embed_dim: Shared token embedding dimension D for sources.
        patch_size: Patch size used by the PROFILE / FIELD encoder and decoder.
        latent_dim: Latent embedding dimension Dz.
        num_latents: Number of latent tokens M.
        num_layers: Number of latent self-attention blocks.
        num_heads: Number of attention heads in the latent blocks; must divide
            ``latent_dim``.
        cross_num_heads: Number of attention heads in both cross-attentions; must
            divide both ``embed_dim`` and ``latent_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of the block dim.
        dropout: Dropout probability used inside every block.
    """

    def __init__(
        self,
        sources_metadata: MultisourceMetadata,
        *,
        embed_dim: int,
        patch_size: int,
        latent_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        cross_num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(
            sources_metadata, embed_dim=embed_dim, patch_size=patch_size
        )
        self.decoder = MultiSourceDecoder(
            sources_metadata, embed_dim=embed_dim, patch_size=patch_size
        )
        # Learned latent array Z of shape (M, Dz), expanded per batch at runtime.
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # Encode cross-attention: latents (queries) read from the source tokens.
        self.encode_cross = CrossAttentionBlock(
            query_dim=latent_dim,
            kv_dim=embed_dim,
            num_heads=cross_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        # Latent self-attention stack operating purely on the (B, M, Dz) array.
        self.latent_blocks = nn.ModuleList(
            [
                LatentBlock(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Decode cross-attention: source tokens (queries) read from the latents.
        self.decode_cross = CrossAttentionBlock(
            query_dim=embed_dim,
            kv_dim=latent_dim,
            num_heads=cross_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, batch: WindowBatch) -> WindowBatch:
        """Run the full encode -> route -> decode pipeline on a WindowBatch.

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
        # Expand the learned latent array (M, Dz) -> (B, M, Dz) over the batch.
        batch_size = sequence.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        # Encode cross-attention: write source tokens into the latents.
        latents = self.encode_cross(latents, sequence)
        # Process the latents with the self-attention stack.
        for block in self.latent_blocks:
            latents = block(latents)
        # Decode cross-attention: read the latents back out using source tokens as queries.
        sequence = self.decode_cross(sequence, latents)
        # Split the updated sequence back into per-source token tensors.
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
            The concatenated ``(B, L_total, D)`` sequence, with sources laid out
            in the iteration order of ``embedded.sources`` (the same dict, so
            :meth:`_unflatten_sequence` can split it back apart by re-iterating
            that same order).
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
            sequence: Backbone output, shape (B, L_total, D).
            embedded: Original EmbeddedBatch — its iteration order matches the
                concatenation order produced by :meth:`_flatten_sequence`, and it
                provides each source's kind, source_name, token count, and
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
