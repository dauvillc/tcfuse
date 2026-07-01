"""Two-sequence (input -> target) cross-attention transformer backbone.

Encodes every source into tokens, flattens them into one multi-source sequence,
then splits each sample into an **input sequence** (its non-target sources) and a
**target sequence** (its single target source). A stack of blocks updates the
target stream by cross-attending it onto the (fixed) input stream, self-attending
within the target, and applying an MLP. Unlike the Perceiver, the target keeps its
full token resolution and reads from the inputs directly — there is no latent
bottleneck that compresses the sources.

Because the target source differs per sample within a batch (and each sample has a
single target), the per-sample target/input split yields ragged sequence lengths;
both streams are gathered into padded ``(B, L_max, D)`` tensors and the padding is
masked out of attention. The processed target tokens are scattered back to their
original positions before decoding. Like the masked target tokens in the Perceiver,
the target queries carry only coordinate information (their values were zeroed
before embedding), so reconstructing them from the inputs leaks no value
information.
"""

from __future__ import annotations

import dataclasses

import torch
from einops import rearrange
from torch import Tensor, nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.cross_transformer.block import CrossSeqBlock
from tcfuse.models.decoders.multisource import MultiSourceDecoder
from tcfuse.models.encoders.embedded import EmbeddedBatch, EmbeddedSource
from tcfuse.models.encoders.multisource import MultiSourceEncoder
from tcfuse.models.encoders.positional import CoordEncodingConfig


class CrossSequenceTransformerBackbone(nn.Module):
    """Encode, route inputs into the target stream, and decode a WindowBatch.

    Owns its own :class:`MultiSourceEncoder` / :class:`MultiSourceDecoder` (built
    from ``sources_metadata``, ``embed_dim``, ``patch_size``), so it can be dropped
    in wherever a plain ``WindowBatch -> WindowBatch`` backbone is expected (e.g.
    :class:`~tcfuse.lightning.base_module.BaseLightningModule`).

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
        embed_dim: Shared token embedding dimension D for sources.
        patch_size: Patch size used by the PROFILE / FIELD encoder and decoder.
        num_layers: Number of stacked cross-sequence blocks.
        num_heads: Number of attention heads per block; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability used inside every block.
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
        # Stack of (cross-attention, self-attention, MLP) blocks over the target stream.
        self.blocks = nn.ModuleList(
            [
                CrossSeqBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Final pre-decoder norm: a pre-LN stack leaves the residual stream
        # unnormalized, so normalize the target tokens once before decoding.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, batch: WindowBatch) -> WindowBatch:
        """Run the full encode -> route -> decode pipeline on a WindowBatch.

        Args:
            batch: Collated window batch (values expected in normalized space).

        Returns:
            A new :class:`WindowBatch` with transformed target source values;
            masks, coords, and all other fields are unchanged.
        """
        # Patch-embed every source into per-source token tensors.
        embedded = self.encoder(batch)
        # Flatten and concatenate every source's tokens into one (B, L_total, D) sequence.
        sequence = self._flatten_sequence(embedded)
        # Per-token target flag (B, L_total): True where the token belongs to a
        # source that this sample treats as its target.
        token_is_target = self._token_is_target(embedded, sequence.shape[1])
        # Split into padded target (queries) and input (keys/values) streams.
        target_seq, target_mask, idx_tgt, input_seq, input_mask = self._split_streams(
            sequence, token_is_target
        )
        # Update the target stream by reading from the fixed input stream.
        for block in self.blocks:
            target_seq = block(target_seq, input_seq, input_mask, target_mask)
        # Normalize the pre-LN target residual stream before scattering it back.
        target_seq = self.norm(target_seq)
        # Write the processed target tokens back to their original positions.
        sequence = self._scatter_target(sequence, target_seq, target_mask, idx_tgt)
        # Split the updated sequence back into per-source token tensors.
        new_embedded = self._unflatten_sequence(sequence, embedded)
        # Un-embed tokens back into raw per-source values.
        decoded = self.decoder(new_embedded)
        # Rebuild the output WindowBatch, replacing only each source's values.
        new_sources = dict(batch.sources)
        for key, decoded_source in decoded.sources.items():
            new_sources[key] = dataclasses.replace(batch.sources[key], values=decoded_source.values)
        return dataclasses.replace(batch, sources=new_sources)

    def _token_is_target(self, embedded: EmbeddedBatch, total_tokens: int) -> Tensor:
        """Build a ``(B, L_total)`` per-token target mask.

        Each source occupies a contiguous span of ``n_tokens`` columns in the
        flattened sequence (in the iteration order of ``embedded.sources``, matching
        :meth:`_flatten_sequence`). Within a span, every token inherits that
        source's ``(B,)`` target flag, so a token is a target exactly when its
        sample marked the owning source as a target.

        Args:
            embedded: Per-source token tensors and their per-sample target flags.
            total_tokens: L_total, the concatenated sequence length.

        Returns:
            A ``(B, L_total)`` boolean tensor, ``True`` at target tokens.
        """
        ref = next(iter(embedded.sources.values())).features
        flags = torch.zeros(ref.shape[0], total_tokens, dtype=torch.bool, device=ref.device)
        offset = 0
        for key, source in embedded.sources.items():
            n = source.n_tokens
            # is_target[key] is (B,); broadcast it across this source's token span.
            flags[:, offset : offset + n] = embedded.is_target[key][:, None]
            offset += n
        return flags

    @staticmethod
    def _split_streams(
        sequence: Tensor, token_is_target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Split the sequence into padded target (query) and input (kv) streams.

        Per-sample token counts differ (the target source — hence the number of
        target/input tokens — varies between samples), so each sample's ragged
        target/input token sets are gathered front-aligned into padded
        ``(B, L_max, D)`` tensors with validity masks. A single stable descending
        argsort on the 0/1 target flag yields, per row, all target columns first
        (in original order) followed by all input columns (in original order);
        both streams are read off that one permutation.

        Args:
            sequence: Flattened multi-source sequence, shape (B, L_total, D).
            token_is_target: ``(B, L_total)`` bool, ``True`` at target tokens.

        Returns:
            ``(target_seq, target_mask, idx_tgt, input_seq, input_mask)`` where the
            ``*_seq`` are ``(B, L_*_max, D)`` padded streams, the ``*_mask`` are
            ``(B, L_*_max)`` bool (``True`` at real tokens), and ``idx_tgt`` is
            ``(B, L_tgt_max)`` mapping each target column back to its column in
            ``sequence`` (used to scatter results back).
        """
        _B, L_total, D = sequence.shape
        # Per-row target / input token counts (they sum to L_total).
        tgt_counts = token_is_target.sum(dim=1)
        in_counts = L_total - tgt_counts
        # Documented invariant: masked-source reconstruction always leaves at
        # least one visible source, so every sample must have >=1 input token.
        # A fully-empty input stream would feed SDPA an all-masked key row and
        # silently produce NaN, so fail loudly instead.
        if bool((in_counts == 0).any()):
            raise ValueError(
                "CrossSequenceTransformerBackbone requires every sample to keep at "
                "least one non-target (visible) source, but found a sample whose "
                "sources are all targets (empty input stream)."
            )
        # Batch-max padded lengths for each stream.
        max_tgt = int(tgt_counts.max())
        max_in = int(in_counts.max())
        # Single stable descending argsort on the 0/1 flag: per row, target
        # columns come first (in original order), then input columns (in order).
        order = token_is_target.int().argsort(dim=1, descending=True, stable=True)
        positions = torch.arange(max(max_tgt, max_in), device=sequence.device)
        # Target stream: the first `tgt_counts[b]` columns of `order` per row.
        idx_tgt = order[:, :max_tgt]
        target_mask = positions[:max_tgt][None, :] < tgt_counts[:, None]
        # Input stream: columns [tgt_counts[b], tgt_counts[b] + max_in) of `order`,
        # front-aligned. Clamp overflow (padding) into range; it is masked out.
        in_cols = (tgt_counts[:, None] + positions[:max_in][None, :]).clamp(max=L_total - 1)
        idx_in = order.gather(1, in_cols)
        input_mask = positions[:max_in][None, :] < in_counts[:, None]
        # Gather both streams from `sequence` using their column indices.
        target_seq = sequence.gather(1, idx_tgt[..., None].expand(-1, -1, D))
        input_seq = sequence.gather(1, idx_in[..., None].expand(-1, -1, D))
        return target_seq, target_mask, idx_tgt, input_seq, input_mask

    @staticmethod
    def _scatter_target(
        sequence: Tensor, target_seq: Tensor, target_mask: Tensor, idx_tgt: Tensor
    ) -> Tensor:
        """Write processed target tokens back to their original sequence positions.

        ``idx_tgt`` is a per-row permutation slice, so scatter indices never collide
        within a row. Padding columns (beyond a sample's target count) point at
        non-target positions; we write the original value there so they are no-ops,
        which also keeps any padding-row garbage out of the output.

        Args:
            sequence: Original flattened sequence, shape (B, L_total, D).
            target_seq: Processed target stream, shape (B, L_tgt_max, D).
            target_mask: ``(B, L_tgt_max)`` bool, ``True`` at real target tokens.
            idx_tgt: ``(B, L_tgt_max)`` original-position index for each target column.

        Returns:
            A new ``(B, L_total, D)`` sequence with target tokens replaced.
        """
        idx_exp = idx_tgt[..., None].expand(-1, -1, sequence.shape[-1])
        # At padding columns, write back the original value (a no-op); elsewhere
        # write the processed target token. Match the source dtype: under AMP
        # autocast the LayerNorm'd target stream can come back in a lower precision
        # than `sequence`, and scatter_ requires self and src to share a dtype.
        write = torch.where(target_mask[..., None], target_seq, sequence.gather(1, idx_exp))
        out = sequence.clone()
        out.scatter_(1, idx_exp, write.to(out.dtype))
        return out

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
