"""Pre-norm cross-sequence block: cross-attention, self-attention, then MLP."""

from __future__ import annotations

from torch import Tensor, nn

from tcfuse.models.cross_transformer.attention import CrossAttention, SelfAttention
from tcfuse.models.cross_transformer.feedforward import FeedForward


class CrossSeqBlock(nn.Module):
    """One block of the two-sequence transformer, run on the target stream.

    All three sub-layers are pre-norm with the residual carried on the target
    stream, in the requested order::

        target = target + cross_attn(norm_cross_q(target), norm_cross_kv(input_kv))
        target = target + self_attn(norm_self(target))
        target = target + ffn(norm_mlp(target))

    The cross-attention reads the (fixed) input sequence into the target; the
    self-attention then mixes information within the target; the MLP refines each
    target token. ``input_kv`` is the encoded input sequence, passed unchanged
    into every block — only the target stream is updated.

    Args:
        embed_dim: Shared token embedding dimension D (target and input).
        num_heads: Number of attention heads; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability used in attention and feed-forward.
    """

    def __init__(
        self, *, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        # Cross-attention sub-layer: target (query) reads the input sequence (kv).
        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            query_dim=embed_dim, kv_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        # Self-attention sub-layer: mix information within the target sequence.
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # Position-wise feed-forward sub-layer.
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        target: Tensor,
        input_kv: Tensor,
        input_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """Run cross-attention, self-attention, and the MLP on the target stream.

        Args:
            target: Target sequence (queries), shape (B, L_tgt, D).
            input_kv: Encoded input sequence (keys/values), shape (B, L_in, D).
            input_mask: ``(B, L_in)`` bool mask, ``True`` at real input tokens.
            target_mask: ``(B, L_tgt)`` bool mask, ``True`` at real target tokens.

        Returns:
            Updated target sequence, shape (B, L_tgt, D).
        """
        # Pre-norm cross-attention; padded input tokens are masked out as keys.
        target = target + self.cross_attn(
            self.norm_cross_q(target),
            self.norm_cross_kv(input_kv),
            key_padding_mask=input_mask,
        )
        # Pre-norm self-attention; padded target tokens are masked out as keys.
        target = target + self.self_attn(self.norm_self(target), key_padding_mask=target_mask)
        # Pre-norm feed-forward sub-layer, residual added afterward.
        target = target + self.ffn(self.norm_mlp(target))
        return target
