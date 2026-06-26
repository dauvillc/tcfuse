"""Pre-norm Perceiver blocks: a latent self-attention block and a cross-attention block."""

from __future__ import annotations

from torch import Tensor, nn

from tcfuse.models.perceiver.attention import CrossAttention, SelfAttention
from tcfuse.models.perceiver.feedforward import FeedForward


class LatentBlock(nn.Module):
    """One pre-norm self-attention block over the latent array.

    Each sub-layer normalizes its input first (pre-LN), then adds its output
    back onto the un-normalized residual stream::

        z = z + attn(norm1(z))
        z = z + ffn(norm2(z))

    Args:
        embed_dim: Latent embedding dimension Dz.
        num_heads: Number of attention heads; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability used in attention and feed-forward.
    """

    def __init__(
        self, *, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, z: Tensor) -> Tensor:
        """Apply self-attention and feed-forward, shape (B, M, Dz) -> (B, M, Dz)."""
        # Pre-norm self-attention sub-layer, residual added afterward.
        z = z + self.attn(self.norm1(z))
        # Pre-norm feed-forward sub-layer, residual added afterward.
        z = z + self.ffn(self.norm2(z))
        return z


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with residual on the query stream.

    Normalizes both the query and key/value inputs (pre-LN), runs cross-attention
    with the query stream carrying the residual, then a pre-norm feed-forward
    sub-layer::

        q = q + attn(norm_q(q), norm_kv(kv))
        q = q + ffn(norm_mlp(q))

    Args:
        query_dim: Dimension of the query tokens (and of the output).
        kv_dim: Dimension of the key/value tokens.
        num_heads: Number of attention heads; must divide ``query_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``query_dim``.
        dropout: Dropout probability used in attention and feed-forward.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = CrossAttention(
            query_dim=query_dim, kv_dim=kv_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm_mlp = nn.LayerNorm(query_dim)
        self.ffn = FeedForward(embed_dim=query_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        """Cross-attend ``query`` over ``kv`` then feed-forward.

        Args:
            query: Query sequence, shape (B, Lq, query_dim).
            kv: Key/value sequence, shape (B, Lkv, kv_dim).

        Returns:
            Updated query sequence, shape (B, Lq, query_dim).
        """
        # Pre-norm cross-attention sub-layer; residual carried on the query stream.
        query = query + self.attn(self.norm_q(query), self.norm_kv(kv))
        # Pre-norm feed-forward sub-layer, residual added afterward.
        query = query + self.ffn(self.norm_mlp(query))
        return query
