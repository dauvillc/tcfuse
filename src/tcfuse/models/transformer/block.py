"""Pre-norm transformer block: attention sub-layer + feed-forward sub-layer."""

from __future__ import annotations

from torch import Tensor, nn

from tcfuse.models.transformer.attention import MultiHeadSelfAttention
from tcfuse.models.transformer.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """One pre-norm transformer block with residual connections.

    Each sub-layer normalizes its input first (pre-LN), then adds its output
    back onto the un-normalized residual stream::

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability used in both the attention output
            projection and the feed-forward sub-layer.
    """

    def __init__(
        self, *, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the attention and feed-forward sub-layers, shape (B, L, D) -> (B, L, D)."""
        # Pre-norm attention sub-layer, residual added afterward.
        x = x + self.attn(self.norm1(x))
        # Pre-norm feed-forward sub-layer, residual added afterward.
        x = x + self.ffn(self.norm2(x))
        return x
