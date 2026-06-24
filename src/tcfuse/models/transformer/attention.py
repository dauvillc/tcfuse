"""Multi-head self-attention over a flattened multi-source token sequence."""

from __future__ import annotations

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """Full self-attention over a ``(B, L, D)`` sequence, computed with SDPA.

    No causal mask, no key-padding mask: every token in the flattened
    multi-source sequence attends densely to every other token. Token-validity
    masking is a deferred follow-up (see ``.agents/architecture.md``).

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads; must divide ``embed_dim``.
        dropout: Attention dropout probability, applied only during training.
    """

    def __init__(self, *, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.dropout = dropout
        # Combined projection for query, key, value in a single matmul.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention to every token in the sequence.

        Args:
            x: Token sequence, shape (B, L, D) — L is the full concatenated
                multi-source sequence length.

        Returns:
            Attended sequence, shape (B, L, D).
        """
        # Project to packed q/k/v, then split into three (B, L, D) tensors.
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, D) -> (B, num_heads, L, head_dim) for per-head attention.
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.num_heads) for t in (q, k, v))
        # Fused, optimized attention kernel (flash / mem-efficient / math backend
        # chosen automatically by PyTorch). Internally scales by 1/sqrt(head_dim).
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        # (B, num_heads, L, head_dim) -> (B, L, D), merging heads back together.
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        return self.out_proj(attn_out)
