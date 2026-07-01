"""Mask-capable self- and cross-attention for the cross-sequence transformer.

Mirrors :mod:`tcfuse.models.perceiver.attention` (deliberate per-backbone
redundancy, see ``.agents/architecture.md``) but adds an optional key-padding
mask so the input and target sequences can be padded to the batch-max length and
have their padding tokens excluded from attention. Both are computed with
PyTorch's fused ``scaled_dot_product_attention``.
"""

from __future__ import annotations

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def _key_padding_to_attn_mask(key_padding_mask: Tensor | None) -> Tensor | None:
    """Reshape a ``(B, Lkv)`` keep-mask into an SDPA-broadcastable attention mask.

    Args:
        key_padding_mask: Boolean tensor, shape ``(B, Lkv)``, ``True`` where the
            key/value token is real and should participate in attention. ``None``
            disables masking (dense attention).

    Returns:
        A ``(B, 1, 1, Lkv)`` boolean mask broadcasting over heads and all query
        rows — the layout SDPA expects (``True`` = participate) — or ``None``.
    """
    if key_padding_mask is None:
        return None
    # (B, Lkv) -> (B, 1, 1, Lkv): broadcasts over the head and query-position axes.
    return key_padding_mask[:, None, None, :]


class SelfAttention(nn.Module):
    """Self-attention over a ``(B, L, D)`` sequence, with optional key padding.

    Used on the target sequence. With ``key_padding_mask`` the padded target
    tokens are excluded as keys; every real query token still attends to every
    real key token.

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

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Apply self-attention to every token in the sequence.

        Args:
            x: Token sequence, shape (B, L, D).
            key_padding_mask: Optional ``(B, L)`` bool mask, ``True`` where a token
                is real and may be attended to as a key.

        Returns:
            Attended sequence, shape (B, L, D).
        """
        # Project to packed q/k/v, then split into three (B, L, D) tensors.
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, D) -> (B, num_heads, L, head_dim) for per-head attention.
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.num_heads) for t in (q, k, v))
        # Fused attention kernel; internally scales by 1/sqrt(head_dim).
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=_key_padding_to_attn_mask(key_padding_mask),
            dropout_p=self.dropout if self.training else 0.0,
        )
        # (B, num_heads, L, head_dim) -> (B, L, D), merging heads back together.
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        return self.out_proj(attn_out)


class CrossAttention(nn.Module):
    """Cross-attention from a query sequence onto a key/value sequence.

    Query and key/value sequences may live in different dimensionalities
    (``query_dim`` vs ``kv_dim``); the output is projected back into ``query_dim``
    space so a residual can be added on the query stream. With ``key_padding_mask``
    the padded key/value tokens are excluded.

    Args:
        query_dim: Dimension of the query tokens (and of the output).
        kv_dim: Dimension of the key/value tokens.
        num_heads: Number of attention heads; must divide ``query_dim``.
        dropout: Attention dropout probability, applied only during training.
    """

    def __init__(
        self, *, query_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.kv_proj = nn.Linear(kv_dim, 2 * query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query: Tensor, kv: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Attend ``query`` over ``kv``.

        Args:
            query: Query sequence, shape (B, Lq, query_dim).
            kv: Key/value sequence, shape (B, Lkv, kv_dim).
            key_padding_mask: Optional ``(B, Lkv)`` bool mask, ``True`` where a
                key/value token is real and may be attended to.

        Returns:
            Attended query sequence, shape (B, Lq, query_dim).
        """
        # Project each stream into the shared head space.
        q = self.q_proj(query)
        # Single fused key/value matmul, then split into (B, Lkv, query_dim) each.
        k, v = self.kv_proj(kv).chunk(2, dim=-1)
        # (B, L, query_dim) -> (B, num_heads, L, head_dim) for per-head attention.
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.num_heads) for t in (q, k, v))
        # Fused attention kernel; internally scales by 1/sqrt(head_dim).
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=_key_padding_to_attn_mask(key_padding_mask),
            dropout_p=self.dropout if self.training else 0.0,
        )
        # (B, num_heads, Lq, head_dim) -> (B, Lq, query_dim), merging heads back.
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        return self.out_proj(attn_out)
