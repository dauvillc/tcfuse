"""Spatiotemporal attention: feature scores plus a coordinate-derived score bias.

The low-level attention used by the MoTiF cross-source layer. It computes the usual
feature (query/key) attention scores and adds a separate coordinate-score term that
acts as a learnable relative positional bias, following

    V_out = softmax( Qf*Kf^T / sqrt(d_f)  +  alpha * Qc*Kc^T / sqrt(d_c) ) * V

(the alpha-weighted coordinate term of the MoTiF scheme). Both terms are fused into a
single :func:`torch.nn.functional.scaled_dot_product_attention` call — the project's
universal attention idiom — by passing the coordinate scores (with any structural
mask folded in) as SDPA's additive float ``attn_mask``.

Unlike a standard multi-head attention block there is no output projection: the value
sequence carries the window-stacked features, which the enclosing cross-source layer
un-stacks and projects back itself. The only learnable parameter here is the scalar alpha.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class SpatiotemporalAttention(nn.Module):
    """Multi-head attention whose scores blend feature and coordinate similarities.

    The feature term is handled by the fused SDPA kernel (which scales it by
    ``1/sqrt(feature_head_dim)``); the coordinate term is computed here, scaled by the
    learnable weight ``alpha``, and passed in as SDPA's additive ``attn_mask`` so the
    two are summed before the softmax.

    Args:
        feature_dim: Width of the feature queries/keys (Df); must divide ``num_heads``.
        coord_dim: Width of the coordinate queries/keys (Dc); must divide ``num_heads``.
        num_heads: Number of attention heads. Must also divide the value width Dvs
            passed to :meth:`forward` (the value sequence is split into the same heads).
        dropout: Attention dropout probability, applied only during training.
    """

    def __init__(
        self, *, feature_dim: int, coord_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        # Per-head coordinate width, used to scale the coordinate scores like SDPA
        # scales the feature scores.
        self.coord_head_dim = coord_dim // num_heads
        # alpha: learnable weight on the coordinate-score bias. Initialised to 0 so the
        # layer starts as pure feature attention and learns to inject positional bias.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        f_q: Tensor,
        f_k: Tensor,
        c_q: Tensor,
        c_k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend with combined feature and coordinate scores.

        Args:
            f_q: Feature queries, shape (B, N, Df).
            f_k: Feature keys, shape (B, N, Df).
            c_q: Coordinate queries, shape (B, N, Dc).
            c_k: Coordinate keys, shape (B, N, Dc).
            v: Value sequence, shape (B, N, Dvs) — Dvs must divide ``num_heads``.
            attn_mask: Optional boolean keep-mask, shape broadcastable to (N, N)
                (e.g. (N, N) or (1, N, N)); ``True`` means the query/key pair may
                attend. ``None`` disables masking.

        Returns:
            Attended value sequence, shape (B, N, Dvs).
        """
        # Split every stream into heads: (B, N, H*d) -> (B, H, N, d).
        fq, fk = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (f_q, f_k))
        cq, ck = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (c_q, c_k))
        vh = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Coordinate scores, scaled like SDPA scales the feature scores. (B, H, N, N).
        coord_scores = (cq @ ck.transpose(-2, -1)) / math.sqrt(self.coord_head_dim)
        # Weight the coordinate bias by the learnable alpha; this is SDPA's additive mask.
        bias = self.alpha * coord_scores
        # Fold the structural keep-mask in: masked pairs get -inf so they vanish in
        # the softmax. attn_mask is bool (True = keep); broadcast over batch and heads.
        if attn_mask is not None:
            bias = bias.masked_fill(~attn_mask, float("-inf"))

        # Fused attention: softmax(Qf*Kf^T/sqrt(d_f) + bias) * V, in one kernel.
        out = F.scaled_dot_product_attention(
            fq,
            fk,
            vh,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # Merge heads back together: (B, H, N, d) -> (B, N, Dvs).
        return rearrange(out, "b h n d -> b n (h d)")
