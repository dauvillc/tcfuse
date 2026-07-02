"""Windowed cross-source attention for the MoTiF backbone (FIELD sources).

The first of the three layers in a MoTiF block. It lets every source read from the
*other* sources at a coarse (windowed) spatial resolution, while preserving each
source's full resolution on the value side:

- For each source, the value and coordinate tokens are tiled into ``window_size``x
  ``window_size`` spatial windows and **averaged** within each window, giving pooled
  query/key sequences whose token count is divided by ``window_size^2``.
- The pooled feature and coordinate sequences of all sources are concatenated into one
  cross-source sequence and fed to :class:`SpatiotemporalAttention` as queries/keys, so
  a single attention matrix mixes information across sources.
- The **value** tokens are not averaged (that would discard resolution); instead each
  window's pixels are stacked along the embedding dim (``Dv -> window_size^2*Dv``), so the
  re-weighted values ``V' = A*V`` keep every pixel. ``V'`` is then un-stacked back to the
  per-source token grid and returned.

Positional information enters only through the coordinate-score bias inside the
attention (relative positional bias); there is no RoPE. Only FIELD (2-D) sources are
supported in this version.
"""

from __future__ import annotations

from collections.abc import Hashable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from tcfuse.models.motif.attention import SpatiotemporalAttention


class MultiSourceCrossAttention(nn.Module):
    """Cross-source attention over windowed FIELD tokens.

    Consumes per-source value and coordinate token grids (already conditioned by the
    enclosing block) and returns updated value tokens of the same shape; the residual
    connection is the block's responsibility. Passing two plain ``{key: tensor}`` dicts
    (rather than the ``MotifEmbeddedSource`` dataclass) keeps the layer decoupled from
    the block's LN/shift/scale conditioning and directly testable.

    Args:
        dim: Value-token embedding dimension Dv (input and output).
        inner_dim: Width of the feature queries/keys; must divide ``num_heads``.
        window_size: Side length of the square spatial windows.
        num_heads: Number of attention heads. Must divide ``inner_dim``,
            ``coord_inner_dim`` and ``window_size^2 * value_inner_dim`` (the stacked
            value width).
        coord_dim: Coordinate-token embedding dimension Dc (input).
        coord_inner_dim: Width of the coordinate queries/keys; must divide ``num_heads``.
        value_inner_dim: Per-pixel value width used inside the attention. Defaults to
            ``inner_dim``.
        mask_self_attention: If ``True`` (default), mask out attention between tokens of
            the same source so this layer prioritises cross-source information.
        dropout: Attention dropout probability, applied only during training.
    """

    def __init__(
        self,
        *,
        dim: int,
        inner_dim: int,
        window_size: int,
        num_heads: int,
        coord_dim: int,
        coord_inner_dim: int,
        value_inner_dim: int | None = None,
        mask_self_attention: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.mask_self_attention = mask_self_attention
        # Per-pixel value width; defaults to the feature inner width.
        value_inner_dim = value_inner_dim if value_inner_dim is not None else inner_dim
        self.value_inner_dim = value_inner_dim

        # Project pooled features to (query, key), then RMS-normalise (QK-norm).
        self.f_qk_proj = nn.Sequential(nn.Linear(dim, inner_dim * 2), nn.RMSNorm(inner_dim * 2))
        # Project pooled coordinates to (query, key), then RMS-normalise.
        self.c_qk_proj = nn.Sequential(
            nn.Linear(coord_dim, coord_inner_dim * 2), nn.RMSNorm(coord_inner_dim * 2)
        )
        # Per-pixel compression to the value width, and the inverse projection back.
        self.v_proj = nn.Linear(dim, value_inner_dim, bias=False)
        self.v_back_proj = nn.Linear(value_inner_dim, dim, bias=False)
        # The attention itself: feature scores plus alpha-weighted coordinate-score bias.
        self.attention = SpatiotemporalAttention(
            feature_dim=inner_dim, coord_dim=coord_inner_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(
        self, values: dict[Hashable, Tensor], coords: dict[Hashable, Tensor]
    ) -> dict[Hashable, Tensor]:
        """Run windowed cross-source attention over all sources.

        Args:
            values: Dict from source key to value tokens, shape (B, Eh, Ew, dim).
            coords: Dict from source key to coordinate tokens, shape (B, Eh, Ew, coord_dim).
                Keys and (Eh, Ew) grids must match ``values``.

        Returns:
            Dict from source key to updated value tokens, shape (B, Eh, Ew, dim) —
            identical shapes to the ``values`` inputs.
        """
        w = self.window_size
        f_queries, f_keys, c_queries, c_keys, stacked_values = {}, {}, {}, {}, {}
        # Number of windows per source, in iteration order — used to build the
        # block-diagonal mask and to split the attention output back per source.
        n_windows: list[int] = []

        for key in values:
            feats, crds = values[key], coords[key]  # (B, Eh, Ew, dim) / (B, Eh, Ew, coord_dim)

            # Pad the spatial dims up to a whole number of windows (0-pad on the right).
            _, eh, ew, _ = feats.shape
            pad_h = (w - eh % w) % w
            pad_w = (w - ew % w) % w
            feats = F.pad(feats, (0, 0, 0, pad_w, 0, pad_h))
            crds = F.pad(crds, (0, 0, 0, pad_w, 0, pad_h))

            # Tile into windows: (B, Wh, Ww, w*w, D) with the window pixels on axis -2.
            feats = rearrange(feats, "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d", w1=w, w2=w)
            crds = rearrange(crds, "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d", w1=w, w2=w)
            # Number of windows for this source (Wh * Ww).
            wh, ww = feats.shape[1], feats.shape[2]
            n_windows.append(wh * ww)

            # Pool features/coords over the window pixels -> one token per window.
            f_avg = feats.mean(dim=-2)  # (B, Wh, Ww, dim)
            c_avg = crds.mean(dim=-2)  # (B, Wh, Ww, coord_dim)
            # Project to (query, key) and flatten the window grid to a token axis.
            f_qk = rearrange(self.f_qk_proj(f_avg), "b Wh Ww d -> b (Wh Ww) d")
            c_qk = rearrange(self.c_qk_proj(c_avg), "b Wh Ww d -> b (Wh Ww) d")
            f_queries[key], f_keys[key] = f_qk.chunk(2, dim=-1)
            c_queries[key], c_keys[key] = c_qk.chunk(2, dim=-1)

            # Values: compress per pixel, then stack the window pixels into the feature
            # dim so the attention keeps full resolution. (B, Wh*Ww, w*w * value_inner_dim).
            v = self.v_proj(feats)  # (B, Wh, Ww, w*w, value_inner_dim)
            stacked_values[key] = rearrange(v, "b Wh Ww n d -> b (Wh Ww) (n d)")

        # Concatenate every source's tokens into one cross-source sequence.
        f_q = torch.cat(list(f_queries.values()), dim=-2)  # (B, N, inner_dim)
        f_k = torch.cat(list(f_keys.values()), dim=-2)
        c_q = torch.cat(list(c_queries.values()), dim=-2)  # (B, N, coord_inner_dim)
        c_k = torch.cat(list(c_keys.values()), dim=-2)
        v = torch.cat(list(stacked_values.values()), dim=-2)  # (B, N, w*w * value_inner_dim)

        # Optionally mask same-source blocks so tokens attend only across sources.
        attn_mask = None
        if self.mask_self_attention:
            # One all-True block per source on the diagonal; invert so True = cross-source.
            blocks = [torch.full((n, n), True) for n in n_windows]
            attn_mask = ~torch.block_diag(*blocks)  # (N, N), True off the source blocks
            attn_mask = attn_mask.to(v.device)

        # Re-weight the stacked values by cross-source attention. (B, N, w*w * value_inner_dim).
        v_out = self.attention(f_q, f_k, c_q, c_k, v, attn_mask=attn_mask)

        # Un-stack the window pixels and project each back to the value dim.
        v_out = rearrange(v_out, "b N (n d) -> b N n d", n=w * w)  # (B, N, w*w, value_inner_dim)
        v_out = self.v_back_proj(v_out)  # (B, N, w*w, dim)
        # Split the cross-source sequence back into per-source chunks.
        chunks = torch.split(v_out, n_windows, dim=1)

        outputs: dict[Hashable, Tensor] = {}
        for key, chunk in zip(values, chunks):
            eh, ew = values[key].shape[1], values[key].shape[2]
            # Number of windows along each axis (from the padded, tiled grid).
            wh = -(-eh // w)
            # Un-tile windows back to the padded spatial grid, then crop off the padding.
            out = rearrange(chunk, "b (Wh Ww) (w1 w2) d -> b (Wh w1) (Ww w2) d", Wh=wh, w1=w, w2=w)
            outputs[key] = out[:, :eh, :ew, :]
        return outputs
