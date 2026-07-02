"""MoTiF patch-embedding encoders, one per SourceKind.

Each encoder owns its NaN handling: NaN-fill positions in values, coords, and
time (absent or masked slots) are zeroed inside the encoder before use, so no
upstream cleaning step is required. Each encoder patch-embeds the values into a
**value-only** token tensor (no positional term added), and separately embeds
the per-token coordinates + per-sample time into a standalone coordinate tensor
(see :class:`~tcfuse.models.encoders_motif.positional.CoordEmbedding`) that the
MoTiF backbone injects as positional conditioning at every layer. Both tensors
share the same spatial/token dims; their last (embedding) dims may differ.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders_motif.base import MotifSourceEncoder
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedSource
from tcfuse.models.encoders_motif.positional import ANGULAR, VERTICAL, MotifCoordEncodingConfig


class MotifScalarEncoder(MotifSourceEncoder):
    """Embed a SCALAR source: ``values (B, C)`` → ``values (B, Dv)`` + ``coords (B, Dc)``.

    The value tensor is a single linear projection of the channel vector; the
    coordinate tensor is a standalone Fourier embedding of the source's ``(B, 2)``
    lat/lon and ``(B, 2)`` time. ``patch_size`` is accepted for a uniform
    constructor signature but is unused (a scalar has no spatial axis).

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        value_dim: Output value-embedding dimension Dv.
        coord_dim: Output coordinate-embedding dimension Dc.
        patch_size: Ignored for SCALAR sources.
        coord_encoding: Fourier coordinate-embedding hyperparameters.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        value_dim: int,
        coord_dim: int,
        patch_size: int,
        coord_encoding: MotifCoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            value_dim=value_dim,
            coord_dim=coord_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Project the channel vector to the value-embedding dimension.
        self.proj = nn.Linear(num_channels, value_dim)
        # SCALAR coords are [lat, lon] (both angular); no spatial reduction needed.
        self.coord_embed = self._make_coord_embedding([ANGULAR, ANGULAR])

    def forward(self, source: TorchSource) -> MotifEmbeddedSource:
        """Embed (B, C) channel vectors into (B, Dv) values and (B, Dc) coords."""
        # Zero NaN-fill values (absent/masked slots) before projecting.
        values = torch.nan_to_num(source.values, nan=0.0)
        # values: (B, C) → (B, Dv); no positional term is added to the value tokens.
        value_tokens = self.proj(values)
        # Standalone coordinate embedding from per-sample coords/time: (B, Dc).
        coord_tokens = self.coord_embed(source.coords, source.time)
        return MotifEmbeddedSource(
            kind=source.kind,
            values=value_tokens,
            coords=coord_tokens,
            source_name=source.source_name,
            input_shape=(),
        )


class MotifProfileEncoder(MotifSourceEncoder):
    """Embed a PROFILE source: ``(B, L, C)`` → ``values (B, El, Dv)`` + ``coords (B, El, Dc)``.

    Patchify along the level axis with a 1D strided convolution; ``El = L // p``.
    Per-level coordinates are average-pooled to patch centers before the Fourier
    coordinate embedding, so both tensors share the ``(B, El)`` token layout.

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        value_dim: Output value-embedding dimension Dv.
        coord_dim: Output coordinate-embedding dimension Dc.
        patch_size: Patch length p along the level axis.
        coord_encoding: Fourier coordinate-embedding hyperparameters.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        value_dim: int,
        coord_dim: int,
        patch_size: int,
        coord_encoding: MotifCoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            value_dim=value_dim,
            coord_dim=coord_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Non-overlapping 1D patches: kernel == stride == patch_size.
        self.proj = nn.Conv1d(num_channels, value_dim, kernel_size=patch_size, stride=patch_size)
        # PROFILE coords are [lat, lon, alt] = two angular axes + one vertical axis.
        self.coord_embed = self._make_coord_embedding([ANGULAR, ANGULAR, VERTICAL])

    def forward(self, source: TorchSource) -> MotifEmbeddedSource:
        """Embed (B, L, C) profiles into (B, El, Dv) values and (B, El, Dc) coords."""
        L = source.values.shape[1]
        # Zero NaN-fill values (absent/masked slots) before patch-embedding.
        values = torch.nan_to_num(source.values, nan=0.0)
        # Conv1d expects channels-first: (B, L, C) → (B, C, L).
        x = values.permute(0, 2, 1)
        # Pad L to the next multiple of patch_size so every level belongs to a complete patch.
        pad_l = (-L) % self.patch_size
        if pad_l > 0:
            x = F.pad(x, (0, pad_l))
        # Strided conv collapses each patch of p levels into one token: (B, Dv, El).
        x = self.proj(x)
        # Back to tokens-last: (B, Dv, El) → (B, El, Dv); no positional term added.
        value_tokens = x.permute(0, 2, 1)
        # Reduce per-level coords to per-patch centers, matching the token layout.
        patch_coords = self._reduce_coords(source.coords)
        # Standalone coordinate embedding from patch-center coords and time: (B, El, Dc).
        coord_tokens = self.coord_embed(patch_coords, source.time)
        return MotifEmbeddedSource(
            kind=source.kind,
            values=value_tokens,
            coords=coord_tokens,
            source_name=source.source_name,
            input_shape=(L,),
        )

    def _reduce_coords(self, coords: Tensor) -> Tensor:
        """Average-pool per-level coords (B, L, 3) to per-patch centers (B, El, 3)."""
        L = coords.shape[1]
        # Channels-first for pooling: (B, L, 3) → (B, 3, L).
        c = coords.permute(0, 2, 1)
        # Replicate-pad so edge patches average real coords, not zero-padding.
        pad_l = (-L) % self.patch_size
        if pad_l > 0:
            c = F.pad(c, (0, pad_l), mode="replicate")
        # Non-overlapping mean over each patch of p levels: (B, 3, El).
        c = F.avg_pool1d(c, kernel_size=self.patch_size, stride=self.patch_size)
        # Back to coords-last: (B, 3, El) → (B, El, 3).
        return c.permute(0, 2, 1)


class MotifFieldEncoder(MotifSourceEncoder):
    """Embed a FIELD source into ``(B, Eh, Ew, Dv)`` values + ``(B, Eh, Ew, Dc)`` coords.

    Patchify the spatial grid with a 2D strided convolution; ``Eh = H // p`` and
    ``Ew = W // p``. Per-pixel coordinates are average-pooled to patch centers
    before the Fourier coordinate embedding, so both tensors share the
    ``(B, Eh, Ew)`` token layout.

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        value_dim: Output value-embedding dimension Dv.
        coord_dim: Output coordinate-embedding dimension Dc.
        patch_size: Square patch size p along height and width.
        coord_encoding: Fourier coordinate-embedding hyperparameters.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        value_dim: int,
        coord_dim: int,
        patch_size: int,
        coord_encoding: MotifCoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            value_dim=value_dim,
            coord_dim=coord_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Non-overlapping 2D patches: kernel == stride == patch_size.
        self.proj = nn.Conv2d(num_channels, value_dim, kernel_size=patch_size, stride=patch_size)
        # FIELD coords are [lat, lon] (both angular); one per pixel.
        self.coord_embed = self._make_coord_embedding([ANGULAR, ANGULAR])

    def forward(self, source: TorchSource) -> MotifEmbeddedSource:
        """Embed (B, H, W, C) fields into (B, Eh, Ew, Dv) values and (B, Eh, Ew, Dc) coords."""
        H, W = source.values.shape[1], source.values.shape[2]
        # Zero NaN-fill values (absent/masked slots) before patch-embedding.
        values = torch.nan_to_num(source.values, nan=0.0)
        # Conv2d expects channels-first: (B, H, W, C) → (B, C, H, W).
        x = values.permute(0, 3, 1, 2)
        # Pad H and W to the next multiple of patch_size so every spatial patch is complete.
        # F.pad order for 4-D: (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom).
        pad_h, pad_w = (-H) % self.patch_size, (-W) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # Strided conv collapses each p-by-p patch into one token: (B, Dv, Eh, Ew).
        x = self.proj(x)
        # Back to tokens-last: (B, Dv, Eh, Ew) → (B, Eh, Ew, Dv); no positional term added.
        value_tokens = x.permute(0, 2, 3, 1)
        # Reduce per-pixel coords to per-patch centers, matching the token layout.
        patch_coords = self._reduce_coords(source.coords)
        # Standalone coordinate embedding from patch-center coords and time: (B, Eh, Ew, Dc).
        coord_tokens = self.coord_embed(patch_coords, source.time)
        return MotifEmbeddedSource(
            kind=source.kind,
            values=value_tokens,
            coords=coord_tokens,
            source_name=source.source_name,
            input_shape=(H, W),
        )

    def _reduce_coords(self, coords: Tensor) -> Tensor:
        """Average-pool per-pixel coords (B, H, W, 2) to per-patch centers (B, Eh, Ew, 2)."""
        H, W = coords.shape[1], coords.shape[2]
        # Channels-first for pooling: (B, H, W, 2) → (B, 2, H, W).
        c = coords.permute(0, 3, 1, 2)
        # Replicate-pad so edge patches average real coords, not zero-padding.
        pad_h, pad_w = (-H) % self.patch_size, (-W) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            c = F.pad(c, (0, pad_w, 0, pad_h), mode="replicate")
        # Non-overlapping mean over each p-by-p patch: (B, 2, Eh, Ew).
        c = F.avg_pool2d(c, kernel_size=self.patch_size, stride=self.patch_size)
        # Back to coords-last: (B, 2, Eh, Ew) → (B, Eh, Ew, 2).
        return c.permute(0, 2, 3, 1)
