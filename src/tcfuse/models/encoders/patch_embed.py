"""Patch-embedding encoders, one per SourceKind.

Each encoder owns its NaN handling: NaN-fill positions in values, coords, and
time (absent or masked slots) are zeroed inside the encoder before use, so no
upstream cleaning step is required. Each encoder patch-embeds the values, then
adds a spatio-temporal Fourier positional encoding (see
:class:`~tcfuse.models.encoders.positional.SpatioTemporalEncoding`) built from
the per-token coordinates and the per-sample time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.base import SourceEncoder
from tcfuse.models.encoders.embedded import EmbeddedSource
from tcfuse.models.encoders.positional import ANGULAR, VERTICAL, CoordEncodingConfig


class ScalarEncoder(SourceEncoder):
    """Embed a SCALAR source: ``values (B, C)`` → ``features (B, D)``.

    A single linear projection of the channel vector, plus an additive
    spatio-temporal positional encoding from the source's ``(B, 2)`` lat/lon and
    ``(B, 2)`` time. ``patch_size`` is accepted for a uniform constructor signature
    but is unused (a scalar has no spatial axis).

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        embed_dim: Output embedding dimension D.
        patch_size: Ignored for SCALAR sources.
        coord_encoding: Fourier positional-encoding hyperparameters.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        embed_dim: int,
        patch_size: int,
        coord_encoding: CoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Project the channel vector to the embedding dimension.
        self.proj = nn.Linear(num_channels, embed_dim)
        # SCALAR coords are [lat, lon] (both angular); no spatial reduction needed.
        self.coord_encoder = self._make_coord_encoding([ANGULAR, ANGULAR])

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Project (B, C) channel vectors to (B, D) embeddings."""
        # Zero NaN-fill values (absent/masked slots) before projecting.
        values = torch.nan_to_num(source.values, nan=0.0)
        # values: (B, C) → (B, D)
        features = self.proj(values)
        # Add the spatio-temporal positional encoding from per-sample coords/time.
        if self.coord_encoder is not None:
            features = features + self.coord_encoder(source.coords, source.time)
        return EmbeddedSource(
            kind=source.kind, features=features, source_name=source.source_name, input_shape=()
        )


class ProfileEncoder(SourceEncoder):
    """Embed a PROFILE source: ``values (B, L, C)`` → ``features (B, El, D)``.

    Patchify along the level axis with a 1D strided convolution; ``El = L // p``.

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        embed_dim: Output embedding dimension D.
        patch_size: Patch length p along the level axis.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        embed_dim: int,
        patch_size: int,
        coord_encoding: CoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Non-overlapping 1D patches: kernel == stride == patch_size.
        self.proj = nn.Conv1d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # PROFILE coords are [lat, lon, alt] = two angular axes + one vertical axis.
        self.coord_encoder = self._make_coord_encoding([ANGULAR, ANGULAR, VERTICAL])

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Patch-embed (B, L, C) profiles into (B, El, D) tokens."""
        L = source.values.shape[1]
        # Zero NaN-fill values (absent/masked slots) before patch-embedding.
        values = torch.nan_to_num(source.values, nan=0.0)
        # Conv1d expects channels-first: (B, L, C) → (B, C, L).
        x = values.permute(0, 2, 1)
        # Pad L to the next multiple of patch_size so every level belongs to a complete patch.
        pad_l = (-L) % self.patch_size
        if pad_l > 0:
            x = F.pad(x, (0, pad_l))
        # Strided conv collapses each patch of p levels into one token: (B, D, El).
        x = self.proj(x)
        # Back to tokens-last: (B, D, El) → (B, El, D).
        features = x.permute(0, 2, 1)
        # Add the positional encoding from per-token (patch-center) coords and time.
        if self.coord_encoder is not None:
            patch_coords = self._reduce_coords(source.coords)
            features = features + self.coord_encoder(patch_coords, source.time)
        return EmbeddedSource(
            kind=source.kind, features=features, source_name=source.source_name, input_shape=(L,)
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


class FieldEncoder(SourceEncoder):
    """Embed a FIELD source: ``values (B, H, W, C)`` → ``features (B, Eh, Ew, D)``.

    Patchify the spatial grid with a 2D strided convolution; ``Eh = H // p`` and
    ``Ew = W // p``.

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        embed_dim: Output embedding dimension D.
        patch_size: Square patch size p along height and width.
    """

    def __init__(
        self,
        *,
        source_name: str,
        num_channels: int,
        embed_dim: int,
        patch_size: int,
        coord_encoding: CoordEncodingConfig,
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            coord_encoding=coord_encoding,
        )
        # Non-overlapping 2D patches: kernel == stride == patch_size.
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # FIELD coords are [lat, lon] (both angular); one per pixel.
        self.coord_encoder = self._make_coord_encoding([ANGULAR, ANGULAR])

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Patch-embed (B, H, W, C) fields into (B, Eh, Ew, D) tokens."""
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
        # Strided conv collapses each p-by-p patch into one token: (B, D, Eh, Ew).
        x = self.proj(x)
        # Back to tokens-last: (B, D, Eh, Ew) → (B, Eh, Ew, D).
        features = x.permute(0, 2, 3, 1)
        # Add the positional encoding from per-patch (patch-center) coords and time.
        if self.coord_encoder is not None:
            patch_coords = self._reduce_coords(source.coords)
            features = features + self.coord_encoder(patch_coords, source.time)
        return EmbeddedSource(
            kind=source.kind, features=features, source_name=source.source_name, input_shape=(H, W)
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
