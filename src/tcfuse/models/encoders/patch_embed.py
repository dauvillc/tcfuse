"""Value-only patch-embedding encoders, one per SourceKind.

Inputs are assumed finite: NaN-fill positions are zeroed upstream by
``BaseLightningModule.preprocess_batch`` before the batch reaches the encoder.
Coordinate / time encoding is intentionally deferred to a later iteration.
"""

from __future__ import annotations

from torch import nn

from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.base import SourceEncoder
from tcfuse.models.encoders.embedded import EmbeddedSource


class ScalarEncoder(SourceEncoder):
    """Embed a SCALAR source: ``values (B, C)`` → ``features (B, D)``.

    A single linear projection of the channel vector. ``patch_size`` is accepted
    for a uniform constructor signature but is unused (a scalar has no spatial axis).

    Args:
        source_name: Source identifier this encoder is bound to.
        num_channels: Number of input channels C.
        embed_dim: Output embedding dimension D.
        patch_size: Ignored for SCALAR sources.
    """

    def __init__(
        self, *, source_name: str, num_channels: int, embed_dim: int, patch_size: int
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        # Project the channel vector to the embedding dimension.
        self.proj = nn.Linear(num_channels, embed_dim)

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Project (B, C) channel vectors to (B, D) embeddings."""
        # values: (B, C) → (B, D)
        features = self.proj(source.values)
        return EmbeddedSource(kind=source.kind, features=features, source_name=source.source_name)


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
        self, *, source_name: str, num_channels: int, embed_dim: int, patch_size: int
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        # Non-overlapping 1D patches: kernel == stride == patch_size.
        self.proj = nn.Conv1d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Patch-embed (B, L, C) profiles into (B, El, D) tokens."""
        # Conv1d expects channels-first: (B, L, C) → (B, C, L).
        x = source.values.permute(0, 2, 1)
        # Strided conv collapses each patch of p levels into one token: (B, D, El).
        x = self.proj(x)
        # Back to tokens-last: (B, D, El) → (B, El, D).
        features = x.permute(0, 2, 1)
        return EmbeddedSource(kind=source.kind, features=features, source_name=source.source_name)


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
        self, *, source_name: str, num_channels: int, embed_dim: int, patch_size: int
    ) -> None:
        super().__init__(
            source_name=source_name,
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        # Non-overlapping 2D patches: kernel == stride == patch_size.
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, source: TorchSource) -> EmbeddedSource:
        """Patch-embed (B, H, W, C) fields into (B, Eh, Ew, D) tokens."""
        # Conv2d expects channels-first: (B, H, W, C) → (B, C, H, W).
        x = source.values.permute(0, 3, 1, 2)
        # Strided conv collapses each p-by-p patch into one token: (B, D, Eh, Ew).
        x = self.proj(x)
        # Back to tokens-last: (B, D, Eh, Ew) → (B, Eh, Ew, D).
        features = x.permute(0, 2, 3, 1)
        return EmbeddedSource(kind=source.kind, features=features, source_name=source.source_name)
