"""Value-only patch-unembedding decoders, one per SourceKind.

Mirrors :mod:`tcfuse.models.encoders.patch_embed`: each decoder inverts the
shape transform of its corresponding encoder. Coordinate / time conditioning
is intentionally deferred, matching the encoder side.
"""

from __future__ import annotations

from torch import nn

from tcfuse.models.decoders.base import SourceDecoder
from tcfuse.models.decoders.decoded import DecodedSource
from tcfuse.models.decoders.icnr import icnr
from tcfuse.models.encoders.embedded import EmbeddedSource


class ScalarDecoder(SourceDecoder):
    """Un-embed a SCALAR source: ``features (B, D)`` → ``values (B, C)``.

    A single linear projection of the embedding vector. ``patch_size`` is
    accepted for a uniform constructor signature but is unused.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input embedding dimension D.
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
        # Project the embedding back to the channel vector.
        self.proj = nn.Linear(embed_dim, num_channels)

    def forward(self, embedded: EmbeddedSource) -> DecodedSource:
        """Project (B, D) embeddings back to (B, C) values."""
        # features: (B, D) → (B, C)
        values = self.proj(embedded.features)
        return DecodedSource(kind=embedded.kind, values=values, source_name=embedded.source_name)


class ProfileDecoder(SourceDecoder):
    """Un-embed a PROFILE source: ``features (B, El, D)`` → ``values (B, L, C)``.

    Un-patchify along the level axis with a 1D strided transposed convolution;
    the exact inverse of :class:`~tcfuse.models.encoders.patch_embed.ProfileEncoder`'s
    ``Conv1d``. ``L = El * patch_size``.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input embedding dimension D.
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
        # Non-overlapping 1D un-patchify: kernel == stride == patch_size.
        self.proj = nn.ConvTranspose1d(
            embed_dim, num_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, embedded: EmbeddedSource) -> DecodedSource:
        """Un-patch-embed (B, El, D) tokens into (B, L, C) profiles."""
        # ConvTranspose1d expects channels-first: (B, El, D) → (B, D, El).
        x = embedded.features.permute(0, 2, 1)
        # Strided transposed conv expands each token back into p levels: (B, C, L_padded).
        x = self.proj(x)
        # Back to channels-last: (B, C, L_padded) → (B, L_padded, C).
        values = x.permute(0, 2, 1)
        # Crop away any padding added by the encoder to reach a patch_size multiple.
        (L,) = embedded.input_shape
        values = values[:, :L, :]
        return DecodedSource(kind=embedded.kind, values=values, source_name=embedded.source_name)


class FieldDecoder(SourceDecoder):
    """Un-embed a FIELD source: ``features (B, Eh, Ew, D)`` → ``values (B, H, W, C)``.

    Sub-pixel convolution (Conv2d + PixelShuffle) with ICNR-initialized weights,
    rather than a plain transposed conv, to avoid the checkerboard artifacts that
    strided deconvolutions tend to produce on image-like FIELD data. The 3x3 conv
    mixes neighbouring tokens at the embedded resolution ``(Eh, Ew)`` before
    ``PixelShuffle`` does the actual upsampling to ``(Eh * p, Ew * p)`` — unlike
    the encoder's patchify conv, this is intentionally not a plain shape inverse.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input embedding dimension D.
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
        # 3x3 conv at stride 1 keeps (Eh, Ew) fixed; PixelShuffle does the upsampling.
        # No bias: ICNR initializes every patch_size**2 output group identically,
        # and a learnable bias would immediately break that symmetry at init.
        self.conv = nn.Conv2d(
            embed_dim,
            num_channels * patch_size**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        # ICNR init reduces checkerboard artifacts at the start of training.
        weight = icnr(
            self.conv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size,
        )
        self.conv.weight.data.copy_(weight)

    def forward(self, embedded: EmbeddedSource) -> DecodedSource:
        """Un-patch-embed (B, Eh, Ew, D) tokens into (B, H, W, C) fields."""
        # Conv2d expects channels-first: (B, Eh, Ew, D) → (B, D, Eh, Ew).
        x = embedded.features.permute(0, 3, 1, 2)
        # Mix neighbouring tokens and expand the channel axis for pixel shuffle.
        x = self.conv(x)
        # Rearrange the expanded channels into spatial resolution: (B, C, H_padded, W_padded).
        x = self.pixel_shuffle(x)
        # Back to channels-last: (B, C, H_padded, W_padded) → (B, H_padded, W_padded, C).
        values = x.permute(0, 2, 3, 1)
        # Crop away any padding added by the encoder to reach a patch_size multiple.
        H, W = embedded.input_shape
        values = values[:, :H, :W, :]
        return DecodedSource(kind=embedded.kind, values=values, source_name=embedded.source_name)
