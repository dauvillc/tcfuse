"""Value-only patch-unembedding decoders for MoTiF, one per SourceKind.

Mirrors :mod:`tcfuse.models.decoders.patch_unembed`: each decoder inverts the
shape transform of its corresponding
:class:`~tcfuse.models.encoders_motif.patch_embed` encoder. Only the
``MotifEmbeddedSource.values`` token tensor is un-embedded; the standalone
coordinate tokens are ignored (the MoTiF backbone consumes them internally as
positional conditioning, not on the way out).
"""

from __future__ import annotations

from torch import nn

from tcfuse.models.decoders_motif.base import MotifSourceDecoder
from tcfuse.models.decoders_motif.decoded import MotifDecodedSource
from tcfuse.models.decoders_motif.icnr import icnr
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedSource


class MotifScalarDecoder(MotifSourceDecoder):
    """Un-embed a SCALAR source: ``values (B, Dv)`` → ``values (B, C)``.

    A single linear projection of the value-embedding vector. ``patch_size`` is
    accepted for a uniform constructor signature but is unused.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input value-embedding dimension Dv.
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
        # Project the value embedding back to the channel vector.
        self.proj = nn.Linear(embed_dim, num_channels)

    def forward(self, embedded: MotifEmbeddedSource) -> MotifDecodedSource:
        """Project (B, Dv) value tokens back to (B, C) values."""
        # values: (B, Dv) → (B, C); coords are ignored.
        values = self.proj(embedded.values)
        return MotifDecodedSource(
            kind=embedded.kind, values=values, source_name=embedded.source_name
        )


class MotifProfileDecoder(MotifSourceDecoder):
    """Un-embed a PROFILE source: ``values (B, El, Dv)`` → ``values (B, L, C)``.

    Un-patchify along the level axis with a 1D strided transposed convolution;
    the exact inverse of
    :class:`~tcfuse.models.encoders_motif.patch_embed.MotifProfileEncoder`'s
    ``Conv1d``. ``L = El * patch_size`` before cropping back to the original length.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input value-embedding dimension Dv.
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

    def forward(self, embedded: MotifEmbeddedSource) -> MotifDecodedSource:
        """Un-patch-embed (B, El, Dv) value tokens into (B, L, C) profiles."""
        # ConvTranspose1d expects channels-first: (B, El, Dv) → (B, Dv, El).
        x = embedded.values.permute(0, 2, 1)
        # Strided transposed conv expands each token back into p levels: (B, C, L_padded).
        x = self.proj(x)
        # Back to channels-last: (B, C, L_padded) → (B, L_padded, C).
        values = x.permute(0, 2, 1)
        # Crop away any padding added by the encoder to reach a patch_size multiple.
        (L,) = embedded.input_shape
        values = values[:, :L, :]
        return MotifDecodedSource(
            kind=embedded.kind, values=values, source_name=embedded.source_name
        )


class MotifFieldDecoder(MotifSourceDecoder):
    """Un-embed a FIELD source: ``values (B, Eh, Ew, Dv)`` → ``values (B, H, W, C)``.

    Sub-pixel convolution (Conv2d + PixelShuffle) with ICNR-initialized weights,
    rather than a plain transposed conv, to avoid the checkerboard artifacts that
    strided deconvolutions tend to produce on image-like FIELD data. The 3x3 conv
    mixes neighbouring tokens at the embedded resolution ``(Eh, Ew)`` before
    ``PixelShuffle`` does the actual upsampling to ``(Eh * p, Ew * p)`` — unlike
    the encoder's patchify conv, this is intentionally not a plain shape inverse.

    Args:
        source_name: Source identifier this decoder is bound to.
        num_channels: Number of output channels C.
        embed_dim: Input value-embedding dimension Dv.
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

    def forward(self, embedded: MotifEmbeddedSource) -> MotifDecodedSource:
        """Un-patch-embed (B, Eh, Ew, Dv) value tokens into (B, H, W, C) fields."""
        # Conv2d expects channels-first: (B, Eh, Ew, Dv) → (B, Dv, Eh, Ew).
        x = embedded.values.permute(0, 3, 1, 2)
        # Mix neighbouring tokens and expand the channel axis for pixel shuffle.
        x = self.conv(x)
        # Rearrange the expanded channels into spatial resolution: (B, C, H_padded, W_padded).
        x = self.pixel_shuffle(x)
        # Back to channels-last: (B, C, H_padded, W_padded) → (B, H_padded, W_padded, C).
        values = x.permute(0, 2, 3, 1)
        # Crop away any padding added by the encoder to reach a patch_size multiple.
        H, W = embedded.input_shape
        values = values[:, :H, :W, :]
        return MotifDecodedSource(
            kind=embedded.kind, values=values, source_name=embedded.source_name
        )
