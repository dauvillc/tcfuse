"""Unit tests for MoTiF value-only un-embedding decoders (synthetic tensors only, no real data)."""

from __future__ import annotations

import torch

from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.models.decoders_motif.decoded import MotifDecodedBatch
from tcfuse.models.decoders_motif.multisource import MotifMultiSourceDecoder
from tcfuse.models.decoders_motif.patch_unembed import (
    MotifFieldDecoder,
    MotifProfileDecoder,
    MotifScalarDecoder,
)
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedBatch, MotifEmbeddedSource

# Value / coordinate embedding dims; distinct to confirm coords are never read.
DV, DC = 16, 8
# Patch size shared across the PROFILE / FIELD decoders under test.
P = 4


# ---------------------------------------------------------------------------
# Synthetic MotifEmbeddedSource builders (values + independent coords).
# ---------------------------------------------------------------------------


def make_scalar_embedded(B: int = 3, source_name: str = "best_track") -> MotifEmbeddedSource:
    """Build a synthetic SCALAR MotifEmbeddedSource: values (B, Dv), coords (B, Dc)."""
    return MotifEmbeddedSource(
        kind=SourceKind.SCALAR,
        values=torch.randn(B, DV),
        coords=torch.randn(B, DC),
        source_name=source_name,
        input_shape=(),
    )


def make_profile_embedded(
    B: int = 3, El: int = 3, L: int = 10, source_name: str = "dropsonde"
) -> MotifEmbeddedSource:
    """Build a synthetic PROFILE MotifEmbeddedSource: values (B, El, Dv), coords (B, El, Dc)."""
    return MotifEmbeddedSource(
        kind=SourceKind.PROFILE,
        values=torch.randn(B, El, DV),
        coords=torch.randn(B, El, DC),
        source_name=source_name,
        input_shape=(L,),
    )


def make_field_embedded(
    B: int = 2, Eh: int = 2, Ew: int = 3, H: int = 7, W: int = 10, source_name: str = "pmw_ssmi"
) -> MotifEmbeddedSource:
    """Build a synthetic FIELD MotifEmbeddedSource: values/coords (B, Eh, Ew, Dv/Dc)."""
    return MotifEmbeddedSource(
        kind=SourceKind.FIELD,
        values=torch.randn(B, Eh, Ew, DV),
        coords=torch.randn(B, Eh, Ew, DC),
        source_name=source_name,
        input_shape=(H, W),
    )


# ---------------------------------------------------------------------------
# Per-kind shape inverse, including crop-back for non-patch-multiple lengths.
# ---------------------------------------------------------------------------


def test_scalar_decoder_shape() -> None:
    """MotifScalarDecoder maps (B, Dv) value tokens back to (B, C) values."""
    B, C = 3, 5
    embedded = make_scalar_embedded(B=B)
    decoder = MotifScalarDecoder(
        source_name="best_track", num_channels=C, embed_dim=DV, patch_size=P
    )
    decoded = decoder(embedded)
    assert decoded.kind is SourceKind.SCALAR
    assert decoded.values.shape == (B, C)
    assert torch.isfinite(decoded.values).all()


def test_profile_decoder_shape_and_crop() -> None:
    """MotifProfileDecoder inverts the level patchify and crops back to the original L."""
    B, C, L, El = 3, 5, 10, 3  # L not a multiple of P; El = ceil(L / P)
    embedded = make_profile_embedded(B=B, El=El, L=L)
    decoder = MotifProfileDecoder(
        source_name="dropsonde", num_channels=C, embed_dim=DV, patch_size=P
    )
    decoded = decoder(embedded)
    assert decoded.kind is SourceKind.PROFILE
    # Exact crop-back to the original (non-patch-multiple) level count.
    assert decoded.values.shape == (B, L, C)
    assert torch.isfinite(decoded.values).all()


def test_field_decoder_shape_and_crop() -> None:
    """MotifFieldDecoder inverts the spatial patchify and crops back to the original (H, W)."""
    B, C, H, W, Eh, Ew = 2, 2, 7, 10, 2, 3  # H, W not multiples of P
    embedded = make_field_embedded(B=B, Eh=Eh, Ew=Ew, H=H, W=W)
    decoder = MotifFieldDecoder(source_name="pmw_ssmi", num_channels=C, embed_dim=DV, patch_size=P)
    decoded = decoder(embedded)
    assert decoded.kind is SourceKind.FIELD
    # Exact crop-back to the original (non-patch-multiple) spatial grid.
    assert decoded.values.shape == (B, H, W, C)
    assert torch.isfinite(decoded.values).all()


# ---------------------------------------------------------------------------
# The key MoTiF property: decoding uses values only, never the coord tokens.
# ---------------------------------------------------------------------------


def test_coords_are_ignored() -> None:
    """Decoding is invariant to the coordinate tokens; only values drive the output."""
    B, C = 2, 2
    decoder = MotifFieldDecoder(source_name="pmw_ssmi", num_channels=C, embed_dim=DV, patch_size=P)
    decoder.eval()
    # Same value tokens, two different coordinate tensors.
    values = torch.randn(B, 2, 3, DV)
    first = MotifEmbeddedSource(
        kind=SourceKind.FIELD,
        values=values,
        coords=torch.randn(B, 2, 3, DC),
        source_name="pmw_ssmi",
        input_shape=(7, 10),
    )
    second = MotifEmbeddedSource(
        kind=SourceKind.FIELD,
        values=values,
        coords=torch.randn(B, 2, 3, DC) + 5.0,
        source_name="pmw_ssmi",
        input_shape=(7, 10),
    )
    # Different coords must not change the decoded values at all.
    assert torch.equal(decoder(first).values, decoder(second).values)


# ---------------------------------------------------------------------------
# MotifMultiSourceDecoder dispatcher.
# ---------------------------------------------------------------------------


def test_multisource_decoder_dispatch() -> None:
    """MotifMultiSourceDecoder decodes one source per kind and carries is_target through."""
    B = 2
    L, H, W = 10, 7, 10
    El, Eh, Ew = 3, 2, 3
    Cs, Cp, Cf = 5, 4, 2

    # One metadata entry per source/kind, matching the synthetic embedded sources below.
    metadata = MultisourceMetadata(
        sources={
            "best_track": SourceMetadata(
                name="best_track",
                type="best_track",
                kind=SourceKind.SCALAR,
                channels=[f"ch{i}" for i in range(Cs)],
                shape=(),
            ),
            "dropsonde": SourceMetadata(
                name="dropsonde",
                type="profile",
                kind=SourceKind.PROFILE,
                channels=[f"ch{i}" for i in range(Cp)],
                shape=(L,),
            ),
            "pmw_ssmi": SourceMetadata(
                name="pmw_ssmi",
                type="microwave",
                kind=SourceKind.FIELD,
                channels=[f"ch{i}" for i in range(Cf)],
                shape=(H, W),
            ),
        }
    )

    # One embedded slot (index 0) per source.
    sources = {
        ("best_track", 0): make_scalar_embedded(B=B, source_name="best_track"),
        ("dropsonde", 0): make_profile_embedded(B=B, El=El, L=L, source_name="dropsonde"),
        ("pmw_ssmi", 0): make_field_embedded(B=B, Eh=Eh, Ew=Ew, H=H, W=W, source_name="pmw_ssmi"),
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[("pmw_ssmi", 0)] = torch.ones(B, dtype=torch.bool)
    batch = MotifEmbeddedBatch(sources=sources, is_target=is_target)

    decoder = MotifMultiSourceDecoder(metadata, embed_dim=DV, patch_size=P)
    decoded = decoder(batch)

    # Same keys, decoded back to each kind's raw value layout with metadata channels.
    assert isinstance(decoded, MotifDecodedBatch)
    assert set(decoded.sources) == set(sources)
    assert decoded.sources[("best_track", 0)].values.shape == (B, Cs)
    assert decoded.sources[("dropsonde", 0)].values.shape == (B, L, Cp)
    assert decoded.sources[("pmw_ssmi", 0)].values.shape == (B, H, W, Cf)
    assert decoded.batch_size == B
    # is_target is passed through unchanged.
    assert torch.equal(decoded.is_target[("pmw_ssmi", 0)], torch.ones(B, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Backward pass.
# ---------------------------------------------------------------------------


def test_backward_produces_finite_grads() -> None:
    """A backward pass yields finite gradients on the decoder parameters."""
    decoder = MotifFieldDecoder(source_name="pmw_ssmi", num_channels=2, embed_dim=DV, patch_size=P)
    decoder.train()
    embedded = make_field_embedded(B=2, Eh=2, Ew=3, H=7, W=10)
    decoded = decoder(embedded)
    # Simple scalar objective over the decoded values.
    decoded.values.sum().backward()
    # Every trainable parameter that received a gradient must be finite.
    for param in decoder.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
