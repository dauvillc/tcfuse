"""Unit tests for source embedding encoders (synthetic tensors only, no real data)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.embedded import EmbeddedSource
from tcfuse.models.encoders.multisource import MultiSourceEncoder
from tcfuse.models.encoders.patch_embed import FieldEncoder, ProfileEncoder, ScalarEncoder
from tcfuse.models.encoders.positional import CoordEncodingConfig

# Default coordinate-encoding config used across encoder construction in tests.
COORD_CFG = CoordEncodingConfig()

# ---------------------------------------------------------------------------
# Synthetic source builders (mirror tests/test_sources.py conventions).
# ---------------------------------------------------------------------------


def make_scalar_source(B: int = 3, C: int = 5, source_name: str = "best_track") -> TorchSource:
    """Build a synthetic batched SCALAR source: values (B, C)."""
    values = torch.randn(B, C)
    return TorchSource(
        kind=SourceKind.SCALAR,
        values=values,
        coords=torch.randn(B, 2),
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


def make_profile_source(
    B: int = 3, L: int = 12, C: int = 5, source_name: str = "dropsonde"
) -> TorchSource:
    """Build a synthetic batched PROFILE source: values (B, L, C)."""
    values = torch.randn(B, L, C)
    return TorchSource(
        kind=SourceKind.PROFILE,
        values=values,
        coords=torch.randn(B, L, 3),
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


def make_field_source(
    B: int = 2, H: int = 8, W: int = 12, C: int = 2, source_name: str = "pmw_ssmi"
) -> TorchSource:
    """Build a synthetic batched FIELD source: values (B, H, W, C)."""
    values = torch.randn(B, H, W, C)
    return TorchSource(
        kind=SourceKind.FIELD,
        values=values,
        coords=torch.randn(B, H, W, 2),
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


# ---------------------------------------------------------------------------
# Per-kind encoder shape tests.
# ---------------------------------------------------------------------------


def test_scalar_encoder_shape() -> None:
    """ScalarEncoder maps (B, C) values to (B, D) features."""
    B, C, D = 3, 5, 16
    source = make_scalar_source(B=B, C=C)
    encoder = ScalarEncoder(
        source_name="best_track",
        num_channels=C,
        embed_dim=D,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.SCALAR
    assert embedded.features.shape == (B, D)
    assert embedded.embed_dim == D
    assert embedded.n_tokens == 1


def test_profile_encoder_shape() -> None:
    """ProfileEncoder maps (B, L, C) to (B, L // p, D)."""
    B, L, C, D, p = 3, 12, 5, 16, 4
    source = make_profile_source(B=B, L=L, C=C)
    encoder = ProfileEncoder(
        source_name="dropsonde", num_channels=C, embed_dim=D, patch_size=p, coord_encoding=COORD_CFG
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.PROFILE
    assert embedded.features.shape == (B, L // p, D)
    assert embedded.embedded_shape == (L // p,)


def test_field_encoder_shape() -> None:
    """FieldEncoder maps (B, H, W, C) to (B, H // p, W // p, D)."""
    B, H, W, C, D, p = 2, 8, 12, 2, 16, 4
    source = make_field_source(B=B, H=H, W=W, C=C)
    encoder = FieldEncoder(
        source_name="pmw_ssmi", num_channels=C, embed_dim=D, patch_size=p, coord_encoding=COORD_CFG
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.FIELD
    assert embedded.features.shape == (B, H // p, W // p, D)
    assert embedded.embedded_shape == (H // p, W // p)
    assert embedded.n_tokens == (H // p) * (W // p)


# ---------------------------------------------------------------------------
# EmbeddedSource validation.
# ---------------------------------------------------------------------------


def test_embedded_source_rejects_wrong_rank() -> None:
    """EmbeddedSource._validate rejects a tensor rank that mismatches its kind."""
    # A FIELD source must be 4-D (B, Eh, Ew, D); a 3-D tensor is invalid.
    with pytest.raises(ValueError, match="FIELD features must be 4-D"):
        EmbeddedSource(kind=SourceKind.FIELD, features=torch.randn(2, 4, 16), source_name="x")


# ---------------------------------------------------------------------------
# MultiSourceEncoder dispatcher.
# ---------------------------------------------------------------------------


def test_multisource_encoder_dispatch() -> None:
    """MultiSourceEncoder embeds one source per kind and carries is_target through."""
    B, D, p = 2, 16, 4
    L, H, W = 12, 8, 12
    Cs, Cp, Cf = 5, 4, 2

    # Build a metadata entry per source/kind matching the synthetic sources below.
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

    # Assemble a WindowBatch with one slot (index 0) per source.
    sources = {
        ("best_track", 0): make_scalar_source(B=B, C=Cs, source_name="best_track"),
        ("dropsonde", 0): make_profile_source(B=B, L=L, C=Cp, source_name="dropsonde"),
        ("pmw_ssmi", 0): make_field_source(B=B, H=H, W=W, C=Cf, source_name="pmw_ssmi"),
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[("pmw_ssmi", 0)] = torch.ones(B, dtype=torch.bool)
    batch = WindowBatch(
        sources=sources,
        is_target=is_target,
        sample_ids=[f"s{i}" for i in range(B)],
        window_ref_times_utc=["t"] * B,
        window_start_times_utc=["t"] * B,
        window_end_times_utc=["t"] * B,
        sids=["sid"] * B,
        seasons=[2020] * B,
        basins=["NA"] * B,
        subbasins=["MM"] * B,
        usa_atcf_ids=[None] * B,
    )

    encoder = MultiSourceEncoder(metadata, embed_dim=D, patch_size=p)
    embedded = encoder(batch)

    # Same keys, embedded to the expected per-kind shapes.
    assert set(embedded.sources) == set(sources)
    assert embedded.sources[("best_track", 0)].features.shape == (B, D)
    assert embedded.sources[("dropsonde", 0)].features.shape == (B, L // p, D)
    assert embedded.sources[("pmw_ssmi", 0)].features.shape == (B, H // p, W // p, D)
    assert embedded.batch_size == B
    # is_target is passed through unchanged.
    assert torch.equal(embedded.is_target[("pmw_ssmi", 0)], torch.ones(B, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Spatio-temporal Fourier positional encoding.
# ---------------------------------------------------------------------------


def test_coord_encoding_changes_features_per_kind() -> None:
    """Additive PE makes features depend on coords: different coords -> different features."""
    B, D = 3, 16
    # One encoder per kind; fixed seed so the comparison is deterministic.
    cases = [
        (
            ScalarEncoder(
                source_name="s", num_channels=4, embed_dim=D, patch_size=4, coord_encoding=COORD_CFG
            ),
            make_scalar_source(B=B, C=4),
        ),
        (
            ProfileEncoder(
                source_name="p", num_channels=4, embed_dim=D, patch_size=4, coord_encoding=COORD_CFG
            ),
            make_profile_source(B=B, L=12, C=4),
        ),
        (
            FieldEncoder(
                source_name="f", num_channels=2, embed_dim=D, patch_size=4, coord_encoding=COORD_CFG
            ),
            make_field_source(B=B, H=8, W=12, C=2),
        ),
    ]
    for encoder, source in cases:
        encoder.eval()
        # Baseline embedding with the source's original coords.
        base = encoder(source).features
        # Same values, shifted coords (+10 degrees): the PE must shift the features.
        shifted = dataclasses.replace(source, coords=source.coords + 10.0)
        moved = encoder(shifted).features
        # Shapes preserved, but the content differs because the PE depends on coords.
        assert moved.shape == base.shape
        assert not torch.allclose(moved, base)


def test_coord_encoding_is_deterministic() -> None:
    """Identical inputs through the same encoder give identical features."""
    encoder = FieldEncoder(
        source_name="f", num_channels=2, embed_dim=16, patch_size=4, coord_encoding=COORD_CFG
    )
    encoder.eval()
    source = make_field_source(B=2, H=8, W=12, C=2)
    first = encoder(source).features
    second = encoder(source).features
    assert torch.equal(first, second)


def test_coord_encoding_disabled_ignores_coords() -> None:
    """With enabled=False, features are value-only and invariant to coords."""
    disabled = CoordEncodingConfig(enabled=False)
    encoder = FieldEncoder(
        source_name="f", num_channels=2, embed_dim=16, patch_size=4, coord_encoding=disabled
    )
    encoder.eval()
    # No positional-encoding submodule should be allocated when disabled.
    assert encoder.coord_encoder is None
    source = make_field_source(B=2, H=8, W=12, C=2)
    base = encoder(source).features
    # Changing coords must not change the (value-only) features.
    shifted = dataclasses.replace(source, coords=source.coords + 25.0)
    moved = encoder(shifted).features
    assert torch.equal(base, moved)


def test_coord_encoding_frequencies_are_not_trainable() -> None:
    """The Fourier frequency bank is a non-trainable buffer, not a parameter."""
    encoder = ScalarEncoder(
        source_name="s", num_channels=4, embed_dim=16, patch_size=4, coord_encoding=COORD_CFG
    )
    assert encoder.coord_encoder is not None
    freqs = encoder.coord_encoder.frequencies
    assert not freqs.requires_grad
    # The frequency buffer must not appear among trainable parameters.
    param_ids = {id(p) for p in encoder.parameters()}
    assert id(freqs) not in param_ids


def test_coord_encoding_finite_output() -> None:
    """Finite coords/time produce finite features (no NaN/inf leaking through)."""
    encoder = ProfileEncoder(
        source_name="p", num_channels=4, embed_dim=16, patch_size=4, coord_encoding=COORD_CFG
    )
    encoder.eval()
    source = make_profile_source(B=3, L=12, C=4)
    features = encoder(source).features
    assert torch.isfinite(features).all()


def test_encoder_zeros_nan_in_values_coords_and_time() -> None:
    """NaN-fill in values/coords/time is handled inside the encoder, yielding finite features."""
    encoder = FieldEncoder(
        source_name="f", num_channels=2, embed_dim=16, patch_size=4, coord_encoding=COORD_CFG
    )
    encoder.eval()
    source = make_field_source(B=2, H=8, W=12, C=2)
    # Mark the entire second sample as a missing slot: values, coords, and time all NaN.
    source.values[1] = float("nan")
    source.coords[1] = float("nan")
    source.time[1] = float("nan")
    features = encoder(source).features
    # The encoder must not let NaNs leak into the embedded tokens.
    assert torch.isfinite(features).all()


def test_encoder_nan_values_match_zeroed_values() -> None:
    """Encoding NaN values equals encoding the same source with those NaNs replaced by 0."""
    encoder = ScalarEncoder(
        source_name="s", num_channels=4, embed_dim=16, patch_size=4, coord_encoding=COORD_CFG
    )
    encoder.eval()
    source = make_scalar_source(B=3, C=4)
    # Inject NaN into a couple of value entries (coords/time stay finite and shared).
    source.values[0, 0] = float("nan")
    source.values[2, 3] = float("nan")
    with_nan = encoder(source).features
    # Same source but with NaN values pre-zeroed: the encoder must produce identical output.
    zeroed = dataclasses.replace(source, values=torch.nan_to_num(source.values, nan=0.0))
    without_nan = encoder(zeroed).features
    assert torch.equal(with_nan, without_nan)
