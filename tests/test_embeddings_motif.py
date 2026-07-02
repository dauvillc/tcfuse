"""Unit tests for MoTiF dual-tensor embedding encoders (synthetic tensors only, no real data)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders_motif.base import MotifSourceEncoder
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedSource
from tcfuse.models.encoders_motif.multisource import MotifMultiSourceEncoder
from tcfuse.models.encoders_motif.patch_embed import (
    MotifFieldEncoder,
    MotifProfileEncoder,
    MotifScalarEncoder,
)
from tcfuse.models.encoders_motif.positional import MotifCoordEncodingConfig

# Default coordinate-embedding config used across encoder construction in tests.
COORD_CFG = MotifCoordEncodingConfig()

# Distinct value / coordinate embedding dims to exercise the Dv != Dc asymmetry.
DV, DC = 16, 8

# ---------------------------------------------------------------------------
# Synthetic source builders (mirror tests/test_embeddings.py conventions).
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
# Per-kind encoder shape tests (Dv != Dc throughout).
# ---------------------------------------------------------------------------


def test_scalar_encoder_shapes() -> None:
    """MotifScalarEncoder maps (B, C) values to (B, Dv) values and (B, Dc) coords."""
    B, C = 3, 5
    source = make_scalar_source(B=B, C=C)
    encoder = MotifScalarEncoder(
        source_name="best_track",
        num_channels=C,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.SCALAR
    # Both output tensors, each with its own embedding dim.
    assert embedded.values.shape == (B, DV)
    assert embedded.coords.shape == (B, DC)
    assert embedded.value_dim == DV
    assert embedded.coord_dim == DC
    assert embedded.n_tokens == 1


def test_profile_encoder_shapes() -> None:
    """MotifProfileEncoder maps (B, L, C) to (B, L // p, Dv) values and (B, L // p, Dc) coords."""
    B, L, C, p = 3, 12, 5, 4
    source = make_profile_source(B=B, L=L, C=C)
    encoder = MotifProfileEncoder(
        source_name="dropsonde",
        num_channels=C,
        value_dim=DV,
        coord_dim=DC,
        patch_size=p,
        coord_encoding=COORD_CFG,
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.PROFILE
    # Both tensors share the (B, El) token layout with distinct embedding dims.
    assert embedded.values.shape == (B, L // p, DV)
    assert embedded.coords.shape == (B, L // p, DC)
    assert embedded.embedded_shape == (L // p,)


def test_field_encoder_shapes() -> None:
    """MotifFieldEncoder maps (B, H, W, C) to (B, H // p, W // p, Dv/Dc) tensor pair."""
    B, H, W, C, p = 2, 8, 12, 2, 4
    source = make_field_source(B=B, H=H, W=W, C=C)
    encoder = MotifFieldEncoder(
        source_name="pmw_ssmi",
        num_channels=C,
        value_dim=DV,
        coord_dim=DC,
        patch_size=p,
        coord_encoding=COORD_CFG,
    )
    embedded = encoder(source)
    assert embedded.kind is SourceKind.FIELD
    # Both tensors share the (B, Eh, Ew) token layout with distinct embedding dims.
    assert embedded.values.shape == (B, H // p, W // p, DV)
    assert embedded.coords.shape == (B, H // p, W // p, DC)
    assert embedded.embedded_shape == (H // p, W // p)
    assert embedded.n_tokens == (H // p) * (W // p)


def test_non_divisible_spatial_dims_stay_in_sync() -> None:
    """Value padding and replicate-pooled coords produce matching token dims when p ∤ L/H/W."""
    p = 4
    # PROFILE with L not a multiple of p: both tensors must end up with ceil(L / p) tokens.
    L = 10
    profile = make_profile_source(B=2, L=L, C=3)
    profile_encoder = MotifProfileEncoder(
        source_name="dropsonde",
        num_channels=3,
        value_dim=DV,
        coord_dim=DC,
        patch_size=p,
        coord_encoding=COORD_CFG,
    )
    embedded = profile_encoder(profile)
    El = -(-L // p)  # ceil division
    assert embedded.values.shape[:-1] == embedded.coords.shape[:-1] == (2, El)
    assert embedded.input_shape == (L,)
    # FIELD with H, W not multiples of p: same ceil-divided token grid for both tensors.
    H, W = 7, 10
    fieldsrc = make_field_source(B=2, H=H, W=W, C=2)
    field_encoder = MotifFieldEncoder(
        source_name="pmw_ssmi",
        num_channels=2,
        value_dim=DV,
        coord_dim=DC,
        patch_size=p,
        coord_encoding=COORD_CFG,
    )
    embedded = field_encoder(fieldsrc)
    Eh, Ew = -(-H // p), -(-W // p)
    assert embedded.values.shape[:-1] == embedded.coords.shape[:-1] == (2, Eh, Ew)
    assert embedded.input_shape == (H, W)


# ---------------------------------------------------------------------------
# MotifEmbeddedSource validation.
# ---------------------------------------------------------------------------


def test_embedded_source_rejects_wrong_rank() -> None:
    """MotifEmbeddedSource rejects tensor ranks that mismatch the kind's layout."""
    # A FIELD source must be 4-D (B, Eh, Ew, D); a 3-D values tensor is invalid.
    with pytest.raises(ValueError, match="FIELD values must be 4-D"):
        MotifEmbeddedSource(
            kind=SourceKind.FIELD,
            values=torch.randn(2, 4, DV),
            coords=torch.randn(2, 4, 3, DC),
            source_name="x",
        )
    # Likewise for the coords tensor with a valid values tensor.
    with pytest.raises(ValueError, match="FIELD coords must be 4-D"):
        MotifEmbeddedSource(
            kind=SourceKind.FIELD,
            values=torch.randn(2, 4, 3, DV),
            coords=torch.randn(2, 4, DC),
            source_name="x",
        )


def test_embedded_source_rejects_mismatched_spatial_dims() -> None:
    """MotifEmbeddedSource enforces that values/coords share all non-embedding dims."""
    # Same rank but different token counts along the embedded axis: must raise.
    with pytest.raises(ValueError, match="share all dims except the last"):
        MotifEmbeddedSource(
            kind=SourceKind.PROFILE,
            values=torch.randn(2, 4, DV),
            coords=torch.randn(2, 5, DC),
            source_name="x",
        )


# ---------------------------------------------------------------------------
# Value / coordinate independence — the key MoTiF property.
# ---------------------------------------------------------------------------


def test_coords_and_time_only_affect_coord_tensor() -> None:
    """Shifting coords or time changes only the coords tensor; values stay bit-identical."""
    encoder_cases: list[tuple[MotifSourceEncoder, TorchSource]] = [
        (
            MotifScalarEncoder(
                source_name="s",
                num_channels=4,
                value_dim=DV,
                coord_dim=DC,
                patch_size=4,
                coord_encoding=COORD_CFG,
            ),
            make_scalar_source(B=3, C=4),
        ),
        (
            MotifProfileEncoder(
                source_name="p",
                num_channels=4,
                value_dim=DV,
                coord_dim=DC,
                patch_size=4,
                coord_encoding=COORD_CFG,
            ),
            make_profile_source(B=3, L=12, C=4),
        ),
        (
            MotifFieldEncoder(
                source_name="f",
                num_channels=2,
                value_dim=DV,
                coord_dim=DC,
                patch_size=4,
                coord_encoding=COORD_CFG,
            ),
            make_field_source(B=3, H=8, W=12, C=2),
        ),
    ]
    for encoder, source in encoder_cases:
        encoder.eval()
        # Baseline embedding with the source's original coords/time.
        base = encoder(source)
        # Shifting coords (+10 degrees) must change the coord tensor only.
        coord_shifted = encoder(dataclasses.replace(source, coords=source.coords + 10.0))
        assert torch.equal(coord_shifted.values, base.values)
        assert not torch.allclose(coord_shifted.coords, base.coords)
        # Shifting time must also change the coord tensor only.
        time_shifted = encoder(dataclasses.replace(source, time=source.time + 0.3))
        assert torch.equal(time_shifted.values, base.values)
        assert not torch.allclose(time_shifted.coords, base.coords)


def test_values_only_affect_value_tensor() -> None:
    """Changing source values changes only the values tensor; coords stay bit-identical."""
    encoder = MotifFieldEncoder(
        source_name="f",
        num_channels=2,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    encoder.eval()
    source = make_field_source(B=2, H=8, W=12, C=2)
    base = encoder(source)
    # Perturb the raw values while keeping coords/time untouched.
    perturbed = encoder(dataclasses.replace(source, values=source.values + 1.0))
    assert not torch.allclose(perturbed.values, base.values)
    assert torch.equal(perturbed.coords, base.coords)


# ---------------------------------------------------------------------------
# Coordinate embedding internals.
# ---------------------------------------------------------------------------


def test_coord_embedding_frequencies_are_not_trainable() -> None:
    """The Fourier frequency bank is a non-trainable buffer, not a parameter."""
    encoder = MotifScalarEncoder(
        source_name="s",
        num_channels=4,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    freqs = encoder.coord_embed.frequencies
    assert not freqs.requires_grad
    # The frequency buffer must not appear among trainable parameters.
    param_ids = {id(p) for p in encoder.parameters()}
    assert id(freqs) not in param_ids


def test_encoder_is_deterministic() -> None:
    """Identical inputs through the same encoder give identical value and coord tensors."""
    encoder = MotifFieldEncoder(
        source_name="f",
        num_channels=2,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    encoder.eval()
    source = make_field_source(B=2, H=8, W=12, C=2)
    first = encoder(source)
    second = encoder(source)
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.coords, second.coords)


# ---------------------------------------------------------------------------
# NaN handling.
# ---------------------------------------------------------------------------


def test_encoder_zeros_nan_in_values_coords_and_time() -> None:
    """NaN-fill in values/coords/time is handled inside the encoder, yielding finite outputs."""
    encoder = MotifFieldEncoder(
        source_name="f",
        num_channels=2,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    encoder.eval()
    source = make_field_source(B=2, H=8, W=12, C=2)
    # Mark the entire second sample as a missing slot: values, coords, and time all NaN.
    source.values[1] = float("nan")
    source.coords[1] = float("nan")
    source.time[1] = float("nan")
    embedded = encoder(source)
    # The encoder must not let NaNs leak into either embedded tensor.
    assert torch.isfinite(embedded.values).all()
    assert torch.isfinite(embedded.coords).all()


def test_encoder_nan_values_match_zeroed_values() -> None:
    """Encoding NaN values equals encoding the same source with those NaNs replaced by 0."""
    encoder = MotifScalarEncoder(
        source_name="s",
        num_channels=4,
        value_dim=DV,
        coord_dim=DC,
        patch_size=4,
        coord_encoding=COORD_CFG,
    )
    encoder.eval()
    source = make_scalar_source(B=3, C=4)
    # Inject NaN into a couple of value entries (coords/time stay finite and shared).
    source.values[0, 0] = float("nan")
    source.values[2, 3] = float("nan")
    with_nan = encoder(source)
    # Same source but with NaN values pre-zeroed: the encoder must produce identical output.
    zeroed = dataclasses.replace(source, values=torch.nan_to_num(source.values, nan=0.0))
    without_nan = encoder(zeroed)
    assert torch.equal(with_nan.values, without_nan.values)
    assert torch.equal(with_nan.coords, without_nan.coords)


# ---------------------------------------------------------------------------
# MotifMultiSourceEncoder dispatcher.
# ---------------------------------------------------------------------------


def test_multisource_encoder_dispatch() -> None:
    """MotifMultiSourceEncoder embeds one source per kind and carries is_target through."""
    B, p = 2, 4
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

    encoder = MotifMultiSourceEncoder(metadata, value_dim=DV, coord_dim=DC, patch_size=p)
    embedded = encoder(batch)

    # Same keys, embedded to the expected per-kind dual-tensor shapes.
    assert set(embedded.sources) == set(sources)
    assert embedded.sources[("best_track", 0)].values.shape == (B, DV)
    assert embedded.sources[("best_track", 0)].coords.shape == (B, DC)
    assert embedded.sources[("dropsonde", 0)].values.shape == (B, L // p, DV)
    assert embedded.sources[("dropsonde", 0)].coords.shape == (B, L // p, DC)
    assert embedded.sources[("pmw_ssmi", 0)].values.shape == (B, H // p, W // p, DV)
    assert embedded.sources[("pmw_ssmi", 0)].coords.shape == (B, H // p, W // p, DC)
    assert embedded.batch_size == B
    # is_target is passed through unchanged.
    assert torch.equal(embedded.is_target[("pmw_ssmi", 0)], torch.ones(B, dtype=torch.bool))
