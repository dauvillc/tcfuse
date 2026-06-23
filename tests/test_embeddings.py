"""Unit tests for source embedding encoders (synthetic tensors only, no real data)."""

from __future__ import annotations

import pytest
import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.encoders.embedded import EmbeddedSource
from tcfuse.models.encoders.multisource import MultiSourceEncoder
from tcfuse.models.encoders.patch_embed import FieldEncoder, ProfileEncoder, ScalarEncoder

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
    encoder = ScalarEncoder(source_name="best_track", num_channels=C, embed_dim=D, patch_size=4)
    embedded = encoder(source)
    assert embedded.kind is SourceKind.SCALAR
    assert embedded.features.shape == (B, D)
    assert embedded.embed_dim == D
    assert embedded.n_tokens == 1


def test_profile_encoder_shape() -> None:
    """ProfileEncoder maps (B, L, C) to (B, L // p, D)."""
    B, L, C, D, p = 3, 12, 5, 16, 4
    source = make_profile_source(B=B, L=L, C=C)
    encoder = ProfileEncoder(source_name="dropsonde", num_channels=C, embed_dim=D, patch_size=p)
    embedded = encoder(source)
    assert embedded.kind is SourceKind.PROFILE
    assert embedded.features.shape == (B, L // p, D)
    assert embedded.embedded_shape == (L // p,)


def test_field_encoder_shape() -> None:
    """FieldEncoder maps (B, H, W, C) to (B, H // p, W // p, D)."""
    B, H, W, C, D, p = 2, 8, 12, 2, 16, 4
    source = make_field_source(B=B, H=H, W=W, C=C)
    encoder = FieldEncoder(source_name="pmw_ssmi", num_channels=C, embed_dim=D, patch_size=p)
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
