"""Unit tests for the Perceiver IO backbone (synthetic tensors only)."""

from __future__ import annotations

import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.perceiver.backbone import PerceiverIOBackbone

# ---------------------------------------------------------------------------
# Synthetic source builders (mirror tests/test_transformer.py conventions).
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
    B: int = 3, H: int = 8, W: int = 12, C: int = 2, source_name: str = "pmw_ssmi"
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


def make_window_batch_and_metadata(B: int = 3) -> tuple[WindowBatch, MultisourceMetadata]:
    """Build a small multi-source WindowBatch (SCALAR + PROFILE + FIELD) and matching metadata."""
    scalar = make_scalar_source(B=B)
    profile = make_profile_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (profile.source_name, 0): profile,
        (field.source_name, 0): field,
    }
    # No target slots: every sample of every source is treated as visible input.
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    batch = WindowBatch(
        sources=sources,
        is_target=is_target,
        sample_ids=[f"sample{i}" for i in range(B)],
        window_ref_times_utc=["2020-01-01T00:00:00"] * B,
        window_start_times_utc=["2020-01-01T00:00:00"] * B,
        window_end_times_utc=["2020-01-01T00:00:00"] * B,
        sids=["2020001N00000"] * B,
        seasons=[2020] * B,
        basins=["NA"] * B,
        subbasins=["NA"] * B,
        usa_atcf_ids=[None] * B,
    )
    metadata = MultisourceMetadata(
        sources={
            scalar.source_name: SourceMetadata(
                name=scalar.source_name,
                type="best_track",
                kind=SourceKind.SCALAR,
                channels=scalar.channels,
                shape=(),
            ),
            profile.source_name: SourceMetadata(
                name=profile.source_name,
                type="dropsonde",
                kind=SourceKind.PROFILE,
                channels=profile.channels,
                shape=(profile.values.shape[1],),
            ),
            field.source_name: SourceMetadata(
                name=field.source_name,
                type="microwave",
                kind=SourceKind.FIELD,
                channels=field.channels,
                shape=(field.values.shape[1], field.values.shape[2]),
            ),
        }
    )
    return batch, metadata


def make_backbone(metadata: MultisourceMetadata) -> PerceiverIOBackbone:
    """Build a small PerceiverIOBackbone with dims that divide the head counts."""
    return PerceiverIOBackbone(
        metadata,
        embed_dim=16,
        patch_size=4,
        latent_dim=24,
        num_latents=8,
        num_layers=2,
        num_heads=2,
        cross_num_heads=2,
    )


# ---------------------------------------------------------------------------
# Backbone tests.
# ---------------------------------------------------------------------------


def test_forward_preserves_shapes_and_keys() -> None:
    """The backbone returns a WindowBatch with the same keys and value shapes as the input."""
    batch, metadata = make_window_batch_and_metadata()
    backbone = make_backbone(metadata)
    output = backbone(batch)
    assert set(output.sources.keys()) == set(batch.sources.keys())
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape


def test_gradients_flow_through_all_components() -> None:
    """Backprop reaches the encoder, latents, cross-attentions, latent blocks, and decoder."""
    batch, metadata = make_window_batch_and_metadata()
    backbone = make_backbone(metadata)
    output = backbone(batch)
    loss = torch.stack([source.values.sum() for source in output.sources.values()]).sum()
    loss.backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"


def test_forward_preserves_shapes_non_divisible_patch_size() -> None:
    """Backbone output shapes match inputs even when H, W, L are not multiples of patch_size.

    patch_size=4 with H=9 (pad 3), W=10 (pad 2), L=13 (pad 3) exercises the
    encoder pad + decoder crop path end-to-end.
    """
    B = 2
    scalar = make_scalar_source(B=B)
    # H=9 and W=10 are not divisible by patch_size=4.
    field = make_field_source(B=B, H=9, W=10, source_name="pmw_ssmi")
    # L=13 is not divisible by patch_size=4.
    profile = make_profile_source(B=B, L=13, source_name="dropsonde")
    sources = {
        (scalar.source_name, 0): scalar,
        (field.source_name, 0): field,
        (profile.source_name, 0): profile,
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    batch = WindowBatch(
        sources=sources,
        is_target=is_target,
        sample_ids=[f"sample{i}" for i in range(B)],
        window_ref_times_utc=["2020-01-01T00:00:00"] * B,
        window_start_times_utc=["2020-01-01T00:00:00"] * B,
        window_end_times_utc=["2020-01-01T00:00:00"] * B,
        sids=["2020001N00000"] * B,
        seasons=[2020] * B,
        basins=["NA"] * B,
        subbasins=["NA"] * B,
        usa_atcf_ids=[None] * B,
    )
    metadata = MultisourceMetadata(
        sources={
            scalar.source_name: SourceMetadata(
                name=scalar.source_name,
                type="best_track",
                kind=SourceKind.SCALAR,
                channels=scalar.channels,
                shape=(),
            ),
            field.source_name: SourceMetadata(
                name=field.source_name,
                type="microwave",
                kind=SourceKind.FIELD,
                channels=field.channels,
                shape=(field.values.shape[1], field.values.shape[2]),
            ),
            profile.source_name: SourceMetadata(
                name=profile.source_name,
                type="dropsonde",
                kind=SourceKind.PROFILE,
                channels=profile.channels,
                shape=(profile.values.shape[1],),
            ),
        }
    )
    backbone = make_backbone(metadata)
    output = backbone(batch)
    # Every output source must have the same value shape as the original input.
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape, (
            f"{key}: expected {source.values.shape}, got {output.sources[key].values.shape}"
        )
