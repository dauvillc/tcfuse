"""Unit tests for the cross-sequence transformer backbone (synthetic tensors only)."""

from __future__ import annotations

import pytest
import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.cross_transformer.backbone import CrossSequenceTransformerBackbone

# ---------------------------------------------------------------------------
# Synthetic source builders (mirror tests/test_perceiver.py conventions).
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


def make_metadata(*sources: TorchSource) -> MultisourceMetadata:
    """Build a MultisourceMetadata matching the given synthetic sources."""
    type_by_name = {
        "best_track": "best_track",
        "dropsonde": "dropsonde",
        "pmw_ssmi": "microwave",
    }
    metas: dict[str, SourceMetadata] = {}
    for src in sources:
        if src.kind is SourceKind.SCALAR:
            shape: tuple[int, ...] = ()
        elif src.kind is SourceKind.PROFILE:
            shape = (src.values.shape[1],)
        else:  # FIELD
            shape = (src.values.shape[1], src.values.shape[2])
        metas[src.source_name] = SourceMetadata(
            name=src.source_name,
            type=type_by_name.get(src.source_name, "microwave"),
            kind=src.kind,
            channels=src.channels,
            shape=shape,
        )
    return MultisourceMetadata(sources=metas)


def make_window_batch(
    sources: dict[tuple[str, int], TorchSource],
    is_target: dict[tuple[str, int], torch.Tensor],
    B: int,
) -> WindowBatch:
    """Wrap synthetic sources in a WindowBatch with placeholder metadata fields."""
    return WindowBatch(
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


def make_backbone(metadata: MultisourceMetadata) -> CrossSequenceTransformerBackbone:
    """Build a small CrossSequenceTransformerBackbone with head-divisible dims."""
    return CrossSequenceTransformerBackbone(
        metadata,
        embed_dim=16,
        patch_size=4,
        num_layers=2,
        num_heads=2,
    )


# ---------------------------------------------------------------------------
# Backbone tests.
# ---------------------------------------------------------------------------


def test_forward_preserves_shapes_no_target() -> None:
    """With no targets (empty target stream), shapes/keys are preserved.

    All-False ``is_target`` makes the target stream length zero, so the block
    stack is effectively a no-op; the backbone must still return the same keys
    and value shapes as the input.
    """
    B = 3
    scalar = make_scalar_source(B=B)
    profile = make_profile_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (profile.source_name, 0): profile,
        (field.source_name, 0): field,
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, profile, field))
    output = backbone(batch)
    assert set(output.sources.keys()) == set(batch.sources.keys())
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape


def test_forward_per_sample_varying_target() -> None:
    """Different samples target different sources within one batch.

    Sample 0 targets the FIELD slot, sample 1 the PROFILE slot, sample 2 the
    SCALAR slot — exercising the ragged target/input split, padded attention
    masks, and scatter-back across heterogeneous target sizes.
    """
    B = 3
    scalar = make_scalar_source(B=B)
    profile = make_profile_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (profile.source_name, 0): profile,
        (field.source_name, 0): field,
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[(field.source_name, 0)][0] = True
    is_target[(profile.source_name, 0)][1] = True
    is_target[(scalar.source_name, 0)][2] = True
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, profile, field))
    output = backbone(batch)
    assert set(output.sources.keys()) == set(batch.sources.keys())
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape
        assert torch.isfinite(output.sources[key].values).all()


def test_gradients_flow_through_all_components() -> None:
    """Backprop reaches the encoder, cross/self attentions, blocks, and decoder."""
    B = 3
    scalar = make_scalar_source(B=B)
    profile = make_profile_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (profile.source_name, 0): profile,
        (field.source_name, 0): field,
    }
    # Give every sample a (non-empty) target so the block stack receives gradient.
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[(field.source_name, 0)][:] = True
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, profile, field))
    output = backbone(batch)
    loss = torch.stack([source.values.sum() for source in output.sources.values()]).sum()
    loss.backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"


def test_forward_under_autocast_mixed_dtype() -> None:
    """Forward runs under AMP autocast, where the target stream is lower precision.

    Regression test for the scatter dtype mismatch: under autocast the LayerNorm'd
    target tokens come back in bfloat16 while the cloned source sequence stays
    float32, so the scatter-back must reconcile the dtypes.
    """
    B = 3
    scalar = make_scalar_source(B=B)
    profile = make_profile_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (profile.source_name, 0): profile,
        (field.source_name, 0): field,
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[(field.source_name, 0)][:] = True
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, profile, field))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = backbone(batch)
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape


def test_empty_input_stream_raises() -> None:
    """A sample with every source as a target (no visible input) must raise.

    Masked reconstruction always leaves >=1 visible source; an all-target sample
    would feed SDPA an all-masked key row and silently emit NaN, so the backbone
    guards the invariant with a ValueError instead.
    """
    B = 2
    scalar = make_scalar_source(B=B)
    field = make_field_source(B=B)
    sources = {
        (scalar.source_name, 0): scalar,
        (field.source_name, 0): field,
    }
    # Sample 0 marks *both* of its sources as targets -> empty input stream.
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[(scalar.source_name, 0)][0] = True
    is_target[(field.source_name, 0)][0] = True
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, field))
    with pytest.raises(ValueError, match="at least one non-target"):
        backbone(batch)


def test_forward_preserves_shapes_non_divisible_patch_size() -> None:
    """Output shapes match inputs even when H, W, L are not multiples of patch_size.

    patch_size=4 with H=9 (pad 3), W=10 (pad 2), L=13 (pad 3) exercises the
    encoder pad + decoder crop path end-to-end.
    """
    B = 2
    scalar = make_scalar_source(B=B)
    field = make_field_source(B=B, H=9, W=10, source_name="pmw_ssmi")
    profile = make_profile_source(B=B, L=13, source_name="dropsonde")
    sources = {
        (scalar.source_name, 0): scalar,
        (field.source_name, 0): field,
        (profile.source_name, 0): profile,
    }
    is_target = {key: torch.zeros(B, dtype=torch.bool) for key in sources}
    is_target[(field.source_name, 0)][:] = True
    batch = make_window_batch(sources, is_target, B)
    backbone = make_backbone(make_metadata(scalar, field, profile))
    output = backbone(batch)
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape, (
            f"{key}: expected {source.values.shape}, got {output.sources[key].values.shape}"
        )
