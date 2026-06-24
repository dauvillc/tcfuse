"""Unit tests for the plain single-sequence transformer backbone (synthetic tensors only)."""

from __future__ import annotations

import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.models.transformer.backbone import SingleSequenceTransformerBackbone

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


# ---------------------------------------------------------------------------
# Backbone tests.
# ---------------------------------------------------------------------------


def test_forward_preserves_shapes_and_keys() -> None:
    """The backbone returns a WindowBatch with the same keys and value shapes as the input."""
    batch, metadata = make_window_batch_and_metadata()
    backbone = SingleSequenceTransformerBackbone(
        metadata, embed_dim=16, patch_size=4, num_layers=2, num_heads=2
    )
    output = backbone(batch)
    assert set(output.sources.keys()) == set(batch.sources.keys())
    for key, source in batch.sources.items():
        assert output.sources[key].values.shape == source.values.shape


def test_gradients_flow_through_encoder_blocks_decoder() -> None:
    """Backprop from the output reaches the encoder, every transformer block, and the decoder."""
    batch, metadata = make_window_batch_and_metadata()
    backbone = SingleSequenceTransformerBackbone(
        metadata, embed_dim=16, patch_size=4, num_layers=2, num_heads=2
    )
    output = backbone(batch)
    loss = torch.stack([source.values.sum() for source in output.sources.values()]).sum()
    loss.backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"
