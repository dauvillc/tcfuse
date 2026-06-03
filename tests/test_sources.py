"""Unit tests for the base Source abstraction.

All tests use synthetic numpy arrays — no real data required.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from tcfuse.data.sources import Source, SourceKind, TorchSource

# ---------------------------------------------------------------------------
# Shared test timestamp
# ---------------------------------------------------------------------------

_TIME_UTC = cast(pd.Timestamp, pd.Timestamp("2020-08-01T12:00:00"))

# ---------------------------------------------------------------------------
# Helpers: synthetic numpy Source factories
# ---------------------------------------------------------------------------


def make_scalar_source(C: int = 3, source_name: str = "buoy") -> Source:
    """Create a minimal valid 0D (scalar) Source with numpy arrays."""
    values = np.random.randn(C).astype(np.float32)  # (C,)
    return Source(
        kind=SourceKind.SCALAR,
        values=values,
        coords=np.array([25.0, -80.0], dtype=np.float64),  # [lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=np.isfinite(values),
        time_utc=_TIME_UTC,
    )


def make_profile_source(L: int = 10, C: int = 5, source_name: str = "dropsonde") -> Source:
    """Create a minimal valid 1D (profile) Source with L levels and C channels."""
    values = np.random.randn(L, C).astype(np.float32)  # (L, C)
    return Source(
        kind=SourceKind.PROFILE,
        values=values,
        coords=np.random.randn(L, 3).astype(np.float64),  # (L, 3): [lat, lon, alt]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=np.isfinite(values),
        time_utc=_TIME_UTC,
    )


def make_field_source(H: int = 8, W: int = 8, C: int = 2, source_name: str = "pmw_ssmi") -> Source:
    """Create a minimal valid 2D (field/image) Source."""
    values = np.random.randn(H, W, C).astype(np.float32)  # (H, W, C)
    return Source(
        kind=SourceKind.FIELD,
        values=values,
        coords=np.random.randn(H, W, 2).astype(np.float32),  # (H, W, 2): [lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=np.isfinite(values),
        time_utc=_TIME_UTC,
    )


# ---------------------------------------------------------------------------
# Helpers: synthetic TorchSource factories (always batched)
# ---------------------------------------------------------------------------


def make_batched_scalar_source(B: int = 4, C: int = 3, source_name: str = "buoy") -> TorchSource:
    """Create a minimal valid batched 0D (scalar) TorchSource."""
    values = torch.randn(B, C)  # (B, C)
    return TorchSource(
        kind=SourceKind.SCALAR,
        values=values,
        coords=torch.randn(B, 2),  # (B, 2): [lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


def make_batched_profile_source(
    B: int = 3, L: int = 10, C: int = 5, source_name: str = "dropsonde"
) -> TorchSource:
    """Create a minimal valid batched 1D (profile) TorchSource."""
    values = torch.randn(B, L, C)  # (B, L, C)
    return TorchSource(
        kind=SourceKind.PROFILE,
        values=values,
        coords=torch.randn(B, L, 3),  # (B, L, 3): [lat, lon, alt]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


def make_batched_field_source(
    B: int = 2, H: int = 8, W: int = 8, C: int = 2, source_name: str = "pmw_ssmi"
) -> TorchSource:
    """Create a minimal valid batched 2D (field/image) TorchSource."""
    values = torch.randn(B, H, W, C)  # (B, H, W, C)
    return TorchSource(
        kind=SourceKind.FIELD,
        values=values,
        coords=torch.randn(B, H, W, 2),  # (B, H, W, 2): [lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
        mask=torch.isfinite(values),
        time=torch.rand(B, 2),
    )


# ---------------------------------------------------------------------------
# Construction and shape validation — Source
# ---------------------------------------------------------------------------


class TestSourceConstruction:
    def test_scalar_valid(self) -> None:
        src = make_scalar_source()
        assert src.kind is SourceKind.SCALAR
        assert src.mask.shape == src.values.shape

    def test_profile_valid(self) -> None:
        src = make_profile_source()
        assert src.kind is SourceKind.PROFILE
        assert src.mask.shape == src.values.shape

    def test_field_valid(self) -> None:
        src = make_field_source()
        assert src.kind is SourceKind.FIELD
        assert src.mask.shape == src.values.shape

    def test_time_utc_preserved(self) -> None:
        src = make_scalar_source()
        assert src.time_utc == _TIME_UTC

    def test_scalar_wrong_values_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.SCALAR,
                values=np.zeros((3, 3), dtype=np.float32),  # should be 1-D
                coords=np.zeros(2),
                source_name="bad",
                channels=["a", "b", "c"],
                mask=np.ones((3, 3), dtype=bool),
                time_utc=_TIME_UTC,
            )

    def test_profile_wrong_coords_shape_raises(self) -> None:
        # PROFILE coords should be (L, 3) = [lat, lon, alt]; passing (L, 2) raises.
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.PROFILE,
                values=np.zeros((10, 5), dtype=np.float32),
                coords=np.zeros((10, 2)),  # should be (L, 3)
                source_name="bad",
                channels=[f"ch{i}" for i in range(5)],
                mask=np.ones((10, 5), dtype=bool),
                time_utc=_TIME_UTC,
            )

    def test_field_wrong_coords_shape_raises(self) -> None:
        # FIELD coords should be (H, W, 2) = [lat, lon]; passing (H, W, 3) raises.
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.FIELD,
                values=np.zeros((8, 8, 2), dtype=np.float32),
                coords=np.zeros((8, 8, 3)),  # should be (H, W, 2)
                source_name="bad",
                channels=["ch0", "ch1"],
                mask=np.ones((8, 8, 2), dtype=bool),
                time_utc=_TIME_UTC,
            )


# ---------------------------------------------------------------------------
# Construction and shape validation — TorchSource
# ---------------------------------------------------------------------------


class TestTorchSourceConstruction:
    def test_batched_scalar_valid(self) -> None:
        src = make_batched_scalar_source()
        assert src.kind is SourceKind.SCALAR
        assert src.mask.shape == src.values.shape
        assert src.time.shape == (src.batch_size, 2)

    def test_batched_profile_valid(self) -> None:
        src = make_batched_profile_source()
        assert src.kind is SourceKind.PROFILE
        assert src.mask.shape == src.values.shape

    def test_batched_field_valid(self) -> None:
        src = make_batched_field_source()
        assert src.kind is SourceKind.FIELD
        assert src.mask.shape == src.values.shape

    def test_torch_source_wrong_time_shape_raises(self) -> None:
        B, C = 4, 3
        values = torch.randn(B, C)
        with pytest.raises(ValueError):
            TorchSource(
                kind=SourceKind.SCALAR,
                values=values,
                coords=torch.randn(B, 2),
                source_name="bad",
                channels=[f"ch{i}" for i in range(C)],
                mask=torch.isfinite(values),
                time=torch.rand(B, 3),  # should be (B, 2)
            )


# ---------------------------------------------------------------------------
# n_tokens
# ---------------------------------------------------------------------------


class TestNTokens:
    def test_scalar_n_tokens(self) -> None:
        assert make_scalar_source().n_tokens == 1

    def test_profile_n_tokens(self) -> None:
        L = 12
        assert make_profile_source(L=L).n_tokens == L

    def test_field_n_tokens(self) -> None:
        H, W = 6, 10
        assert make_field_source(H=H, W=W).n_tokens == H * W

    def test_batched_profile_n_tokens_ignores_batch_dim(self) -> None:
        L = 12
        assert make_batched_profile_source(B=5, L=L).n_tokens == L

    def test_batched_field_n_tokens_ignores_batch_dim(self) -> None:
        H, W = 6, 10
        assert make_batched_field_source(B=4, H=H, W=W).n_tokens == H * W


class TestBatchSize:
    def test_batch_size_for_torch_source(self) -> None:
        src = make_batched_field_source(B=7)
        assert src.batch_size == 7


# ---------------------------------------------------------------------------
# NaN / missing value handling
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_scalar_nan_values_preserved(self) -> None:
        src = make_scalar_source(C=4)
        src.values[1] = float("nan")
        assert np.isnan(src.values).any()

    def test_scalar_nan_mask_is_explicit_per_channel(self) -> None:
        values = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        mask = np.isfinite(values)
        src = Source(
            kind=SourceKind.SCALAR,
            values=values,
            coords=np.array([25.0, -80.0], dtype=np.float64),
            source_name="best_track",
            channels=["a", "b", "c"],
            mask=mask,
            time_utc=_TIME_UTC,
        )
        assert src.mask.tolist() == [True, False, True]

    def test_profile_with_mask(self) -> None:
        L, C = 10, 5
        mask = np.ones((L, C), dtype=bool)
        mask[3, 2] = False  # channel 2 at level 3 is missing
        src = Source(
            kind=SourceKind.PROFILE,
            values=np.random.randn(L, C).astype(np.float32),
            coords=np.random.randn(L, 3).astype(np.float64),
            source_name="dropsonde",
            channels=[f"ch{i}" for i in range(C)],
            mask=mask,
            time_utc=_TIME_UTC,
        )
        assert not src.mask[3, 2]

    def test_mask_shape_must_match_values_shape(self) -> None:
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.FIELD,
                values=np.zeros((8, 8, 2), dtype=np.float32),
                coords=np.zeros((8, 8, 2)),
                source_name="pmw",
                channels=["a", "b"],
                mask=np.ones((8, 8), dtype=bool),  # wrong shape
                time_utc=_TIME_UTC,
            )


# ---------------------------------------------------------------------------
# Device transfer — TorchSource only
# ---------------------------------------------------------------------------


class TestDeviceTransfer:
    def test_torch_source_to_cpu(self) -> None:
        src = make_batched_scalar_source()
        moved = src.to("cpu")
        assert moved.values.device.type == "cpu"
        assert moved.coords.device.type == "cpu"
        assert moved.time.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_source_to_cuda(self) -> None:
        src = make_batched_scalar_source()
        moved = src.to("cuda")
        assert moved.values.is_cuda
        assert moved.time.is_cuda
