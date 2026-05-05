"""Unit tests for the base Source abstraction.

All tests use synthetic tensors — no real data required.
"""

import pytest
import torch

from tcfuse.data.sources.base import Source, SourceKind

# ---------------------------------------------------------------------------
# Helpers: synthetic source factories
# ---------------------------------------------------------------------------


def make_scalar_source(C: int = 3, source_name: str = "buoy") -> Source:
    """Create a minimal valid 0D (scalar) source."""
    return Source(
        kind=SourceKind.SCALAR,
        values=torch.randn(C),  # (C,)
        coords=torch.tensor([0.0, 25.0, -80.0]),  # [time, lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
    )


def make_profile_source(L: int = 10, C: int = 5, source_name: str = "dropsonde") -> Source:
    """Create a minimal valid 1D (profile) source with L levels and C channels."""
    return Source(
        kind=SourceKind.PROFILE,
        values=torch.randn(L, C),  # (L, C)
        coords=torch.randn(L, 4),  # (L, 4): [time, lat, lon, alt]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
    )


def make_field_source(H: int = 8, W: int = 8, C: int = 2, source_name: str = "pmw_ssmi") -> Source:
    """Create a minimal valid 2D (field/image) source."""
    return Source(
        kind=SourceKind.FIELD,
        values=torch.randn(H, W, C),  # (H, W, C)
        coords=torch.randn(H, W, 3),  # (H, W, 3): [time, lat, lon]
        source_name=source_name,
        channels=[f"ch{i}" for i in range(C)],
    )


# ---------------------------------------------------------------------------
# Construction and shape validation
# ---------------------------------------------------------------------------


class TestSourceConstruction:
    def test_scalar_valid(self) -> None:
        src = make_scalar_source()
        assert src.kind is SourceKind.SCALAR

    def test_profile_valid(self) -> None:
        src = make_profile_source()
        assert src.kind is SourceKind.PROFILE

    def test_field_valid(self) -> None:
        src = make_field_source()
        assert src.kind is SourceKind.FIELD

    def test_scalar_wrong_values_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.SCALAR,
                values=torch.randn(3, 3),  # should be 1-D
                coords=torch.zeros(3),
                source_name="bad",
                channels=["a", "b", "c"],
            )

    def test_profile_wrong_coords_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.PROFILE,
                values=torch.randn(10, 5),
                coords=torch.randn(10, 3),  # should be (L, 4)
                source_name="bad",
                channels=[f"ch{i}" for i in range(5)],
            )

    def test_field_wrong_coords_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Source(
                kind=SourceKind.FIELD,
                values=torch.randn(8, 8, 2),
                coords=torch.randn(8, 8, 4),  # should be (H, W, 3)
                source_name="bad",
                channels=["ch0", "ch1"],
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


# ---------------------------------------------------------------------------
# NaN / missing value handling
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_scalar_nan_values_preserved(self) -> None:
        src = make_scalar_source(C=4)
        src.values[1] = float("nan")
        assert torch.isnan(src.values).any()

    def test_profile_with_mask(self) -> None:
        L, C = 10, 5
        mask = torch.ones(L, dtype=torch.bool)
        mask[3] = False  # level 3 is missing
        src = Source(
            kind=SourceKind.PROFILE,
            values=torch.randn(L, C),
            coords=torch.randn(L, 4),
            source_name="dropsonde",
            channels=[f"ch{i}" for i in range(C)],
            mask=mask,
        )
        assert src.mask is not None
        assert not src.mask[3]


# ---------------------------------------------------------------------------
# Device transfer
# ---------------------------------------------------------------------------


class TestDeviceTransfer:
    def test_scalar_to_cpu(self) -> None:
        src = make_scalar_source()
        moved = src.to("cpu")
        assert moved.values.device.type == "cpu"
        assert moved.coords.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scalar_to_cuda(self) -> None:
        src = make_scalar_source()
        moved = src.to("cuda")
        assert moved.values.is_cuda
