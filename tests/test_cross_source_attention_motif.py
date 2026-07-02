"""Unit tests for the MoTiF windowed cross-source attention layer (synthetic tensors only)."""

from __future__ import annotations

import torch

from tcfuse.models.motif.cross_source_attention import MultiSourceCrossAttention

# Layer dims chosen so num_heads divides inner_dim, coord_inner_dim, and the stacked
# value width (window_size^2 * value_inner_dim = 9 * 16 = 144).
DIM, COORD_DIM = 16, 8
INNER, COORD_INNER = 16, 8
NUM_HEADS, WINDOW = 2, 3


def make_layer(*, mask_self_attention: bool = True) -> MultiSourceCrossAttention:
    """Build a cross-source attention layer with the shared test dims."""
    return MultiSourceCrossAttention(
        dim=DIM,
        inner_dim=INNER,
        window_size=WINDOW,
        num_heads=NUM_HEADS,
        coord_dim=COORD_DIM,
        coord_inner_dim=COORD_INNER,
        mask_self_attention=mask_self_attention,
    )


def make_field(B: int, Eh: int, Ew: int, D: int) -> torch.Tensor:
    """Synthetic FIELD token grid of shape (B, Eh, Ew, D)."""
    return torch.randn(B, Eh, Ew, D)


def make_two_sources(B: int = 2) -> tuple[dict, dict]:
    """Two FIELD sources with different, non-window-divisible token grids."""
    values = {
        ("a", 0): make_field(B, 8, 12, DIM),
        ("b", 0): make_field(B, 7, 5, DIM),
    }
    coords = {
        ("a", 0): make_field(B, 8, 12, COORD_DIM),
        ("b", 0): make_field(B, 7, 5, COORD_DIM),
    }
    return values, coords


def test_shape_round_trip() -> None:
    """Each source's output value grid matches its input shape and stays finite."""
    layer = make_layer()
    values, coords = make_two_sources()
    out = layer(values, coords)
    # Same keys back out.
    assert set(out) == set(values)
    for key in values:
        # Output value grid identical in shape to the input, and finite.
        assert out[key].shape == values[key].shape
        assert torch.isfinite(out[key]).all()


def test_padding_and_crop_non_divisible_grid() -> None:
    """A grid that is not a multiple of window_size is padded internally and cropped back."""
    # Single source with mask_self_attention off (a lone source has no cross-source keys).
    layer = make_layer(mask_self_attention=False)
    B, Eh, Ew = 2, 7, 5  # neither dim divisible by WINDOW=3
    values = {("only", 0): make_field(B, Eh, Ew, DIM)}
    coords = {("only", 0): make_field(B, Eh, Ew, COORD_DIM)}
    out = layer(values, coords)
    # Cropped exactly back to the original (Eh, Ew) grid.
    assert out[("only", 0)].shape == (B, Eh, Ew, DIM)
    assert torch.isfinite(out[("only", 0)]).all()


def test_self_attention_mask_changes_output() -> None:
    """Masking same-source blocks changes the output (and stays finite) vs. dense attention."""
    layer = make_layer()
    layer.eval()
    values, coords = make_two_sources()
    # Masked (cross-source only) pass.
    masked = layer(values, coords)
    # Same weights, dense pass: toggle the flag so only the mask differs.
    layer.mask_self_attention = False
    dense = layer(values, coords)
    for key in values:
        assert torch.isfinite(masked[key]).all()
        # The mask must actually change the attention output.
        assert not torch.allclose(masked[key], dense[key])


def test_deterministic_in_eval() -> None:
    """Two eval passes over identical inputs give identical outputs."""
    layer = make_layer()
    layer.eval()
    values, coords = make_two_sources()
    first = layer(values, coords)
    second = layer(values, coords)
    for key in values:
        assert torch.equal(first[key], second[key])


def test_backward_produces_finite_grads() -> None:
    """A backward pass yields finite gradients, including on the coordinate-bias weight alpha."""
    layer = make_layer()
    layer.train()
    values, coords = make_two_sources()
    out = layer(values, coords)
    # Simple scalar objective over all source outputs.
    loss = torch.stack([t.sum() for t in out.values()]).sum()
    loss.backward()
    # The learnable coordinate-bias weight must receive a finite gradient.
    assert layer.attention.alpha.grad is not None
    assert torch.isfinite(layer.attention.alpha.grad).all()
    # Every trainable parameter that received a gradient must be finite.
    for param in layer.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
