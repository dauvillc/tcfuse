"""Hardware-aware resolution of the Lightning Trainer precision setting.

Shared by training and inference entry points so both pick the same mixed-precision
mode on a given machine.
"""

from __future__ import annotations

import torch


def resolve_precision(precision: str) -> str:
    """Resolve the sentinel ``"auto"`` precision to a concrete Lightning precision.

    Returns ``"bf16-mixed"`` when the current CUDA device supports bfloat16, otherwise
    ``"16-mixed"``. Any non-``"auto"`` value is returned unchanged so callers can still
    force a specific precision (e.g. ``32-true``).

    Args:
        precision: The configured ``trainer.precision`` value.

    Returns:
        The precision string to hand to ``pl.Trainer``.

    Raises:
        RuntimeError: If ``precision`` is ``"auto"`` but no CUDA device is available
            (we never run on CPU, so this signals a misconfigured environment).
    """
    # Only the sentinel triggers hardware detection; explicit values are honored as-is.
    if precision != "auto":
        return precision
    # We never train/infer on CPU; a missing CUDA device means a broken environment.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "precision='auto' requires a CUDA device, but none is available. "
            "Set trainer.precision explicitly if you really intend a CPU run."
        )
    # bf16-mixed on hardware that supports bfloat16 (A100/H100), else fall back to fp16.
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
