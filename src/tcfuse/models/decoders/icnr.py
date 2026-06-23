"""ICNR initialization for sub-pixel convolution weights.

Adapted from https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
(thanks to A03ki for the implementation).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def icnr(tensor: Tensor, initializer: Callable[[Tensor], Tensor], upscale_factor: int) -> Tensor:
    """Return an ICNR-initialized weight for a conv feeding a PixelShuffle(upscale_factor).

    ICNR (Aitken et al., 2017) makes every one of the ``upscale_factor**2``
    output-channel groups produced by the conv start identical, which avoids
    the checkerboard artifacts that plain random init causes in sub-pixel
    upsampling. Does not mutate ``tensor``; the caller is responsible for
    copying the result into the conv weight (e.g. ``conv.weight.data.copy_(...)``).

    Args:
        tensor: Conv weight to initialize from, shape ``(out_channels, *rest)``
            where ``out_channels`` must be divisible by ``upscale_factor**2``.
        initializer: In-place-style init function called as ``initializer(sub_kernel)``,
            e.g. ``nn.init.kaiming_normal_``.
        upscale_factor: PixelShuffle upscale factor used downstream.

    Returns:
        A new tensor with the same shape as ``tensor``, ICNR-initialized.
    """
    upscale_factor_squared = upscale_factor**2
    if tensor.shape[0] % upscale_factor_squared != 0:
        raise ValueError(
            f"tensor.shape[0]={tensor.shape[0]} must be divisible by "
            f"upscale_factor**2={upscale_factor_squared}"
        )
    # Initialize a single sub-kernel covering one output-channel group...
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel)
    # ...then tile it across all upscale_factor**2 groups along the output-channel axis.
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)
