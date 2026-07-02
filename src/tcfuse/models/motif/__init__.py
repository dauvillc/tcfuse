"""MoTiF backbone: a diffusion-transformer-style encoder for multi-source geospatial data.

MoTiF blocks chain three layers — windowed cross-source attention, spatial self-attention,
and an MLP — with the standalone coordinate embedding injected as positional conditioning at
every layer. This package currently implements the cross-source attention layer and its
low-level spatiotemporal attention; the remaining layers, block, backbone, and decoder are
still to come.
"""

from tcfuse.models.motif.attention import SpatiotemporalAttention
from tcfuse.models.motif.cross_source_attention import MultiSourceCrossAttention

__all__ = [
    "MultiSourceCrossAttention",
    "SpatiotemporalAttention",
]
