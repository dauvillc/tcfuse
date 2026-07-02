"""MoTiF source embedding encoders: TorchSource → separate value and coordinate tensors.

Each source is embedded into two tensors sharing the same spatial/token dims but
possibly different embedding dims: a value-only token tensor (``Dv``) and a
standalone Fourier coordinate embedding (``Dc``) that the MoTiF backbone injects
as positional conditioning at every layer.
"""

from tcfuse.models.encoders_motif.base import MotifSourceEncoder
from tcfuse.models.encoders_motif.embedded import MotifEmbeddedBatch, MotifEmbeddedSource
from tcfuse.models.encoders_motif.multisource import MotifMultiSourceEncoder
from tcfuse.models.encoders_motif.patch_embed import (
    MotifFieldEncoder,
    MotifProfileEncoder,
    MotifScalarEncoder,
)
from tcfuse.models.encoders_motif.positional import CoordEmbedding, MotifCoordEncodingConfig

__all__ = [
    "CoordEmbedding",
    "MotifCoordEncodingConfig",
    "MotifEmbeddedBatch",
    "MotifEmbeddedSource",
    "MotifFieldEncoder",
    "MotifMultiSourceEncoder",
    "MotifProfileEncoder",
    "MotifScalarEncoder",
    "MotifSourceEncoder",
]
