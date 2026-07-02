"""MoTiF source un-embedding decoders: MotifEmbeddedSource → MotifDecodedSource.

The symmetric inverse of ``encoders_motif``: each decoder un-embeds one source's
value tokens back into raw values, ignoring the standalone coordinate tokens.
Batched via :class:`MotifMultiSourceDecoder`. Fully self-contained (its own
decoded containers and ``icnr`` helper), so the MoTiF stack can evolve without
touching the baseline :mod:`tcfuse.models.decoders` package.
"""

from tcfuse.models.decoders_motif.base import MotifSourceDecoder
from tcfuse.models.decoders_motif.decoded import MotifDecodedBatch, MotifDecodedSource
from tcfuse.models.decoders_motif.icnr import icnr
from tcfuse.models.decoders_motif.multisource import MotifMultiSourceDecoder
from tcfuse.models.decoders_motif.patch_unembed import (
    MotifFieldDecoder,
    MotifProfileDecoder,
    MotifScalarDecoder,
)

__all__ = [
    "MotifDecodedBatch",
    "MotifDecodedSource",
    "MotifFieldDecoder",
    "MotifMultiSourceDecoder",
    "MotifProfileDecoder",
    "MotifScalarDecoder",
    "MotifSourceDecoder",
    "icnr",
]
