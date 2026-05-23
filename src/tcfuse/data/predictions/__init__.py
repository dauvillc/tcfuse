"""Prediction storage for TC-Fuse forecast outputs.

Public entry points:

- :class:`SamplePrediction` — per-window container for predicted/target Sources.
- :class:`PredictionRun` — run-level writer/reader (manifest + index + IBTrACS table).
- :func:`build_long_rows` — helper to build the tidy-long IBTrACS schema for a sample.
- :data:`IBTRACS_LONG_COLUMNS` and :func:`ibtracs_long_schema` — schema constants.
"""

from tcfuse.data.predictions.ibtracs import (
    IBTRACS_LONG_COLUMNS,
    build_long_rows,
    empty_long_frame,
    ibtracs_long_schema,
    long_to_pivot,
)
from tcfuse.data.predictions.run import PredictionRun
from tcfuse.data.predictions.sample import SamplePrediction

__all__ = [
    "IBTRACS_LONG_COLUMNS",
    "PredictionRun",
    "SamplePrediction",
    "build_long_rows",
    "empty_long_frame",
    "ibtracs_long_schema",
    "long_to_pivot",
]
