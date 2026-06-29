"""Offline evaluation of saved prediction runs.

This package holds the plugin-based evaluation pipeline: a set of independent
:class:`~tcfuse.evaluation.base.Evaluation` plugins that each consume a
:class:`~tcfuse.data.predictions.run.PredictionRun` and write their results into
their own subfolder.  Plugins are enabled/disabled and instantiated through the
``conf/evaluation/`` Hydra config group and driven by
``scripts/evaluation/evaluate.py``.
"""

from __future__ import annotations

from tcfuse.evaluation.base import Evaluation

__all__ = ["Evaluation"]
