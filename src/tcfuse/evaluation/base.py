"""Evaluation plugin base class.

An :class:`Evaluation` is one self-contained unit of offline analysis run over a
saved :class:`~tcfuse.data.predictions.run.PredictionRun` — for example the core
quantitative metrics, a power-spectrum comparison, or a set of diagnostic
figures.  The evaluation entry point (``scripts/evaluation/evaluate.py``)
instantiates the enabled plugins from the ``conf/evaluation/`` config group,
creates one output subfolder per plugin, and calls :meth:`Evaluation.run`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tcfuse.data.predictions.run import PredictionRun


class Evaluation(ABC):
    """Base class for an evaluation plugin.

    Each plugin lives in its own module under :mod:`tcfuse.evaluation`, is
    enabled through the ``conf/evaluation/`` Hydra config group, and writes all
    of its results under the ``output_dir`` it is handed.

    The contract is deliberately minimal: a plugin receives the **whole**
    :class:`~tcfuse.data.predictions.run.PredictionRun` of structured
    :class:`~tcfuse.data.sources.source.Source` objects and imposes no shape on
    the data.  How the data is consumed is entirely up to the plugin — point-wise
    metrics flatten everything to ``(N, C)`` over valid positions (see
    :func:`tcfuse.evaluation.flatten.flatten_valid`), whereas a spatial metric
    such as a radial power spectrum keeps the 2-D ``FIELD`` structure intact and
    reads ``source.values`` / ``source.mask`` directly.  Flattening is therefore
    a plugin choice, never part of this base contract.
    """

    # Subfolder name for this plugin's outputs (also its config key). Concrete
    # plugins override this with a short, filesystem-safe identifier.
    name: str

    @abstractmethod
    def run(self, run: PredictionRun, output_dir: Path) -> None:
        """Evaluate ``run`` and write all results under ``output_dir``.

        Args:
            run: The opened prediction run to evaluate. Read it via
                ``run.iter_samples()`` (streaming) and ``run.manifest`` /
                ``run.index``; values are in physical units.
            output_dir: This plugin's own results directory (already created by
                the caller). The plugin must write everything it produces here
                and never reach outside it.
        """
        ...
