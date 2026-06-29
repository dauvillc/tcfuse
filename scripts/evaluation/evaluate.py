#!/usr/bin/env python3
"""Evaluation entry point — run evaluation plugins over a saved PredictionRun.

Locates the predictions written by ``scripts/inference/infer.py`` for a given
``(run_id, experiment_name)``, opens the
:class:`~tcfuse.data.predictions.run.PredictionRun`, and dispatches it to the set
of enabled :class:`~tcfuse.evaluation.base.Evaluation` plugins (selected via the
``conf/evaluation/`` config group). Each plugin writes its results into its own
subfolder under ``paths.results/<run_id>/<experiment_name>/``.

Usage::

    python scripts/evaluation/evaluate.py \
        run_id=0627015132 experiment_name=pmw-gmi-dummy

``run_id`` and ``experiment_name`` together identify the prediction run; they
must match the ``run_id`` and experiment ``name`` used when running inference.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Resolve project root so tcfuse imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcfuse.data.predictions.run import PredictionRun
from tcfuse.evaluation.base import Evaluation

# On-disk name of the evaluation-level manifest written next to the plugin folders.
_MANIFEST_FILENAME = "manifest.yaml"


@hydra.main(config_path="../../conf/", config_name="evaluation", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))
    run_id = str(cfg["run_id"])
    experiment_name = str(cfg["experiment_name"])

    # Locate the prediction run on disk (written by infer.py under the same key).
    pred_dir = Path(cfg["paths"]["predictions"]) / run_id / experiment_name
    if not (pred_dir / "manifest.yaml").exists():
        raise FileNotFoundError(
            f"No prediction run found at {pred_dir}. Run inference first, e.g.:\n"
            f"    python scripts/inference/infer.py experiment=<exp> "
            f"run_id={run_id} split=test"
        )
    run = PredictionRun.open(pred_dir)

    # Results root for this (run_id, experiment_name), mirroring the predictions
    # layout. Each plugin gets its own subfolder beneath it.
    results_dir = Path(cfg["paths"]["results"]) / run_id / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run each enabled plugin in turn, isolating its outputs to its own subfolder.
    # cfg["evaluations"] is a mapping {plugin_key: {_target_, ...}} composed from
    # the conf/evaluation/ config group.
    evaluation_names: list[str] = []
    for plugin_key, plugin_cfg in cfg["evaluations"].items():
        # Instantiate the plugin from its _target_ + params.
        evaluation = cast(Evaluation, instantiate(OmegaConf.create(plugin_cfg)))
        # Each plugin owns the subfolder named after itself.
        plugin_dir = results_dir / evaluation.name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running evaluation '{plugin_key}' -> {plugin_dir}")
        evaluation.run(run, plugin_dir)
        evaluation_names.append(evaluation.name)

    # Record what ran so the results dir is self-describing (mirrors the
    # prediction run manifest style).
    manifest = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "predictions_dir": str(pred_dir),
        "evaluations": evaluation_names,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    with open(results_dir / _MANIFEST_FILENAME, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Wrote {len(evaluation_names)} evaluation(s) to {results_dir}")


if __name__ == "__main__":
    main()
