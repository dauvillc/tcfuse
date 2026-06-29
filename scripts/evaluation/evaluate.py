#!/usr/bin/env python3
"""Evaluation entry point — compare models over their saved PredictionRuns.

Locates the predictions written by ``scripts/inference/infer.py`` for every model
declared in the ``models`` config mapping (each a ``(run_id, experiment_name)``
pair), opens each :class:`~tcfuse.data.predictions.run.PredictionRun`, and
dispatches the whole mapping to the set of enabled
:class:`~tcfuse.evaluation.base.Evaluation` plugins (selected via the
``conf/evaluation/`` config group). Each plugin compares the models and writes
its results into its own subfolder under ``paths.results/<eval_name>/``.

Usage::

    python scripts/evaluation/evaluate.py eval_name=baseline-vs-fusion \
        +models.baseline.run_id=0627015132 \
        +models.baseline.experiment_name=pmw-gmi-dummy \
        +models.fusion.run_id=0628231045 \
        +models.fusion.experiment_name=pmw-gmi-fusion

Each model's ``run_id`` and ``experiment_name`` must match the ``run_id`` and
experiment ``name`` used when running inference for that model.
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
    eval_name = str(cfg["eval_name"])
    predictions_root = Path(cfg["paths"]["predictions"])

    # Open every model's prediction run, preserving the config declaration order
    # so plugins produce stable column / plot order. Each model is identified by
    # its own (run_id, experiment_name) pair under paths.predictions.
    runs: dict[str, PredictionRun] = {}
    model_manifest: dict[str, dict[str, str]] = {}
    for model_name, model_cfg in cfg["models"].items():
        run_id = str(model_cfg["run_id"])
        experiment_name = str(model_cfg["experiment_name"])
        # Locate this model's prediction run on disk (written by infer.py).
        pred_dir = predictions_root / run_id / experiment_name
        if not (pred_dir / "manifest.yaml").exists():
            raise FileNotFoundError(
                f"No prediction run found for model '{model_name}' at {pred_dir}. "
                f"Run inference first, e.g.:\n"
                f"    python scripts/inference/infer.py experiment=<exp> "
                f"run_id={run_id} split=test"
            )
        runs[model_name] = PredictionRun.open(pred_dir)
        # Record where this model's predictions came from for the manifest.
        model_manifest[model_name] = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "predictions_dir": str(pred_dir),
        }

    # Results root for this comparison, keyed by the user-chosen eval_name. Each
    # plugin gets its own subfolder beneath it.
    results_dir = Path(cfg["paths"]["results"]) / eval_name
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
        # Hand the plugin the whole mapping of models to compare.
        evaluation.run(runs, plugin_dir)
        evaluation_names.append(evaluation.name)

    # Record what ran so the results dir is self-describing (mirrors the
    # prediction run manifest style).
    manifest = {
        "eval_name": eval_name,
        "models": model_manifest,
        "evaluations": evaluation_names,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    with open(results_dir / _MANIFEST_FILENAME, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Wrote {len(evaluation_names)} evaluation(s) to {results_dir}")


if __name__ == "__main__":
    main()
