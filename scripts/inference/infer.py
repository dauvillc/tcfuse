#!/usr/bin/env python3
"""Inference entry point — run a trained checkpoint over a split and save predictions.

Loads a checkpoint into the same Lightning module used for training, runs it over
the chosen split with ``trainer.predict`` (the task applies its own target
masking), and writes a :class:`~tcfuse.data.predictions.run.PredictionRun`
(per-window HDF5 + index + manifest) under ``paths.predictions`` via a
:class:`~tcfuse.lightning.prediction_writer.PredictionWriter`. Optionally computes
evaluation metrics.

Usage::

    python scripts/inference/infer.py experiment=pmw_gmi_reconstruction_dummy \
        run_id=0627015132 split=test

``run_id`` is the training run identifier (the directory name under
``paths.checkpoints``); inference resolves that run's best checkpoint under
``paths.checkpoints/<run_id>/checkpoints/`` automatically.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Resolve project root so tcfuse imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcfuse.data.collate import WindowBatch, collate_window_samples
from tcfuse.data.dataset import TCWindowDataset
from tcfuse.data.predictions.run import PredictionRun
from tcfuse.lightning.prediction_writer import PredictionWriter
from tcfuse.utils.checkpoint import best_checkpoint


@hydra.main(config_path="../../conf/", config_name="inference", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))
    split = str(cfg["split"])

    # Instantiate the data module and load sources_metadata + normalization stats.
    # setup() reads the metadata YAML the module constructor needs below.
    dm = instantiate(OmegaConf.create(cfg["datamodule"]))
    dm.setup("predict")

    # Resolve the run's best checkpoint from its run_id and load it first: the
    # checkpoint carries the full training config, from which we rebuild the model.
    checkpoint_path = best_checkpoint(str(cfg["run_id"]), Path(cfg["paths"]["checkpoints"]))
    # weights_only=False: our own training checkpoints embed the resolved Hydra
    # config (and OmegaConf hparams), which the (newer-PyTorch default) safe
    # unpickler rejects. The file is a trusted local artifact, so full unpickling
    # is fine here.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild the task-specific lightning module exactly as in training. Its
    # architecture (module class + nested model config) is read from the config
    # embedded in the checkpoint at train time, so it always matches the weights
    # and need not be re-specified here. sources_metadata / normalization_stats are
    # passed directly (they are deliberately omitted from the checkpoint hparams);
    # passing the *inference* datamodule's metadata preserves the train != inference
    # source-subset behavior (the model allocates encoders/decoders per locally
    # available source, handled together with strict=False below).
    embedded_cfg = checkpoint.get("hydra_cfg")
    if embedded_cfg is not None:
        lm_cfg = embedded_cfg["lightning_module"]
    else:
        # Pre-embedding checkpoint: fall back to the architecture specified in the
        # inference config, which must then match the trained weights manually.
        print(
            "Checkpoint has no embedded config (predates config embedding); "
            "rebuilding the model from the inference config's lightning_module. "
            "Ensure its architecture matches the trained weights."
        )
        lm_cfg = cfg["lightning_module"]
    lm_factory = instantiate(OmegaConf.create(lm_cfg), _partial_=True)
    module = lm_factory(
        sources_metadata=dm.sources_metadata,
        normalization_stats=dm.normalization_stats,
        experiment_name=cfg["name"],
    )
    # strict=False: a checkpoint may have been trained on a machine where more
    # sources were preprocessed than are available here. The model builds one
    # encoder/decoder per locally-available source, so the loaded module is a
    # subset of the checkpoint. We only tolerate *unexpected* keys (extra,
    # unused source weights); any *missing* key is a real architecture mismatch
    # and must abort.
    incompatible = module.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing weights the model expects "
            f"(architecture mismatch): {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        # Extra weights for sources absent from the local dataset — fine to drop.
        print(
            f"Ignoring {len(incompatible.unexpected_keys)} checkpoint weights "
            "for sources not present locally."
        )

    # Build the dataset + dataloader for the requested split. shuffle=False keeps
    # the run order reproducible and the writer's index fallback valid.
    dataset = TCWindowDataset(
        Path(cfg["paths"]["preprocessed_data"]),
        str(cfg["windows_setup"]["name"]),
        split,  # type: ignore[arg-type]
    )
    batch_size = int(cfg["dataloader"]["batch_size"])
    loader = cast(
        "DataLoader[WindowBatch]",
        DataLoader(
            dataset,
            collate_fn=collate_window_samples,
            shuffle=False,
            batch_size=batch_size,
            num_workers=int(cfg["dataloader"]["num_workers"]),
        ),
    )

    # Open the output run; the writer appends to it as each batch is predicted.
    # The output dir is keyed by the training run_id and the (inference)
    # experiment name.
    run_dir = Path(cfg["paths"]["predictions"]) / str(cfg["run_id"]) / str(cfg["name"])
    run = PredictionRun.create(
        run_dir,
        manifest={
            "run_id": str(cfg["run_id"]),
            "checkpoint_path": str(checkpoint_path),
            "experiment_name": cfg["name"],
            "windows_setup_name": str(cfg["windows_setup"]["name"]),
            "split": split,
            "units": "physical",
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    writer = PredictionWriter(run, dataset)

    # Optional cap for smoke tests: translate a sample limit into a batch limit
    # (1.0 = no limit). Lightning then stops predict after that many batches.
    limit = cfg["limit_samples"]
    limit_predict_batches: float | int = (
        1.0 if limit is None else max(1, math.ceil(int(limit) / batch_size))
    )

    # Let Lightning own the loop, device placement, eval()/no_grad, and progress.
    # Inference is single-GPU by design: the PredictionWriter and PredictionRun
    # index are not DDP-safe (per-rank index buffers, unconditional finalize), so
    # devices is pinned to 1. See PredictionWriter's docstring for what DDP support
    # would require.
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        callbacks=[writer],
        limit_predict_batches=limit_predict_batches,
    )
    # predict_step (masked reconstruction) masks targets, runs the model, and
    # de-normalizes; the writer persists each batch, so we discard the return.
    trainer.predict(module, dataloaders=loader, return_predictions=False)

    # Flush the index + manifest, completing the run on disk.
    run.finalize()
    print(f"Wrote {run.manifest['num_samples']} sample predictions to {run_dir}")

    # Optionally evaluate the run and save a tidy metrics table next to it.
    if cfg["compute_metrics"]:
        metrics = run.compute_metrics()
        metrics_path = run_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        print(f"Wrote metrics for {len(metrics)} (source, channel, metric) rows to {metrics_path}")


if __name__ == "__main__":
    main()
