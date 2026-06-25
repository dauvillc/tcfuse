#!/usr/bin/env python3
"""Training entry point — submits to SLURM via submitit or runs locally.

Usage::

    # Local run
    python scripts/train/train.py experiment=my_exp submitit=false

    # Jean-Zay GPU submission
    python scripts/train/train.py experiment=my_exp paths=jz setup=jz_gpu_v100
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import submitit
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Resolve project root so tcfuse imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcfuse.utils.checkpoint import build_checkpoint_callbacks, latest_checkpoint
from tcfuse.utils.precision import resolve_precision
from tcfuse.utils.submitit_utils import make_executor

# ── trainer builder ───────────────────────────────────────────────────────────


def _build_trainer(cfg: dict[str, Any], checkpoint_dir: Path) -> pl.Trainer:
    """Instantiate a Lightning Trainer from cfg["trainer"] and the checkpoint directory.

    Args:
        cfg: Fully resolved Hydra config dict.
        checkpoint_dir: Absolute directory where ModelCheckpoint writes checkpoints.

    Returns:
        Configured Trainer with a ModelCheckpoint callback attached.
    """
    trainer_cfg = dict(cfg["trainer"])
    # checkpoint_every_n_steps is a ModelCheckpoint kwarg, not a Trainer kwarg.
    checkpoint_every_n_steps: int = trainer_cfg.pop("checkpoint_every_n_steps")
    # Resolve the "auto" precision sentinel against the current hardware.
    trainer_cfg["precision"] = resolve_precision(trainer_cfg["precision"])
    ckpt_cbs = build_checkpoint_callbacks(checkpoint_dir, checkpoint_every_n_steps)
    # W&B cannot truly resume an offline run, so instead of one resumable run we
    # log each process launch (initial run, SLURM requeue, or manual restart) as a
    # distinct W&B "segment" run, all tied together by a shared group. run_id is
    # the stable logical-run key (the checkpoint dir name); segment_id is unique
    # per launch via a fresh timestamp. Unique ids make each offline-run folder a
    # distinct run, so `wandb sync` is idempotent (no --append, no double-counting).
    run_id = checkpoint_dir.parent.name
    experiment_name: str = cfg["name"]
    segment_id = f"{run_id}-{datetime.now():%m%d%H%M%S}"
    # Include experiment_name in the group so W&B groups are human-identifiable
    # without needing to look up what a bare timestamp belongs to.
    wandb_logger = instantiate(
        OmegaConf.create(cfg["logger"]),
        id=segment_id,
        group=f"{experiment_name}-{run_id}",
    )
    return pl.Trainer(
        **trainer_cfg,
        callbacks=ckpt_cbs,
        logger=wandb_logger,
        # Absolute path: stable across SLURM requeues where CWD may change.
        default_root_dir=str(checkpoint_dir.parent),
    )


# ── training task ─────────────────────────────────────────────────────────────


class TrainingTask(submitit.helpers.Checkpointable):
    """Encapsulates a full training run; resumes automatically after preemption.

    cfg is stored as a plain dict (not DictConfig) so that submitit can serialize
    this object via cloudpickle when requeueing a preempted SLURM job.
    checkpoint_dir is an absolute path so it remains valid in the requeued job,
    which may run in a different working directory than the submitting process.

    Args:
        cfg: Fully resolved Hydra config dict (plain dict, not DictConfig).
        checkpoint_dir: Absolute path for checkpoint files (stable across requeues).
    """

    def __init__(self, cfg: dict[str, Any], checkpoint_dir: Path) -> None:
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir

    def __call__(self) -> None:
        """Assemble all components and run trainer.fit(), resuming from last checkpoint."""
        cfg = self.cfg
        # Pick up the checkpoint saved before preemption, if any.
        ckpt_path = latest_checkpoint(self.checkpoint_dir)

        # Instantiate the data module and scan the assembled dataset root.
        dm = instantiate(OmegaConf.create(cfg["datamodule"]))
        # setup() must be called here, before the lightning module is instantiated,
        # because sources_metadata is loaded from disk during setup and must be
        # injected into the lightning module constructor below.
        dm.setup("fit")

        # Instantiate the task-specific lightning module.  The model backbone is
        # embedded in cfg["lightning_module"] as a nested Hydra config with
        # _partial_: true; BaseLightningModule.__init__ calls the resulting partial
        # with sources_metadata so the backbone can allocate per-source parameters.
        # Use _partial_=True so that sources_metadata and normalization_stats are
        # passed directly to the constructor — bypassing Hydra's OmegaConf merge,
        # which would otherwise wrap the MultisourceMetadata dataclass as a
        # structured DictConfig and break attribute access inside __init__.
        lm_factory = instantiate(OmegaConf.create(cfg["lightning_module"]), _partial_=True)
        lightning_module = lm_factory(
            sources_metadata=dm.sources_metadata,
            normalization_stats=dm.normalization_stats,
            experiment_name=cfg["name"],
        )

        trainer = _build_trainer(cfg, self.checkpoint_dir)
        # dm.setup("fit") already ran; Lightning 2.x will not call it again.
        trainer.fit(lightning_module, datamodule=dm, ckpt_path=ckpt_path)

    def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
        """Called by submitit on SIGTERM (SLURM preemption signal).

        Returns a new delayed submission of this same task. On the next
        invocation __call__ will find last.ckpt and resume from that step.
        """
        return submitit.helpers.DelayedSubmission(TrainingTask(self.cfg, self.checkpoint_dir))


# ── entry point ───────────────────────────────────────────────────────────────


@hydra.main(config_path="../../conf/", config_name="train", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))

    # Use Hydra's <date>/<time> run dir names purely as a unique, requeue-stable
    # run identifier (reformatted as MMDDhhmmss so the day is visible at a
    # glance), but root checkpoints under paths.checkpoints (configurable
    # storage tier) rather than under Hydra's own outputs/ directory.
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_started_at = datetime.strptime(
        f"{output_dir.parent.name} {output_dir.name}", "%Y-%m-%d %H-%M-%S"
    )
    # run_id is the logical-run key: it groups all W&B segments and roots the
    # checkpoint dir. Default to the launch timestamp (a fresh run); honor an
    # explicit run_id=<existing id> override to resume that run's checkpoints.
    run_id = cfg.get("run_id") or run_started_at.strftime("%m%d%H%M%S")
    run_id = str(run_id)
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / run_id / "checkpoints"

    task = TrainingTask(cfg, checkpoint_dir)

    # Local execution: run the task directly in the current process.
    if not cfg.get("submitit", True):
        task()
        return

    # SLURM execution: submit via submitit and block until the job finishes.
    executor = make_executor(cfg, "train")
    job = executor.submit(task)

    # Multi-GPU runs set slurm_ntasks_per_node = trainer.devices, so submitit launches
    # one task (process) per GPU and exposes them as sub-jobs (one result per DDP rank).
    # Total task count is tasks-per-node x nodes; default to 1 when the keys are absent.
    setup = cfg["setup"]
    n_total_tasks = int(setup.get("slurm_ntasks_per_node", 1)) * int(setup.get("slurm_nodes", 1))

    # result() (singular) asserts the job has no sub-jobs and would crash on any DDP
    # run; results() (plural) blocks on every rank and re-raises if any rank failed.
    # Both calls block until completion and propagate job-side exceptions to the launcher.
    if n_total_tasks > 1:
        # Multi-task DDP job: collect one result per rank.
        job.results()
    else:
        # Single-task job: collect the lone result.
        job.result()


if __name__ == "__main__":
    main()
