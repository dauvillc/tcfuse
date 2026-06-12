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
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import submitit
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

# Resolve project root so tcfuse imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcfuse.utils.submitit_utils import make_executor

# ── checkpoint utilities ──────────────────────────────────────────────────────


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recent checkpoint in checkpoint_dir, or None if absent.

    Args:
        checkpoint_dir: Directory written by ModelCheckpoint.

    Returns:
        Path to last.ckpt if it exists, else the lexicographically newest .ckpt,
        else None (triggers a fresh training run).
    """
    # last.ckpt is Lightning's canonical "resume here" file.
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    # Fall back to the newest checkpoint by name (epoch/step suffix sorts correctly).
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    return ckpts[-1] if ckpts else None


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
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_train_steps=checkpoint_every_n_steps,
    )
    return pl.Trainer(
        **trainer_cfg,
        callbacks=[ckpt_cb],
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
        ckpt_path = _latest_checkpoint(self.checkpoint_dir)

        # Instantiate the data module and scan the assembled dataset root.
        dm = instantiate(OmegaConf.create(cfg["datamodule"]))
        # setup() must be called here, before the lightning module is instantiated,
        # because sources_metadata is loaded from disk during setup and must be
        # injected into the lightning module constructor below.
        dm.setup("fit")

        # Instantiate the model backbone (experiment config provides _target_).
        model = instantiate(OmegaConf.create(cfg["model"]))

        # Instantiate the task-specific lightning module and inject the two values
        # absent from the YAML: model (instantiated above) and sources_metadata
        # (loaded from disk by dm.setup()).
        lightning_module = instantiate(
            OmegaConf.create(cfg["lightning_module"]),
            model=model,
            sources_metadata=dm.sources_metadata,
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
    # Capture the absolute Hydra output dir before converting cfg to a plain dict.
    # HydraConfig is only available inside @hydra.main scope. The output dir is
    # stored as an absolute path inside TrainingTask so SLURM requeues (different
    # CWD) still point to the same checkpoint directory.
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))
    task = TrainingTask(cfg, checkpoint_dir)

    # Local execution: run the task directly in the current process.
    if not cfg.get("submitit", True):
        task()
        return

    # SLURM execution: submit via submitit and block until the job finishes.
    executor = make_executor(cfg, "train")
    job = executor.submit(task)
    job.result()


if __name__ == "__main__":
    main()
