"""Lightning training callbacks for TC-Fuse training runs."""

from __future__ import annotations

import time

import lightning as pl


class StepProgressCallback(pl.Callback):
    """Prints training progress to stdout at regular intervals.

    PyTorch Lightning's default ``TQDMProgressBar`` relies on ``tqdm``, which
    silently disables itself when stdout is not a TTY — always the case in SLURM
    job log files.  This callback uses plain ``print()`` calls instead, which
    always appear in log files and work with ``jzlog``/``jzask`` monitoring tools.

    Each progress line includes elapsed time, an ETA to reach max_steps, and the
    throughput in steps/s so the user can judge how long the run will take.

    Only rank 0 prints in multi-GPU (DDP) runs to avoid duplicated lines.

    Args:
        log_every_n_steps: Print a progress line every this many global training steps.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        self.log_every_n_steps = log_every_n_steps
        # Run-level timing state; initialized in on_train_start. Throughput and ETA
        # are measured from training start (or the resumed step) rather than per-epoch,
        # since training is iteration-based and epoch boundaries are not meaningful.
        self._run_start_time: float = 0.0
        self._run_start_step: int = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record the wall-clock time and global step at the start of the run."""
        if not trainer.is_global_zero:
            return
        self._run_start_time = time.monotonic()
        # Anchor throughput at the current step (0 for a fresh run, the resumed
        # step after a requeue) so steps/s reflects only this process's progress.
        self._run_start_step = trainer.global_step

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Print a step progress line every log_every_n_steps global steps."""
        # Suppress output on all non-zero DDP ranks to avoid duplicated lines.
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return
        # int() guards against the rare Inf sentinel when max_steps is unset.
        max_steps = int(trainer.max_steps)

        # --- timing ---
        elapsed_s = time.monotonic() - self._run_start_time
        # Steps completed by this process since it started (handles resumes).
        steps_done = step - self._run_start_step
        steps_remaining = max(max_steps - step, 0)

        # Throughput and ETA — only meaningful once at least one step has elapsed.
        if steps_done > 0 and elapsed_s > 0:
            steps_per_sec = steps_done / elapsed_s
            eta_s = steps_remaining / steps_per_sec
            timing_str = (
                f"  {steps_per_sec:.2f} steps/s"
                f"  elapsed={_fmt_seconds(elapsed_s)}"
                f"  eta={_fmt_seconds(eta_s)}"
            )
        else:
            timing_str = f"  elapsed={_fmt_seconds(elapsed_s)}"

        # train/loss is logged on_step=True, on_epoch=False, so the key has no
        # _step suffix (Lightning only appends it when both on_step and on_epoch).
        loss_tensor = trainer.callback_metrics.get("train/loss")
        loss_str = f"  loss={loss_tensor.item():.4f}" if loss_tensor is not None else ""

        print(
            f"[step {step}/{max_steps}]{timing_str}{loss_str}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print a validation summary line after each step-triggered validation run."""
        # Skip the sanity-check pass and non-zero ranks.
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        step = trainer.global_step
        max_steps = int(trainer.max_steps)
        # Wall time since the run started (or resumed).
        elapsed_s = time.monotonic() - self._run_start_time
        # val/loss is logged on_epoch=True only, so the key is plain val/loss.
        val_loss = trainer.callback_metrics.get("val/loss")
        val_str = f"  val_loss={val_loss.item():.4f}" if val_loss is not None else ""
        print(
            f"[step {step}/{max_steps}  validation end"
            f"  elapsed={_fmt_seconds(elapsed_s)}]{val_str}",
            flush=True,
        )


def _fmt_seconds(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"
