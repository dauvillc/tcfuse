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

    Each progress line includes elapsed time, an ETA for the current epoch, and
    the throughput in steps/s so the user can judge how long an epoch will take.

    Only rank 0 prints in multi-GPU (DDP) runs to avoid duplicated lines.

    Args:
        log_every_n_steps: Print a progress line every this many global training steps.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        self.log_every_n_steps = log_every_n_steps
        # Epoch-level timing state; reset in on_train_epoch_start.
        self._epoch_start_time: float = 0.0
        self._epoch_start_step: int = 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record the wall-clock time and global step at the start of each epoch."""
        if not trainer.is_global_zero:
            return
        self._epoch_start_time = time.monotonic()
        # global_step hasn't been incremented yet at epoch start.
        self._epoch_start_step = trainer.global_step

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
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        # --- timing ---
        elapsed_s = time.monotonic() - self._epoch_start_time
        # Steps completed within the current epoch (global_step was just incremented).
        steps_done_in_epoch = step - self._epoch_start_step
        # Total batches this epoch; int() guards against the rare Inf sentinel.
        total_steps_in_epoch = int(trainer.num_training_batches)
        steps_remaining = max(total_steps_in_epoch - steps_done_in_epoch, 0)

        # Throughput and ETA — only meaningful once at least one step has elapsed.
        if steps_done_in_epoch > 0 and elapsed_s > 0:
            steps_per_sec = steps_done_in_epoch / elapsed_s
            eta_s = steps_remaining / steps_per_sec
            timing_str = (
                f"  {steps_per_sec:.2f} steps/s"
                f"  elapsed={_fmt_seconds(elapsed_s)}"
                f"  eta={_fmt_seconds(eta_s)}"
            )
        else:
            timing_str = f"  elapsed={_fmt_seconds(elapsed_s)}"

        # Read the step-level train loss.  Lightning appends _step to the key
        # when a metric is logged with both on_step=True and on_epoch=True
        # (as in BaseLightningModule.training_step).
        loss_tensor = trainer.callback_metrics.get("train/loss_step")
        loss_str = f"  loss={loss_tensor.item():.4f}" if loss_tensor is not None else ""

        print(
            f"[epoch {epoch}/{max_epochs}"
            f"  step {steps_done_in_epoch}/{total_steps_in_epoch}]{timing_str}{loss_str}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print a validation summary line after each validation epoch."""
        # Skip the sanity-check pass (epoch index -1) and non-zero ranks.
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        # Total epoch wall time (validation is included in the elapsed window since
        # on_train_epoch_start fires before val in Lightning's epoch loop order).
        elapsed_s = time.monotonic() - self._epoch_start_time
        # val/loss is logged on_epoch=True only, so the key is plain val/loss.
        val_loss = trainer.callback_metrics.get("val/loss")
        val_str = f"  val_loss={val_loss.item():.4f}" if val_loss is not None else ""
        print(
            f"[epoch {epoch}/{max_epochs}  validation end"
            f"  epoch_time={_fmt_seconds(elapsed_s)}]{val_str}",
            flush=True,
        )


def _fmt_seconds(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"
