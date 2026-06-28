"""Checkpoint resolution and ModelCheckpoint callback construction for training runs."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
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
    # Fall back to the lexicographically newest checkpoint by name; last.ckpt above
    # is the canonical resume point, this is only a best-effort fallback.
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    return ckpts[-1] if ckpts else None


def resolve_checkpoint_spec(spec: str, checkpoints_root: Path) -> Path:
    """Resolve a checkpoint_path config value to an actual .ckpt file.

    Args:
        spec: Either a path to a .ckpt file, or a bare run_id (the directory
            name under checkpoints_root, as assigned by scripts/train/train.py).
        checkpoints_root: paths.checkpoints, used to resolve a run_id.

    Returns:
        Path to the checkpoint file to load.
    """
    direct = Path(spec)
    if direct.is_file():
        return direct

    # Not a file: treat spec as a run_id and look for its best checkpoint.
    run_dir = checkpoints_root / spec / "checkpoints"
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint_path {spec!r} is neither an existing file nor a known "
            f"run_id under {checkpoints_root}"
        )
    best_ckpts = sorted(run_dir.glob("best-*.ckpt"))
    if not best_ckpts:
        raise FileNotFoundError(f"No best-*.ckpt checkpoint found in {run_dir}")
    if len(best_ckpts) > 1:
        # save_top_k=1 on the "best" ModelCheckpoint callback should guarantee
        # at most one file; more than one means that assumption broke.
        raise RuntimeError(
            f"Expected exactly one best-*.ckpt in {run_dir}, found {len(best_ckpts)}: {best_ckpts}"
        )
    return best_ckpts[0]


def build_checkpoint_callbacks(checkpoint_dir: Path, every_n_train_steps: int) -> list[Callback]:
    """Build the ModelCheckpoint callbacks used by the training Trainer.

    Args:
        checkpoint_dir: Absolute directory where checkpoints are written.
        every_n_train_steps: Number of training steps between periodic checkpoints.

    Returns:
        Two ModelCheckpoint callbacks: one that saves periodically by training
        step and keeps last.ckpt up to date (used to resume after preemption),
        and one that keeps only the checkpoint with the best validation loss.
    """
    periodic_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_train_steps=every_n_train_steps,
    )
    best_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{step}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    return [periodic_cb, best_cb]
