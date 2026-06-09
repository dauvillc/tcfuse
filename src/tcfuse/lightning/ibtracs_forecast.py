"""Lightning module for IBTrACS forecasting from multi-source assimilation windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.lightning.lr_scheduler import CosineAnnealingWarmupRestarts

# Target best-track source and channels (used by task heads and validation plots).
# See tcfuse.data.ibtracs.IBTRACS_SOURCE_NAME and IBTRACS_CHANNELS.


class IBTrACSForecastLightningModule(lightning.LightningModule):
    """Train and infer IBTrACS lead-time forecasts from multi-source windows.

    Wraps an injected :class:`nn.Module` that maps a :class:`~tcfuse.data.collate.WindowBatch`
    to predictions. Optimizer and LR scheduler hyperparameters are supplied as kwargs dicts
    (typically from Hydra ``optimizer`` and ``lr_scheduler`` config groups).

    Args:
        model: Backbone plus task heads; must accept :class:`WindowBatch` in ``forward``.
        sources_metadata: Static descriptors for sources present in training samples.
        adamw_kwargs: Keyword arguments for :class:`torch.optim.AdamW` (excluding ``params``).
        lr_scheduler_kwargs: Keyword arguments for :class:`CosineAnnealingWarmupRestarts`.
        validation_dir: Directory where validation figures are written each epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        sources_metadata: MultisourceMetadata,
        adamw_kwargs: dict[str, Any],
        lr_scheduler_kwargs: dict[str, Any],
        validation_dir: str | Path,
    ) -> None:
        super().__init__()
        # Register the Hydra-instantiated model as a submodule for parameter tracking.
        self.model = model
        # Snapshot metadata so later mutations to the injected object cannot leak in.
        self._sources_metadata = MultisourceMetadata.from_dict(sources_metadata.to_dict())
        self._adamw_kwargs = dict(adamw_kwargs)
        self._lr_scheduler_kwargs = dict(lr_scheduler_kwargs)
        self._validation_dir = Path(validation_dir)
        # Do not serialize the full model tree or metadata into checkpoints.
        self.save_hyperparameters(ignore=["model", "sources_metadata"])

    @property
    def sources_metadata(self) -> MultisourceMetadata:
        """Static source descriptors (channels, shape, kind) for this run."""
        return MultisourceMetadata.from_dict(self._sources_metadata.to_dict())

    def forward(self, batch: WindowBatch) -> object:
        """Run the injected model on a collated window batch."""
        return self.model(batch)

    def _shared_step(self, batch: WindowBatch, stage: str) -> torch.Tensor:
        """Forward pass and loss for train / validation."""
        # Run the multi-source model on this batch.
        _outputs = self(batch)
        # Loss computation is task-specific — implemented when the model head exists.
        raise NotImplementedError(f"Loss for stage {stage!r} is not implemented yet.")

    def training_step(self, batch: WindowBatch, batch_idx: int) -> torch.Tensor:
        """One training optimizer step."""
        loss = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: WindowBatch, batch_idx: int) -> torch.Tensor:
        """One validation forward pass."""
        loss = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: WindowBatch, batch_idx: int) -> dict[str, Any]:
        """Inference hook; returns per-batch metadata without writing to disk."""
        # Forward predictions for export into SamplePrediction / PredictionRun later.
        _outputs = self(batch)
        return {
            "sample_ids": batch.sample_ids,
            "sids": batch.sids,
            "init_times_utc": batch.window_ref_times_utc,
            "outputs": _outputs,
        }

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """AdamW optimizer with cosine warmup-restarts LR schedule (per-step)."""
        # Map parameter names to tensors for decay vs no-decay grouping.
        params = dict(self.named_parameters())
        # Decay 2D+ tensors only (weight matrices, conv kernels).
        # Skip 1D tensors (norms, biases, embeddings if 1D).
        decay_params = {k for k, v in params.items() if v.ndim >= 2 and "norm" not in k}
        # Weight decay applies only to the decay group; other kwargs go to AdamW.
        adamw_kwargs = dict(self._adamw_kwargs)
        weight_decay = adamw_kwargs.pop("weight_decay", 0.0)
        # Matrices and conv kernels get L2 decay; biases, norms, and 1D params do not.
        param_groups: list[dict[str, Any]] = [
            {
                "params": [params[k] for k in decay_params],
                "weight_decay": weight_decay,
            },
            {
                "params": [params[k] for k in params if k not in decay_params],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **self._lr_scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_validation_epoch_end(self) -> None:
        """Write validation figures on the primary process after each val epoch."""
        trainer = self.trainer
        if trainer is None:
            return
        # Only rank 0 writes figures; skip the initial sanity-check pass.
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self._save_validation_figures()

    def _save_validation_figures(self) -> None:
        """Persist validation diagnostic plots under validation_dir."""
        self._validation_dir.mkdir(parents=True, exist_ok=True)
        # TODO: call tcfuse.data.visualization.training once implemented.
        epoch = self.current_epoch
        _figure_path = self._validation_dir / f"{epoch:04d}_val_summary.svg"
