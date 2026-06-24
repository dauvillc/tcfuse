"""Base Lightning module for WindowBatch source-value transformation."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import lightning
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.lightning.lr_scheduler import CosineAnnealingWarmupRestarts


class BaseLightningModule(ABC, lightning.LightningModule):
    """Train and infer any source-transformation model over assimilation windows.

    Wraps an injected :class:`nn.Module` that maps a :class:`~tcfuse.data.collate.WindowBatch`
    to a new :class:`~tcfuse.data.collate.WindowBatch` with modified source values.  The
    Lightning module is task-agnostic: it delegates the forward pass entirely to the injected
    model and leaves loss computation to subclasses (see :meth:`_shared_step`).

    Suitable as a backbone for reconstruction, imputation, denoising, or cross-source
    translation tasks.

    Args:
        model: The source-transformation model.  Can be either a fully instantiated
            :class:`~torch.nn.Module` or a factory callable (e.g. a Hydra partial from
            ``_partial_: true``) that accepts ``sources_metadata`` as its sole keyword
            argument and returns an :class:`~torch.nn.Module`.  Using a factory lets
            the backbone constructor read channel counts and shapes directly from the
            metadata without duplicating that information in the Hydra config.
        sources_metadata: Static descriptors for sources present in training samples.
        normalization_stats: Per-source, per-channel ``mean``/``std`` statistics
            (see :meth:`normalize`). Produced by
            ``scripts/preprocess/compute_normalization.py`` and shaped
            ``{source_name: {"kind": str, "channels": {channel: {"mean", "std", "count"}}}}``.
            Every source in ``sources_metadata`` must have an entry.
        adamw_kwargs: Keyword arguments for :class:`torch.optim.AdamW` (excluding ``params``).
        lr_scheduler_kwargs: Keyword arguments for :class:`CosineAnnealingWarmupRestarts`.
        validation_dir: Directory where validation figures are written each epoch.
        experiment_name: Short name from the experiment config (``cfg["name"]``), used to
            build the W&B run's display name as ``{experiment_name}-{run_id}`` (see
            :meth:`on_train_start`).
    """

    def __init__(
        self,
        model: nn.Module | Callable[..., nn.Module],
        sources_metadata: MultisourceMetadata,
        normalization_stats: dict[str, Any],
        adamw_kwargs: dict[str, Any],
        lr_scheduler_kwargs: dict[str, Any],
        validation_dir: str | Path,
        experiment_name: str,
    ) -> None:
        super().__init__()
        # Snapshot metadata first — the model factory needs it to allocate parameters.
        self._sources_metadata = MultisourceMetadata.from_dict(sources_metadata.to_dict())
        self._adamw_kwargs = dict(adamw_kwargs)
        self._lr_scheduler_kwargs = dict(lr_scheduler_kwargs)
        self._validation_dir = Path(validation_dir)
        self._experiment_name = experiment_name
        # If model is a Hydra partial (not yet an nn.Module), call it now so the
        # backbone can read channel counts and shapes from sources_metadata.
        if not isinstance(model, nn.Module):
            model = model(sources_metadata=sources_metadata)
        self.model = model
        # Build per-source mean/std buffers from the training-split statistics.
        self._register_normalization_buffers(normalization_stats)
        # Do not serialize the full model tree, metadata, or raw stats into hparams;
        # the normalization tensors are persisted as buffers instead.
        self.save_hyperparameters(ignore=["model", "sources_metadata", "normalization_stats"])

    @property
    def sources_metadata(self) -> MultisourceMetadata:
        """Static source descriptors (channels, shape, kind) for this run."""
        return MultisourceMetadata.from_dict(self._sources_metadata.to_dict())

    def forward(self, batch: WindowBatch) -> WindowBatch:
        """Run the injected model on a collated window batch, returning a new WindowBatch."""
        return self.model(self.preprocess_batch(batch))  # type: ignore[return-value]

    @abstractmethod
    def _shared_step(self, batch: WindowBatch, stage: str) -> torch.Tensor:
        """Forward pass and loss for train / validation.

        Subclasses must override this method to implement a concrete loss over the
        transformed :class:`WindowBatch` returned by :meth:`forward`.

        Args:
            batch: Collated input batch from the dataloader.
            stage: One of ``"train"`` or ``"val"``.

        Returns:
            Scalar loss tensor to be logged and optimized.
        """

    def training_step(self, batch: WindowBatch, batch_idx: int) -> torch.Tensor:
        """One training optimizer step (in normalized space)."""
        loss = self._shared_step(self.normalize(batch), "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: WindowBatch, batch_idx: int) -> torch.Tensor:
        """One validation forward pass (in normalized space)."""
        loss = self._shared_step(self.normalize(batch), "val")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: WindowBatch, batch_idx: int) -> WindowBatch:
        """Inference hook; runs the model in normalized space and de-normalizes the output.

        Returns the transformed :class:`WindowBatch` back in physical units so
        downstream consumers see predictions on the original scale.
        """
        return self.denormalize(self(self.normalize(batch)))

    def preprocess_batch(self, batch: WindowBatch) -> WindowBatch:
        """Apply pre-backbone preprocessing to every source in a batch.

        Runs after normalization and before the batch reaches the backbone in
        the training / validation / predict steps.

        Args:
            batch: Collated, normalized window batch.

        Returns:
            A new WindowBatch with updated sources.
        """
        # Shallow-copy the sources dict; every entry is rebuilt with cleaned values.
        new_sources: dict[tuple[str, int], TorchSource] = {}
        for key, source in batch.sources.items():
            # Only NaN-fill positions get zeroed; masked-but-finite values (e.g. learned
            # mask tokens substituted by a subclass) are left untouched.
            new_values = torch.where(
                torch.isnan(source.values), source.values.new_zeros(()), source.values
            )
            new_sources[key] = dataclasses.replace(source, values=new_values)
        return dataclasses.replace(batch, sources=new_sources)

    def on_train_start(self) -> None:
        """Log static run metadata to W&B once, at the very start of training."""
        # Only rank 0 holds the real wandb run; other DDP ranks get a no-op logger.
        if not self.trainer.is_global_zero:
            return
        wandb_logger = cast(WandbLogger, self.logger)
        # Display name only; the run id (used for resume across SLURM requeues) is set
        # once in train.py and never touched here. Deriving from .id rather than .name
        # keeps this idempotent if on_train_start runs again after a requeue.
        wandb_logger.experiment.name = f"{self._experiment_name}-{wandb_logger.experiment.id}"
        train_dataloader = self.trainer.train_dataloader
        # Per .agents/context.md W&B conventions: always log source types and
        # the number of training samples.
        wandb_logger.experiment.config.update(
            {
                "sources": self._sources_metadata.to_dict(),
                "num_training_samples": len(train_dataloader.dataset),  # type: ignore[arg-type]
            },
            allow_val_change=True,
        )

    def on_train_end(self) -> None:
        """Log peak GPU memory usage to W&B at the end of training."""
        if not self.trainer.is_global_zero:
            return
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            cast(WandbLogger, self.logger).experiment.summary["train/gpu_mem_peak_mb"] = peak_mb

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
        figure_path = self._validation_dir / f"{epoch:04d}_val_summary.svg"
        # No-op until the TODO above actually writes figure_path; activates
        # automatically once tcfuse.data.visualization.training is implemented.
        if figure_path.exists():
            wandb_logger = cast(WandbLogger, self.logger)
            wandb_logger.experiment.log({"val/summary": wandb.Image(str(figure_path))})

    def _register_normalization_buffers(self, normalization_stats: dict[str, Any]) -> None:
        """Register per-source mean/std buffers ordered by each source's channels.

        Statistics are per-channel (one mean/std per channel, pooled over levels or
        pixels), so a single ``(C,)`` vector per source broadcasts against the trailing
        channel axis for every :class:`SourceKind`. Buffers move with the module and are
        saved into checkpoints, so inference does not need the stats YAML on disk.

        Args:
            normalization_stats: Mapping ``{source_name: {"channels": {channel:
                {"mean", "std", ...}}}}`` covering every source in ``sources_metadata``.

        Raises:
            KeyError: If a source or one of its channels is missing from the stats.
        """
        # Names of sources we normalize; used to look up buffers at runtime.
        self._normalized_sources: list[str] = []
        for name in self._sources_metadata.names:
            if name not in normalization_stats:
                raise KeyError(
                    f"No normalization statistics for source {name!r}. "
                    "Re-run scripts/preprocess/compute_normalization.py."
                )
            channel_stats = normalization_stats[name]["channels"]
            channels = self._sources_metadata[name].channels
            # Order mean/std by the source's canonical channel list (matches values' last axis).
            means = torch.tensor(
                [float(channel_stats[ch]["mean"]) for ch in channels], dtype=torch.float32
            )
            stds = torch.tensor(
                [float(channel_stats[ch]["std"]) for ch in channels], dtype=torch.float32
            )
            # Guard against divide-by-zero on constant channels (std == 0).
            stds = stds.clamp_min(1e-6)
            self.register_buffer(self._mean_buffer_name(name), means)
            self.register_buffer(self._std_buffer_name(name), stds)
            self._normalized_sources.append(name)

    @staticmethod
    def _mean_buffer_name(source_name: str) -> str:
        """Deterministic buffer attribute name holding a source's per-channel means."""
        return f"_norm_mean__{source_name}"

    @staticmethod
    def _std_buffer_name(source_name: str) -> str:
        """Deterministic buffer attribute name holding a source's per-channel stds."""
        return f"_norm_std__{source_name}"

    def _affine_transform(self, batch: WindowBatch, *, invert: bool) -> WindowBatch:
        """Apply ``(v - mean) / std`` (or its inverse) per source, returning a new batch.

        Only sources with registered statistics are transformed; all other batch
        fields (coords, mask, time, scalar attributes) are carried over untouched.
        NaN-fill values at masked/padding positions stay NaN and are handled by masks
        downstream. The input batch is not mutated.

        Args:
            batch: Collated window batch (already on the module's device).
            invert: If ``True`` de-normalize (``v * std + mean``); otherwise normalize.

        Returns:
            A new :class:`WindowBatch` with transformed source values.
        """
        # Shallow-copy the sources dict so untransformed entries are shared, not rebuilt.
        new_sources: dict[tuple[str, int], TorchSource] = dict(batch.sources)
        for key, source in batch.sources.items():
            source_name, _idx = key
            if source_name not in self._normalized_sources:
                continue
            # Buffers are (C,); they broadcast over the trailing channel axis of values.
            mean = getattr(self, self._mean_buffer_name(source_name))
            std = getattr(self, self._std_buffer_name(source_name))
            if invert:
                new_values = source.values * std + mean
            else:
                new_values = (source.values - mean) / std
            # Rebuild the source with new values; every other field is unchanged.
            new_sources[key] = dataclasses.replace(source, values=new_values)
        return dataclasses.replace(batch, sources=new_sources)

    def normalize(self, batch: WindowBatch) -> WindowBatch:
        """Center and scale every source's values into normalized space.

        Subtracts the per-channel training mean and divides by the per-channel
        training std. Applied before the backbone sees the batch; training and
        validation run entirely in this normalized space.
        """
        return self._affine_transform(batch, invert=False)

    def denormalize(self, batch: WindowBatch) -> WindowBatch:
        """Invert :meth:`normalize`, mapping values back to physical units."""
        return self._affine_transform(batch, invert=True)

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
