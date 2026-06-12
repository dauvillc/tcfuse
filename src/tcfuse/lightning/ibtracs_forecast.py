"""Lightning module for IBTrACS best-track masked-prediction forecasting."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.ibtracs import IBTRACS_CHANNELS, IBTRACS_SOURCE_NAME
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.lightning.base_module import BaseLightningModule


class IBTrACSForecastLightningModule(BaseLightningModule):
    """Train IBTrACS best-track forecasting via masked-prediction over assimilation windows.

    At each step the ibtracs best-track sources are hidden from the model: their
    values and coords are replaced with learned mask tokens and the mask is set to
    all-False.  The model must reconstruct the original best-track values from the
    remaining (non-ibtracs) observations.  The training signal is MSE between the
    model's predicted ibtracs values and the original (pre-masking) values,
    averaged over all valid ``(batch_sample, channel)`` positions.

    Args:
        model: Backbone + task head.  Can be a fully instantiated
            :class:`~torch.nn.Module` or a factory callable (e.g. a Hydra partial
            from ``_partial_: true``) that accepts ``sources_metadata`` as its sole
            keyword argument and returns an :class:`~torch.nn.Module`.  The base class
            handles factory instantiation automatically.  The model must accept a
            :class:`WindowBatch` in ``forward`` and return a :class:`WindowBatch`
            whose ibtracs sources carry predictions.
        sources_metadata: Static descriptors for sources present in training samples.
        normalization_stats: Per-source, per-channel mean/std statistics applied by the
            base module (see :meth:`BaseLightningModule.normalize`). The masked-prediction
            loss is therefore computed in normalized space.
        adamw_kwargs: Keyword arguments for :class:`torch.optim.AdamW` (excluding ``params``).
        lr_scheduler_kwargs: Keyword arguments for :class:`CosineAnnealingWarmupRestarts`.
        validation_dir: Directory where validation figures are written each epoch.
    """

    def __init__(
        self,
        model: nn.Module | Callable[..., nn.Module],
        sources_metadata: MultisourceMetadata,
        normalization_stats: dict[str, Any],
        adamw_kwargs: dict[str, Any],
        lr_scheduler_kwargs: dict[str, Any],
        validation_dir: str | Path,
    ) -> None:
        super().__init__(
            model=model,
            sources_metadata=sources_metadata,
            normalization_stats=normalization_stats,
            adamw_kwargs=adamw_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            validation_dir=validation_dir,
        )
        # Learned mask tokens substituted in place of real ibtracs values/coords.
        # Initialised to zeros; trained jointly with the backbone.
        self._ibtracs_values_token = nn.Parameter(torch.zeros(len(IBTRACS_CHANNELS)))
        self._ibtracs_coords_token = nn.Parameter(torch.zeros(2))

    def _mask_ibtracs(
        self,
        batch: WindowBatch,
    ) -> tuple[WindowBatch, dict[tuple[str, int], TorchSource]]:
        """Replace every ibtracs source in ``batch`` with learned mask tokens.

        Args:
            batch: Collated input batch from the dataloader.

        Returns:
            A ``(masked_batch, originals)`` pair where ``masked_batch`` has all
            ibtracs sources replaced and ``originals`` maps each ibtracs key to
            its original :class:`TorchSource` (for loss computation).
        """
        originals: dict[tuple[str, int], TorchSource] = {}
        # Shallow-copy the sources dict so we can replace ibtracs entries safely.
        masked_sources: dict[tuple[str, int], TorchSource] = dict(batch.sources)

        for key, source in batch.sources.items():
            source_name, _idx = key
            if source_name != IBTRACS_SOURCE_NAME:
                continue

            originals[key] = source
            B = source.batch_size

            # Broadcast the learned (C,) / (2,) tokens to (B, C) / (B, 2).
            masked_values = self._ibtracs_values_token.unsqueeze(0).expand(B, -1)
            masked_coords = self._ibtracs_coords_token.unsqueeze(0).expand(B, -1)
            # all-False mask: signals to the model that these values are hidden.
            masked_mask = torch.zeros_like(source.mask)

            masked_sources[key] = TorchSource(
                kind=source.kind,
                values=masked_values,
                coords=masked_coords,
                source_name=source.source_name,
                channels=source.channels,
                mask=masked_mask,
                time=source.time,  # time is not masked
            )

        return dataclasses.replace(batch, sources=masked_sources), originals

    def _shared_step(self, batch: WindowBatch, stage: str) -> torch.Tensor:
        """Mask ibtracs, run the model, return MSE against the original values.

        Args:
            batch: Collated input batch from the dataloader.
            stage: One of ``"train"`` or ``"val"``.

        Returns:
            Scalar MSE loss averaged over all valid ibtracs ``(sample, channel)`` positions.
        """
        # Hide ibtracs sources from the model and stash the originals.
        masked_batch, originals = self._mask_ibtracs(batch)

        # Forward: model must return a WindowBatch with ibtracs predictions.
        output_batch = self(masked_batch)

        # Collect squared residuals over all valid positions across all ibtracs keys.
        all_diffs: list[torch.Tensor] = []
        for key, original in originals.items():
            # original.mask is True where the original best-track value was present.
            valid = original.mask  # (B, C) bool
            if not valid.any():
                # This key is all NaN-fill padding — skip it.
                continue
            pred_values = output_batch.sources[key].values  # (B, C)
            true_values = original.values  # (B, C)
            # Index by valid mask to get a flat vector of residuals.
            all_diffs.append(pred_values[valid] - true_values[valid])

        if not all_diffs:
            # No valid ibtracs positions in this batch; return a zero-grad loss.
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        residuals = torch.cat(all_diffs)
        return (residuals**2).mean()
