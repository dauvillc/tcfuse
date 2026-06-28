"""Lightning module for general masked-source reconstruction."""

from __future__ import annotations

import dataclasses

import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.torch_source import TorchSource
from tcfuse.lightning.base_module import BaseLightningModule


class MaskedReconstructionLightningModule(BaseLightningModule):
    """Train any backbone to reconstruct target sources via masked prediction.

    At each step, every ``(source_name, source_index)`` slot marked as a target
    in :attr:`~tcfuse.data.collate.WindowBatch.is_target` is hidden from the
    model: its values are replaced by NaN and its mask is set to all-False for
    the target batch indexes.  Masked targets are therefore indistinguishable
    from genuinely absent sources, preserving the existing missing-data
    invariant.  The model must reconstruct the original values from the
    remaining visible observations.

    The training signal is MSE between the model's predicted values and the
    original (pre-masking) values, averaged over all valid
    ``(batch_sample, ...)`` positions — i.e., positions that were both marked
    as targets *and* had finite data before masking.  Loss is computed in
    **normalized space**.

    No additional parameters beyond those in :class:`BaseLightningModule` are
    introduced; ``__init__`` is inherited unchanged.  The backbone constructor
    is expected to accept ``sources_metadata`` as a keyword argument so it can
    allocate per-source parameters at construction time; when the config uses
    ``_partial_: true``, :class:`BaseLightningModule` calls the resulting
    factory with ``sources_metadata`` automatically.
    """

    def _mask_targets(
        self,
        batch: WindowBatch,
    ) -> tuple[WindowBatch, dict[tuple[str, int], TorchSource]]:
        """Replace target source slots with NaN / all-False mask.

        For each ``(source_name, source_index)`` key, the ``is_target`` tensor
        marks which batch samples treat that slot as a prediction target.  At
        those batch indexes the source values are set to NaN and the mask is
        zeroed so the backbone cannot see the original observations.

        Args:
            batch: Collated input batch (already normalized by the caller).

        Returns:
            A ``(masked_batch, originals)`` pair.  ``masked_batch`` has all
            target slots hidden; ``originals`` maps each key that had at least
            one target sample to the corresponding pre-masking
            :class:`~tcfuse.data.sources.torch_source.TorchSource`.
        """
        # Shallow-copy so untouched sources share tensors with the original batch.
        masked_sources: dict[tuple[str, int], TorchSource] = dict(batch.sources)
        originals: dict[tuple[str, int], TorchSource] = {}

        for key, source in batch.sources.items():
            # target_flags: (B,) bool — True where this slot is a prediction target.
            target_flags = batch.is_target[key]
            if not target_flags.any():
                continue

            originals[key] = source

            # Clone to avoid mutating tensors shared with the caller.
            new_values = source.values.clone()
            new_mask = source.mask.clone()

            # target_flags is (B,); indexing along the batch dim zeros out all
            # spatial/channel positions for each target sample automatically.
            new_values[target_flags] = float("nan")
            new_mask[target_flags] = False

            masked_sources[key] = dataclasses.replace(source, values=new_values, mask=new_mask)

        return dataclasses.replace(batch, sources=masked_sources), originals

    def predict_step(self, batch: WindowBatch, batch_idx: int) -> WindowBatch:
        """Reconstruct masked target sources for inference, in physical units.

        Overrides :meth:`BaseLightningModule.predict_step` to apply the *same*
        target masking used during training before the forward pass.  Without
        masking the backbone would simply see the ground-truth target values and
        echo them back, so evaluation must hide targets exactly as training does.

        The returned :class:`WindowBatch` carries reconstructed values for the
        target slots (and transformed values for visible slots, which callers
        ignore) de-normalized back to physical units.

        Args:
            batch: Collated input batch in physical units.
            batch_idx: Index of the batch (unused; kept for the Lightning API).

        Returns:
            A de-normalized :class:`WindowBatch` with target slots reconstructed.
        """
        # Normalize, then hide every target slot just like _shared_step does.
        masked_batch, _originals = self._mask_targets(self.normalize(batch))
        # Run the backbone on the masked batch and map values back to physical units.
        return self.denormalize(self(masked_batch))

    def _shared_step(self, batch: WindowBatch, stage: str) -> torch.Tensor:
        """Mask targets, run the model, return MSE against the original values.

        Args:
            batch: Collated input batch (already normalized by the caller).
            stage: One of ``"train"`` or ``"val"``.

        Returns:
            Scalar MSE loss averaged over all valid target ``(sample, ...)``
            positions, or a zero-gradient scalar if the batch contains no
            valid targets.
        """
        # Hide target sources from the backbone and stash originals for the loss.
        masked_batch, originals = self._mask_targets(batch)

        # Forward: the backbone must predict masked values from visible sources.
        output_batch = self(masked_batch)

        # Collect squared residuals over every valid target position.
        all_diffs: list[torch.Tensor] = []
        for key, original in originals.items():
            # is_target is (B,); broadcast to match the full values shape.
            is_target = batch.is_target[key]
            is_target_bc = is_target.reshape([-1] + [1] * (original.mask.ndim - 1))
            # valid: True where the sample was a target AND had real data before masking.
            valid = original.mask & is_target_bc

            if not valid.any():
                continue

            pred_values = output_batch.sources[key].values
            true_values = original.values

            # Accumulate per-channel validation metrics in physical space (no-op at train time).
            if stage == "val":
                source_name, _ = key
                self._update_val_metrics(source_name, pred_values, true_values, valid)

            all_diffs.append(pred_values[valid] - true_values[valid])

        if not all_diffs:
            # No valid target positions in this batch — return a zero-grad loss.
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        residuals = torch.cat(all_diffs)
        return (residuals**2).mean()
