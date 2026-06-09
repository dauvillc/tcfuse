"""LightningDataModule wrapping TCWindowDataset for multi-source TC window training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, override

import lightning
import torch
from torch.utils.data import DataLoader

from tcfuse.data.collate import WindowBatch, collate_window_samples
from tcfuse.data.dataset import TCWindowDataset
from tcfuse.data.sources.metadata import MultisourceMetadata

_SOURCES_METADATA_FILENAME = "sources_metadata.yaml"


class TCWindowDataModule(lightning.LightningDataModule):
    """LightningDataModule over best-track assimilation windows.

    Constructs :class:`~tcfuse.data.dataset.TCWindowDataset` instances for each
    split and exposes them through Lightning's dataloader hooks.

    Args:
        assembled_root: Root directory for assembled preprocessed data; must
            contain ``sources_metadata.yaml`` and a
            ``{windows_setup_name}/`` subdirectory with
            ``train_windows.parquet``, ``val_windows.parquet``, and
            ``test_windows.parquet``.
        windows_setup_name: Name of the windows configuration (selects the
            matching subdirectory under ``assembled_root``).
        dataloader_kwargs: Keyword arguments forwarded to every
            :class:`~torch.utils.data.DataLoader` (e.g. ``batch_size``,
            ``num_workers``, ``pin_memory``).  ``collate_fn`` and ``shuffle``
            are injected by this class and must not appear in this dict.
    """

    def __init__(
        self,
        assembled_root: Path | str,
        windows_setup_name: str,
        dataloader_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self._assembled_root = Path(assembled_root)
        self._windows_setup_name = windows_setup_name
        # Snapshot so later mutations to the injected dict cannot leak in.
        self._dataloader_kwargs = dict(dataloader_kwargs)

        self._sources_metadata: MultisourceMetadata | None = None
        self._train_dataset: TCWindowDataset | None = None
        self._val_dataset: TCWindowDataset | None = None
        self._test_dataset: TCWindowDataset | None = None

    @property
    def sources_metadata(self) -> MultisourceMetadata:
        """Static source descriptors loaded during :meth:`setup`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        if self._sources_metadata is None:
            raise RuntimeError("sources_metadata is not available before setup() is called.")
        return MultisourceMetadata.from_dict(self._sources_metadata.to_dict())

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the requested stage.

        Called by the Lightning Trainer before each stage. Loads
        ``sources_metadata.yaml`` once and reuses it across splits.

        Args:
            stage: One of ``"fit"``, ``"validate"``, ``"test"``,
                ``"predict"``, or ``None`` (all splits).
        """
        # Load metadata once; reuse across splits to avoid redundant I/O.
        if self._sources_metadata is None:
            loaded = MultisourceMetadata.from_yaml(
                self._assembled_root / _SOURCES_METADATA_FILENAME
            )
            self._sources_metadata = MultisourceMetadata.from_dict(loaded.to_dict())

        make = self._make_dataset
        if stage in ("fit", None):
            self._train_dataset = make("train")
            self._val_dataset = make("val")
        if stage in ("validate", None):
            self._val_dataset = make("val")
        if stage in ("test", "predict", None):
            self._test_dataset = make("test")

    def _make_dataset(self, split: str) -> TCWindowDataset:
        return TCWindowDataset(
            self._assembled_root,
            self._windows_setup_name,
            split,  # type: ignore[arg-type]
        )

    def _make_dataloader(
        self, dataset: TCWindowDataset, *, shuffle: bool
    ) -> DataLoader[WindowBatch]:
        # collate_fn yields WindowBatch; cast because DataLoader infers WindowSample from dataset.
        return cast(
            DataLoader[WindowBatch],
            DataLoader(
                dataset,
                collate_fn=collate_window_samples,
                shuffle=shuffle,
                **self._dataloader_kwargs,
            ),
        )

    @override
    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Move TorchSource tensors in a WindowBatch to ``device``."""
        if isinstance(batch, WindowBatch):
            # Move every batched source tensor bundle to the target device.
            batch.sources = {key: src.to(device) for key, src in batch.sources.items()}
            return batch
        # Fallback for any unexpected batch type.
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def train_dataloader(self) -> DataLoader[WindowBatch]:
        """DataLoader for the training split (shuffle=True)."""
        assert self._train_dataset is not None, "Call setup('fit') first."
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[WindowBatch]:
        """DataLoader for the validation split (shuffle=False)."""
        assert self._val_dataset is not None, "Call setup('fit') or setup('validate') first."
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[WindowBatch]:
        """DataLoader for the test split (shuffle=False)."""
        assert self._test_dataset is not None, "Call setup('test') first."
        return self._make_dataloader(self._test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader[WindowBatch]:
        """DataLoader for inference over the test split (shuffle=False)."""
        assert self._test_dataset is not None, "Call setup('predict') first."
        return self._make_dataloader(self._test_dataset, shuffle=False)
