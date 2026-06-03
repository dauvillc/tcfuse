"""LightningDataModule wrapping TCWindowDataset for multi-source TC window training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning
from torch.utils.data import DataLoader

from tcfuse.data.collate import collate_window_samples
from tcfuse.data.dataset import TCWindowDataset, WindowSample
from tcfuse.data.sources.metadata import MultisourceMetadata

if TYPE_CHECKING:
    pass

_SOURCES_METADATA_FILENAME = "sources_metadata.yaml"


class TCWindowDataModule(lightning.LightningDataModule):
    """LightningDataModule over best-track assimilation windows.

    Constructs :class:`~tcfuse.data.dataset.TCWindowDataset` instances for each
    split and exposes them through Lightning's dataloader hooks.

    Args:
        assembled_root: Root directory for assembled preprocessed data; must
            contain ``train.parquet``, ``val.parquet``, ``test.parquet``, and
            ``sources_metadata.yaml``.
        dataloader_kwargs: Keyword arguments forwarded to every
            :class:`~torch.utils.data.DataLoader` (e.g. ``batch_size``,
            ``num_workers``, ``pin_memory``).  ``collate_fn`` and ``shuffle``
            are injected by this class and must not appear in this dict.
    """

    def __init__(
        self,
        assembled_root: Path | str,
        dataloader_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self._assembled_root = Path(assembled_root)
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
            raise RuntimeError(
                "sources_metadata is not available before setup() is called."
            )
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
        assert self._sources_metadata is not None
        return TCWindowDataset(
            self._assembled_root,
            split,  # type: ignore[arg-type]
            sources_metadata=self._sources_metadata,
        )

    def _make_dataloader(
        self, dataset: TCWindowDataset, *, shuffle: bool
    ) -> DataLoader[WindowSample]:
        return DataLoader(
            dataset,
            collate_fn=collate_window_samples,
            shuffle=shuffle,
            **self._dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader[WindowSample]:
        """DataLoader for the training split (shuffle=True)."""
        assert self._train_dataset is not None, "Call setup('fit') first."
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[WindowSample]:
        """DataLoader for the validation split (shuffle=False)."""
        assert self._val_dataset is not None, "Call setup('fit') or setup('validate') first."
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[WindowSample]:
        """DataLoader for the test split (shuffle=False)."""
        assert self._test_dataset is not None, "Call setup('test') first."
        return self._make_dataloader(self._test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader[WindowSample]:
        """DataLoader for inference over the test split (shuffle=False)."""
        assert self._test_dataset is not None, "Call setup('predict') first."
        return self._make_dataloader(self._test_dataset, shuffle=False)
