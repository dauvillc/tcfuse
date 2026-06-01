"""PyTorch Dataset for best-track assimilation windows."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.storm_data import StormData
from tcfuse.data.window_index import SplitName, load_split_index

_LABEL_COLUMN_PREFIX = "lead_"
_SOURCES_METADATA_FILENAME = "sources_metadata.yaml"


@dataclass
class WindowSample:
    """One training sample: window metadata plus filtered storm sources.

    Args:
        storm_data: Sources whose ``snapshot_time_utc`` falls inside the
            assimilation window.
        sample_id: Window identifier from the split index.
        init_time_utc: Assimilation anchor time ``t0``.
        window_start_time_utc: Inclusive lower bound of the assimilation window.
        window_end_time_utc: Inclusive upper bound of the assimilation window.
        sid: IBTrACS storm identifier.
        season: TC season year.
        basin: Ocean basin code.
        subbasin: IBTrACS sub-basin code.
        usa_atcf_id: Optional USA ATCF identifier from the split index.
        _index_row: Full split-index row backing :attr:`labels`.
    """

    storm_data: StormData
    sample_id: str
    init_time_utc: str
    window_start_time_utc: str
    window_end_time_utc: str
    sid: str
    season: int
    basin: str
    subbasin: str
    _index_row: pd.Series = dataclasses.field(repr=False, compare=False)
    usa_atcf_id: str | None = None

    @property
    def labels(self) -> pd.Series:
        """Lead-time target columns from the split index row."""
        return self._index_row.loc[
            self._index_row.index.astype(str).str.startswith(_LABEL_COLUMN_PREFIX)
        ]


class TCWindowDataset(Dataset[WindowSample]):
    """Map-style dataset over best-track assimilation windows.

    Each item corresponds to one row of ``train.parquet``, ``val.parquet``, or
    ``test.parquet`` produced by ``scripts/preprocess/build_splits.py``.

    Args:
        assembled_root: Root directory for assembled data
            (``cfg.paths.preprocessed_data``).
        split: Which window-index parquet to load.
        index: Optional pre-loaded index for tests or subset debugging.
        sources_metadata: Optional pre-loaded source descriptors. When omitted,
            loads ``sources_metadata.yaml`` from ``assembled_root``.
    """

    def __init__(
        self,
        assembled_root: Path,
        split: SplitName,
        *,
        index: pd.DataFrame | None = None,
        sources_metadata: MultisourceMetadata | None = None,
    ) -> None:
        self._assembled_root = assembled_root
        self._split: SplitName = split
        self._index = index if index is not None else load_split_index(assembled_root, split)
        loaded_metadata = (
            sources_metadata
            if sources_metadata is not None
            else MultisourceMetadata.from_yaml(assembled_root / _SOURCES_METADATA_FILENAME)
        )
        # Snapshot so later mutations to injected or returned metadata cannot leak in.
        self._sources_metadata = MultisourceMetadata.from_dict(loaded_metadata.to_dict())

    @property
    def sources_metadata(self) -> MultisourceMetadata:
        """Static descriptors (channels, shape, kind) for every assembled source."""
        return MultisourceMetadata.from_dict(self._sources_metadata.to_dict())

    @property
    def index(self) -> pd.DataFrame:
        """Window-index DataFrame backing this dataset."""
        return self._index

    @property
    def split(self) -> SplitName:
        """Split name used to build this dataset."""
        return self._split

    def __len__(self) -> int:
        """Number of window samples in the dataset."""
        return len(self._index)

    def __getitem__(self, idx: int) -> WindowSample:
        """Get a window sample by index."""
        row = self._index.iloc[idx]
        sid = str(row["sid"])
        window_start = str(row["window_start_time_utc"])
        window_end = str(row["window_end_time_utc"])

        storm_data = StormData.from_disk(
            self._assembled_root,
            sid,
            window_start_utc=window_start,
            window_end_utc=window_end,
        )

        usa_atcf_id = row["usa_atcf_id"]
        atcf_id = None if pd.isna(usa_atcf_id) else str(usa_atcf_id)

        return WindowSample(
            storm_data=storm_data,
            sample_id=str(row["sample_id"]),
            init_time_utc=str(row["init_time_utc"]),
            window_start_time_utc=window_start,
            window_end_time_utc=window_end,
            sid=sid,
            season=int(row["season"]),
            basin=str(row["basin"]),
            subbasin=str(row["subbasin"]),
            _index_row=row,
            usa_atcf_id=atcf_id,
        )
