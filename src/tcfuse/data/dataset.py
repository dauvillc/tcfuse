"""PyTorch Dataset for best-track assimilation windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from torch.utils.data import Dataset

from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.storm_data import StormData
from tcfuse.data.window_index import SplitName

_SOURCES_METADATA_FILENAME = "sources_metadata.yaml"


@dataclass
class WindowSample:
    """One training sample: window metadata plus filtered storm sources.

    Args:
        storm_data: Sources whose ``time_utc`` falls inside the assimilation
            window (plus the target snapshot when it sits outside that span).
        sample_id: Window identifier (``window_id`` from the window index).
        window_ref_time_utc: Assimilation anchor time ``t0``.
        window_start_time_utc: Inclusive lower bound of the assimilation window.
        window_end_time_utc: Inclusive upper bound of the assimilation window.
        sid: IBTrACS storm identifier.
        season: TC season year.
        basin: Ocean basin code.
        subbasin: IBTrACS sub-basin code.
        usa_atcf_id: Optional USA ATCF identifier.
        is_target: Maps ``(source_name, source_index)`` to ``True`` when that
            snapshot is the window's target anchor.  ``source_index`` is the
            chronological rank within each source name, matching the ordering
            used by :func:`~tcfuse.data.collate.collate_window_samples`.
    """

    storm_data: StormData
    sample_id: str
    window_ref_time_utc: str
    window_start_time_utc: str
    window_end_time_utc: str
    sid: str
    season: int
    basin: str
    subbasin: str
    is_target: dict[tuple[str, int], bool]
    usa_atcf_id: str | None = None


class TCWindowDataset(Dataset[WindowSample]):
    """Map-style dataset over best-track assimilation windows.

    Each item corresponds to one unique ``window_id`` in the long-format
    window index produced by ``scripts/preprocess/build_windows.py``.

    Args:
        assembled_root: Root directory for assembled data
            (``cfg.paths.preprocessed_data``).
        windows_setup_name: Name of the windows configuration; selects
            ``{assembled_root}/{windows_setup_name}/{split}_windows.parquet``.
        split: Which window-index parquet to load (``"train"``, ``"val"``,
            or ``"test"``).
    """

    def __init__(
        self,
        assembled_root: Path,
        windows_setup_name: str,
        split: SplitName,
    ) -> None:
        self._assembled_root = assembled_root
        self._split: SplitName = split

        # Load long-format window index (one row per window x source snapshot).
        index_path = assembled_root / windows_setup_name / f"{split}_windows.parquet"
        self._windows_index = pd.read_parquet(index_path)

        # Pre-group for O(1) per-item lookup; preserve parquet row order.
        self._window_ids: list[str] = (
            self._windows_index["window_id"].drop_duplicates().tolist()
        )
        self._window_groups: dict[str, pd.DataFrame] = cast(
            dict[str, pd.DataFrame],
            {wid: grp for wid, grp in self._windows_index.groupby("window_id", sort=False)},
        )

        # Always load sources_metadata from disk.
        metadata_path = assembled_root / _SOURCES_METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"sources_metadata.yaml not found at {metadata_path}"
            )
        loaded = MultisourceMetadata.from_yaml(metadata_path)
        # Snapshot so later external mutations cannot leak in.
        self._sources_metadata = MultisourceMetadata.from_dict(loaded.to_dict())

    @property
    def sources_metadata(self) -> MultisourceMetadata:
        """Static descriptors (channels, shape, kind) for every assembled source."""
        return MultisourceMetadata.from_dict(self._sources_metadata.to_dict())

    @property
    def index(self) -> pd.DataFrame:
        """Long-format window-index DataFrame backing this dataset."""
        return self._windows_index

    @property
    def split(self) -> SplitName:
        """Split name used to build this dataset."""
        return self._split

    def __len__(self) -> int:
        """Number of window samples in the dataset."""
        return len(self._window_ids)

    def __getitem__(self, idx: int) -> WindowSample:
        """Get a window sample by index."""
        window_rows = self._window_groups[self._window_ids[idx]]
        first = window_rows.iloc[0]
        sid = str(first["sid"])

        # Load exactly the (source_name, time_utc) pairs listed in the window
        # index — these are already filtered by build_windows.py according to
        # the input_sources configuration, so no further range arithmetic is needed.
        snapshots: set[tuple[str, str]] = set(
            zip(
                window_rows["source_name"].astype(str),
                window_rows["time_utc"].astype(str),
            )
        )
        storm_data = StormData.from_disk_for_snapshots(
            self._assembled_root,
            sid,
            snapshots,
        )

        # Build a lookup: (source_name, normalised pd.Timestamp) -> is_target.
        # pd.Timestamp normalisation handles format differences between the index
        # string and the isoformat key stored in StormData.sources.
        index_target: dict[tuple[str, pd.Timestamp], bool] = {
            (
                str(row["source_name"]),
                cast(pd.Timestamp, pd.Timestamp(str(row["time_utc"]))),
            ): bool(row["is_target"])
            for _, row in window_rows.iterrows()
        }

        # Assign chronological source_index — mirrors collate_window_samples ordering
        # so that is_target keys are consistent across dataset and collate.
        sorted_keys = sorted(storm_data.sources.keys(), key=lambda k: k[1])
        name_count: dict[str, int] = {}
        is_target: dict[tuple[str, int], bool] = {}
        for source_name, time_utc_str in sorted_keys:
            src_idx = name_count.get(source_name, 0)
            name_count[source_name] = src_idx + 1
            is_target[(source_name, src_idx)] = index_target.get(
                (source_name, cast(pd.Timestamp, pd.Timestamp(time_utc_str))), False
            )

        usa_atcf_id = first["usa_atcf_id"]
        return WindowSample(
            storm_data=storm_data,
            sample_id=str(first["window_id"]),
            window_ref_time_utc=str(first["window_ref_time_utc"]),
            window_start_time_utc=str(first["window_start_time_utc"]),
            window_end_time_utc=str(first["window_end_time_utc"]),
            sid=sid,
            season=int(first["season"]),
            basin=str(first["basin"]),
            subbasin=str(first["subbasin"]),
            usa_atcf_id=None if pd.isna(usa_atcf_id) else str(usa_atcf_id),
            is_target=is_target,
        )
