"""Custom collate function and batched sample container for TC window data."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from tcfuse.data.dataset import WindowSample
from tcfuse.data.sources.source import Source
from tcfuse.data.sources.torch_source import TorchSource


@dataclass
class WindowBatch:
    """Batched collection of window samples for use in a DataLoader.

    Holds data from B independent windows / storms. Scalar attributes are lists
    (one entry per sample). Source tensors are batched along a leading B axis
    and represented as :class:`~tcfuse.data.sources.torch_source.TorchSource`.

    The ``sources`` dict maps ``(source_name, source_index)`` to a batched
    :class:`~tcfuse.data.sources.torch_source.TorchSource`.
    ``source_index`` is the chronological rank of a snapshot within a single
    sample: if every sample has at most one snapshot of source "S", only
    ``("S", 0)`` will ever appear.  When a source appears K times in some
    sample, indices 0 ... K-1 are all present.

    Samples that are missing a given ``(source_name, source_index)`` slot are
    represented as NaN tensors (values, coords, and time all NaN, mask all False).

    Args:
        sources: Dict from ``(source_name, source_index)`` to a TorchSource.
            Keys are sorted lexicographically for deterministic ordering.
        is_target: Dict from ``(source_name, source_index)`` to a ``(B,)`` bool
            tensor indicating which samples have that slot as a target snapshot.
            Keys match those in ``sources``.
        sample_ids: Window identifier strings, one per sample.
        window_ref_times_utc: Assimilation anchor times ``t0``, one per sample.
        window_start_times_utc: Inclusive window lower bounds, one per sample.
        window_end_times_utc: Inclusive window upper bounds, one per sample.
        sids: IBTrACS storm identifiers, one per sample.
        seasons: TC season years, one per sample.
        basins: Ocean basin codes, one per sample.
        subbasins: IBTrACS sub-basin codes, one per sample.
        usa_atcf_ids: Optional USA ATCF identifiers, one per sample (None when absent).
    """

    sources: dict[tuple[str, int], TorchSource]
    is_target: dict[tuple[str, int], torch.Tensor]
    sample_ids: list[str]
    window_ref_times_utc: list[str]
    window_start_times_utc: list[str]
    window_end_times_utc: list[str]
    sids: list[str]
    seasons: list[int]
    basins: list[str]
    subbasins: list[str]
    usa_atcf_ids: list[str | None]

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch."""
        return len(self.sample_ids)

    def to(self, device: torch.device | str) -> WindowBatch:
        """Return a copy with all source tensors and target flags on ``device``.

        Only the batched tensors move; the scalar list attributes (ids, times,
        basins, …) are carried over unchanged.
        """
        return replace(
            self,
            # Move each source's value/coord/mask/time tensors.
            sources={key: src.to(device) for key, src in self.sources.items()},
            # Move the (B,) is_target flags so they share the sources' device.
            is_target={key: flags.to(device) for key, flags in self.is_target.items()},
        )


def _time_encoding(time_utc: pd.Timestamp) -> torch.Tensor:
    """Encode a UTC timestamp as a 2-vector [day_of_year / 366, minute_of_day / 1440].

    The year is discarded; only the seasonal position and time-of-day are kept.

    Args:
        time_utc: UTC observation timestamp.

    Returns:
        Float32 tensor of shape (2,).
    """
    # day_of_year is 1-indexed (1-366); normalise to [1/366, 1.0].
    day_norm = time_utc.day_of_year / 366.0
    # minute_of_day spans [0, 1439]; normalise to [0, 1439/1440].
    minute_norm = (time_utc.hour * 60 + time_utc.minute) / 1440.0
    return torch.tensor([day_norm, minute_norm], dtype=torch.float32)


def collate_window_samples(samples: list[WindowSample]) -> WindowBatch:
    """Collate a list of WindowSamples into a single WindowBatch.

    Sources are merged across samples using a ``(source_name, source_index)``
    key space.  Within each sample, snapshots of the same source are ordered
    chronologically by ``time_utc``; the earliest gets index 0.

    Samples that lack a particular ``(source_name, source_index)`` slot receive
    NaN-filled values, coords, and time, and an all-False mask.

    Numpy arrays from each :class:`~tcfuse.data.sources.source.Source` are
    converted to torch tensors during stacking; the time encoding
    ``[day_of_year / 366, minute_of_day / 1440]`` is computed from
    ``source.time_utc`` and stacked into a ``(B, 2)`` tensor.

    Args:
        samples: Non-empty list of WindowSamples from the same dataset.

    Returns:
        A :class:`WindowBatch` with all tensors on CPU, one
        :class:`~tcfuse.data.sources.torch_source.TorchSource` per key.
    """
    # Step 1 — build a per-sample dict {(source_name, source_index): Source}.
    # Snapshots of the same source_name are sorted ascending by time_utc
    # so that index 0 always corresponds to the earliest overpass.
    per_sample: list[dict[tuple[str, int], Source]] = []
    for sample in samples:
        # Sort the storm_data.sources items by snapshot time (second element of
        # the StormData key tuple).
        sorted_items = sorted(
            sample.storm_data.sources.items(),
            key=lambda kv: kv[0][1],  # kv[0] = (source_name, time_utc)
        )

        # Assign a chronological index to each occurrence of each source_name.
        name_count: dict[str, int] = {}
        indexed: dict[tuple[str, int], Source] = {}
        for (source_name, _snapshot_time), source in sorted_items:
            idx = name_count.get(source_name, 0)
            name_count[source_name] = idx + 1
            indexed[(source_name, idx)] = source

        per_sample.append(indexed)

    # Step 2 — collect the union of all (source_name, source_index) keys seen
    # across every sample in the batch.
    all_keys: set[tuple[str, int]] = set()
    for indexed in per_sample:
        all_keys.update(indexed.keys())

    # Step 3 — for each key, find a reference Source (any sample that has it).
    # The reference provides kind, channels, and canonical tensor shapes.
    ref_source: dict[tuple[str, int], Source] = {}
    for indexed in per_sample:
        for key, source in indexed.items():
            if key not in ref_source:
                ref_source[key] = source

    # Step 4 — for each key, convert numpy->torch and stack B per-sample tensors.
    # Samples missing the key are NaN-filled (values/coords/time = NaN, mask = False).
    batched_sources: dict[tuple[str, int], TorchSource] = {}
    for key in sorted(all_keys):
        ref = ref_source[key]
        # Pre-build reference torch tensors so that NaN-fill shapes are consistent.
        ref_values_t = torch.from_numpy(np.array(ref.values, dtype=np.float32))
        ref_coords_t = torch.from_numpy(ref.coords)
        ref_mask_t = torch.from_numpy(np.array(ref.mask, dtype=bool))

        values_list: list[torch.Tensor] = []
        coords_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        time_list: list[torch.Tensor] = []

        for indexed in per_sample:
            if key in indexed:
                src = indexed[key]
                values_list.append(torch.from_numpy(np.array(src.values, dtype=np.float32)))
                coords_list.append(torch.from_numpy(src.coords))
                mask_list.append(torch.from_numpy(np.array(src.mask, dtype=bool)))
                time_list.append(_time_encoding(src.time_utc))
            else:
                # NaN-fill missing slots: full_like preserves shape automatically.
                values_list.append(torch.full_like(ref_values_t, float("nan")))
                coords_list.append(torch.full_like(ref_coords_t, float("nan")))
                # mask is bool — zeros_like gives all-False (all missing).
                mask_list.append(torch.zeros_like(ref_mask_t))
                # time is NaN for missing slots so the model can detect absence.
                time_list.append(torch.full((2,), float("nan"), dtype=torch.float32))

        # Stack along new leading batch dim -> (B, ...) tensors.
        batched_sources[key] = TorchSource(
            kind=ref.kind,
            values=torch.stack(values_list, dim=0),
            coords=torch.stack(coords_list, dim=0),
            source_name=ref.source_name,
            channels=ref.channels,
            mask=torch.stack(mask_list, dim=0),
            time=torch.stack(time_list, dim=0),
        )

    # Step 5 — stack is_target flags into (B,) bool tensors per key.
    batched_is_target: dict[tuple[str, int], torch.Tensor] = {
        key: torch.tensor(
            [sample.is_target.get(key, False) for sample in samples],
            dtype=torch.bool,
        )
        for key in sorted(all_keys)
    }

    # Step 6 — collect the scalar list attributes from each sample.
    return WindowBatch(
        sources=batched_sources,
        is_target=batched_is_target,
        sample_ids=[s.sample_id for s in samples],
        window_ref_times_utc=[s.window_ref_time_utc for s in samples],
        window_start_times_utc=[s.window_start_time_utc for s in samples],
        window_end_times_utc=[s.window_end_time_utc for s in samples],
        sids=[s.sid for s in samples],
        seasons=[s.season for s in samples],
        basins=[s.basin for s in samples],
        subbasins=[s.subbasin for s in samples],
        usa_atcf_ids=[s.usa_atcf_id for s in samples],
    )
