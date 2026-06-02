"""Custom collate function and batched sample container for TC window data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from tcfuse.data.dataset import WindowSample
from tcfuse.data.sources.source import Source


@dataclass
class WindowBatch:
    """Batched collection of window samples for use in a DataLoader.

    Holds data from B independent windows / storms. Scalar attributes are lists
    (one entry per sample). Source tensors are batched along a leading B axis.

    The ``sources`` dict maps ``(source_name, source_index)`` to a batched
    :class:`~tcfuse.data.sources.source.Source` (``batched=True``).
    ``source_index`` is the chronological rank of a snapshot within a single
    sample: if every sample has at most one snapshot of source "S", only
    ``("S", 0)`` will ever appear.  When a source appears K times in some
    sample, indices 0 … K-1 are all present.

    Samples that are missing a given ``(source_name, source_index)`` slot are
    represented as NaN tensors (values and coords all NaN, mask all False).

    Args:
        sources: Dict from ``(source_name, source_index)`` to a batched Source.
            Keys are sorted lexicographically for deterministic ordering.
        sample_ids: Window identifier strings, one per sample.
        init_times_utc: Assimilation anchor times ``t0``, one per sample.
        window_start_times_utc: Inclusive window lower bounds, one per sample.
        window_end_times_utc: Inclusive window upper bounds, one per sample.
        sids: IBTrACS storm identifiers, one per sample.
        seasons: TC season years, one per sample.
        basins: Ocean basin codes, one per sample.
        subbasins: IBTrACS sub-basin codes, one per sample.
        usa_atcf_ids: Optional USA ATCF identifiers, one per sample (None when absent).
        labels: Lead-time target columns from the split index, one Series per sample.
    """

    sources: dict[tuple[str, int], Source]
    sample_ids: list[str]
    init_times_utc: list[str]
    window_start_times_utc: list[str]
    window_end_times_utc: list[str]
    sids: list[str]
    seasons: list[int]
    basins: list[str]
    subbasins: list[str]
    usa_atcf_ids: list[str | None]
    labels: list[pd.Series] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch."""
        return len(self.sample_ids)


def collate_window_samples(samples: list[WindowSample]) -> WindowBatch:
    """Collate a list of WindowSamples into a single WindowBatch.

    Sources are merged across samples using a ``(source_name, source_index)``
    key space.  Within each sample, snapshots of the same source are ordered
    chronologically by ``snapshot_time_utc``; the earliest gets index 0.

    Samples that lack a particular ``(source_name, source_index)`` slot receive
    NaN-filled values and coords, and an all-False mask (``batched=True``).

    Args:
        samples: Non-empty list of WindowSamples from the same dataset.

    Returns:
        A :class:`WindowBatch` with all tensors on CPU and ``batched=True``
        on every Source.
    """
    # Step 1 — build a per-sample dict {(source_name, source_index): Source}.
    # Snapshots of the same source_name are sorted ascending by snapshot_time_utc
    # so that index 0 always corresponds to the earliest overpass.
    per_sample: list[dict[tuple[str, int], Source]] = []
    for sample in samples:
        # Sort the storm_data.sources items by snapshot time (second element of
        # the StormData key tuple).
        sorted_items = sorted(
            sample.storm_data.sources.items(),
            key=lambda kv: kv[0][1],  # kv[0] = (source_name, snapshot_time_utc)
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
    # The reference provides kind, channels, char_vars, and the canonical tensor
    # shapes — all guaranteed constant across samples for the same source_name.
    ref_source: dict[tuple[str, int], Source] = {}
    for indexed in per_sample:
        for key, source in indexed.items():
            if key not in ref_source:
                ref_source[key] = source

    # Step 4 — for each key, stack B per-sample tensors along dim 0.
    # Samples missing the key are NaN-filled (values/coords = NaN, mask = False).
    batched_sources: dict[tuple[str, int], Source] = {}
    for key in sorted(all_keys):
        ref = ref_source[key]

        values_list: list[torch.Tensor] = []
        coords_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        metas: list[dict] = []  # type: ignore[type-arg]

        for indexed in per_sample:
            if key in indexed:
                src = indexed[key]
                values_list.append(src.values)
                coords_list.append(src.coords)
                mask_list.append(src.mask)
                metas.append(src.meta)
            else:
                # NaN-fill: full_like preserves dtype and device automatically.
                values_list.append(torch.full_like(ref.values, float("nan")))
                coords_list.append(torch.full_like(ref.coords, float("nan")))
                # mask is bool — zeros_like gives all-False (all missing).
                mask_list.append(torch.zeros_like(ref.mask))
                metas.append({})

        # Stack along new leading batch dim → (B, ...) tensors.
        batched_sources[key] = Source(
            kind=ref.kind,
            values=torch.stack(values_list, dim=0),
            coords=torch.stack(coords_list, dim=0),
            source_name=ref.source_name,
            channels=ref.channels,
            mask=torch.stack(mask_list, dim=0),
            batched=True,
            # Store per-sample snapshot metas under a single list entry so the
            # dict[str, Any] contract of Source.meta is preserved.
            meta={"per_sample": metas},
            char_vars=ref.char_vars,
        )

    # Step 5 — collect the scalar list attributes from each sample.
    return WindowBatch(
        sources=batched_sources,
        sample_ids=[s.sample_id for s in samples],
        init_times_utc=[s.init_time_utc for s in samples],
        window_start_times_utc=[s.window_start_time_utc for s in samples],
        window_end_times_utc=[s.window_end_time_utc for s in samples],
        sids=[s.sid for s in samples],
        seasons=[s.season for s in samples],
        basins=[s.basin for s in samples],
        subbasins=[s.subbasin for s in samples],
        usa_atcf_ids=[s.usa_atcf_id for s in samples],
        labels=[s.labels for s in samples],
    )
