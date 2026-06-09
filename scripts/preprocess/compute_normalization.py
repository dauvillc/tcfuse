#!/usr/bin/env python3
"""Compute per-channel normalization statistics for every source in training storms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import h5py
import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.preprocess.utils.runner import resolve_preproc_cfg
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.storm_data import StormData
from tcfuse.utils.time import to_compact_time

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_training_snapshot_index(assembled_root: Path) -> pd.DataFrame:
    """Load canonical snapshot rows for storms present in the training split."""
    train_path = assembled_root / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found at {train_path}. "
            "Run scripts/preprocess/build_splits.py first."
        )

    print(f"Loading training window index from {train_path} …")
    train_samples = pd.read_parquet(train_path)
    if "sid" not in train_samples.columns:
        raise ValueError(f"Training split at {train_path} must contain a sid column.")

    index_path = assembled_root / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Assembled index not found at {index_path}. Run scripts/preprocess/assemble.py first."
        )

    print(f"Loading canonical snapshot index from {index_path} …")
    snapshot_index = pd.read_parquet(index_path)
    train_sids = list(train_samples["sid"].astype(str).unique())
    training_snapshots = snapshot_index[snapshot_index["sid"].astype(str).isin(train_sids)].copy()

    return training_snapshots.drop_duplicates(
        subset=["sid", "source_name", "time_utc"]
    ).reset_index(drop=True)


def _flatten_values_and_mask(
    values: np.ndarray,
    mask: np.ndarray,
    kind: SourceKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a source snapshot to values and per-value availability arrays."""
    if kind == SourceKind.SCALAR:
        flat = values.reshape(1, -1).astype(np.float32)
    elif kind == SourceKind.PROFILE:
        flat = values.astype(np.float32)
    else:
        # FIELD: flatten spatial tokens to (H*W, C) for per-pixel normalization stats.
        h, w, _ = values.shape
        flat = values.reshape(h * w, -1).astype(np.float32)

    if mask.shape != values.shape:
        raise ValueError(
            f"mask shape {mask.shape} must match values shape {values.shape} for {kind.name} source"
        )

    if kind == SourceKind.SCALAR:
        flat_mask = mask.reshape(1, -1).astype(bool)
    elif kind == SourceKind.PROFILE:
        flat_mask = mask.astype(bool)
    else:
        flat_mask = mask.reshape(flat.shape).astype(bool)

    # Combine explicit availability mask with finiteness for Welford accumulation.
    return flat, flat_mask & np.isfinite(flat)


def _welford_update(
    count: float,
    mean: float,
    m2: float,
    batch: np.ndarray,
) -> tuple[float, float, float]:
    """Update Welford accumulators with a 1-D batch of new values."""
    n_b = len(batch)
    if n_b == 0:
        return count, mean, m2

    mean_b = float(batch.mean())
    m2_b = float(((batch - mean_b) ** 2).sum())
    combined_count = count + n_b
    delta = mean_b - mean
    combined_mean = mean + delta * n_b / combined_count
    combined_m2 = m2 + m2_b + delta**2 * count * n_b / combined_count
    return combined_count, combined_mean, combined_m2


def process_source(
    source_name: str,
    rows: pd.DataFrame,
    assembled_root: Path,
    channels_hint: list[str] | None,
) -> dict[str, Any] | None:
    """Compute normalization statistics for one source."""
    print(f"\n[{source_name}] {len(rows)} snapshots")

    channels: list[str] | None = channels_hint
    kind: SourceKind | None = None

    for _, row in rows.iterrows():
        sid = str(row["sid"])
        snap_time = str(row["time_utc"])
        compact = to_compact_time(snap_time)
        storm_path = StormData.path(assembled_root, sid)
        if not storm_path.exists():
            continue
        try:
            with h5py.File(storm_path, "r") as storm_file:
                if source_name not in storm_file:
                    continue
                src_grp = cast(h5py.Group, storm_file[source_name])
                if compact not in src_grp:
                    continue
                grp = cast(h5py.Group, src_grp[compact])
                kind = SourceKind[str(grp.attrs["kind"])]
                if channels is None:
                    channels = json.loads(str(grp.attrs["channels"]))
            break
        except Exception:
            continue

    if channels is None or kind is None:
        print(f"  [WARN] Could not discover channels/kind for {source_name}. Skipping.")
        return None

    c = len(channels)
    counts = np.zeros(c, dtype=np.float64)
    means = np.zeros(c, dtype=np.float64)
    m2s = np.zeros(c, dtype=np.float64)

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=source_name, leave=False):
        sid = str(row["sid"])
        snap_time = str(row["time_utc"])
        compact = to_compact_time(snap_time)
        storm_path = StormData.path(assembled_root, sid)
        if not storm_path.exists():
            continue

        try:
            with h5py.File(storm_path, "r") as storm_file:
                if source_name not in storm_file:
                    continue
                src_grp = cast(h5py.Group, storm_file[source_name])
                if compact not in src_grp:
                    continue
                grp = cast(h5py.Group, src_grp[compact])
                values: np.ndarray = cast(h5py.Dataset, grp["values"])[:]
                if "mask" not in grp:
                    raise ValueError(
                        f"{sid}/{compact}/{source_name} is missing mandatory mask dataset."
                    )
                mask: np.ndarray = cast(h5py.Dataset, grp["mask"])[:]
        except Exception as exc:
            print(f"  [WARN] Failed to read {sid}/{compact}: {exc}")
            continue

        flat_values, flat_mask = _flatten_values_and_mask(values, mask, kind)
        if flat_values.shape[0] == 0:
            continue

        for channel_index in range(c):
            col = flat_values[:, channel_index][flat_mask[:, channel_index]]
            if len(col) == 0:
                continue
            counts[channel_index], means[channel_index], m2s[channel_index] = _welford_update(
                counts[channel_index],
                means[channel_index],
                m2s[channel_index],
                col,
            )

    if counts.max() == 0:
        print(f"  [WARN] No valid values found for {source_name}. Skipping.")
        return None

    stds = np.sqrt(m2s / np.maximum(counts, 1.0))
    channel_stats = {
        ch: {
            "mean": float(means[channel_index]),
            "std": float(stds[channel_index]),
            "count": int(counts[channel_index]),
        }
        for channel_index, ch in enumerate(channels)
    }
    return {"kind": kind.name.lower(), "channels": channel_stats}


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Compute normalization statistics for all sources in the assembled dataset."""
    cfg = resolve_preproc_cfg(raw_cfg)
    assembled_root = Path(cfg["paths"]["preprocessed_data"])

    index = load_training_snapshot_index(assembled_root)
    groups: dict[str, pd.DataFrame] = {str(k): v for k, v in index.groupby("source_name")}
    print(f"  Found {len(groups)} source(s): {sorted(groups)}")

    meta_path = assembled_root / "sources_metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Sources metadata not found at {meta_path}. Run scripts/preprocess/assemble.py first."
        )
    multi_meta = MultisourceMetadata.from_yaml(meta_path)
    channels_hints: dict[str, list[str] | None] = {
        sn: (multi_meta[sn].channels if sn in multi_meta else None) for sn in groups
    }

    if cfg.get("submitit", False):
        from tcfuse.utils.submitit_utils import make_executor

        jobs = {}
        for source_name, rows in groups.items():
            slurm_name = f"norm_{source_name}"
            executor = make_executor(cfg, slurm_name)
            jobs[source_name] = executor.submit(
                process_source,
                source_name,
                rows,
                assembled_root,
                channels_hints[source_name],
            )
        results = {}
        for source_name, job in tqdm(jobs.items(), desc="collecting results"):
            try:
                results[source_name] = job.result()
            except Exception as exc:
                print(f"  [ERROR] Job for {source_name} failed: {exc}")
                results[source_name] = None
    else:
        results = {
            source_name: process_source(
                source_name,
                rows,
                assembled_root,
                channels_hints[source_name],
            )
            for source_name, rows in groups.items()
        }

    merged: dict[str, Any] = {sn: stats for sn, stats in results.items() if stats is not None}
    stats_path = assembled_root / "normalization_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=True)
    print(f"\nWrote normalization stats → {stats_path}")


if __name__ == "__main__":
    main()
