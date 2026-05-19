#!/usr/bin/env python3
"""Compute per-channel normalization statistics for every source in training storms.

Statistics are computed **exclusively from source snapshots belonging to storms that
appear in the training window index**.  ``train.parquet`` is produced by
``scripts/preprocess/build_splits.py`` and contains one row per model sample, while
``index.parquet`` remains the canonical one-row-per-source-snapshot table.

Because the full assembled dataset cannot fit in memory, statistics are computed with
Welford's online algorithm using batched parallel updates.  Histogram samples are
collected with a simple truncated reservoir (up to SAMPLE_CAP values per channel).

Per-source intermediate YAML files are written to ``{preprocessed_data}/normalization/``
so each submitit job is independently checkpointable.  After all jobs complete the main
function merges them into a single ``{preprocessed_data}/normalization_stats.yaml``.

Run from the project root:
    python scripts/preprocess/compute_normalization.py [paths=jz] [submitit=false]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import h5py
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.sources.storm_data import StormData

# Use a non-interactive backend so the script works on headless compute nodes.
matplotlib.use("Agg")

# Maximum number of histogram samples retained per channel (across all snapshots).
SAMPLE_CAP: int = 100_000
# Maximum number of values drawn from a single snapshot for the histogram reservoir.
K_PER_BATCH: int = 1_000


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _compact_time(snapshot_time_utc: str) -> str:
    """Convert an isoformat timestamp to the compact HDF5 group key.

    Mirrors ``storm_data._to_compact_time``.

    Args:
        snapshot_time_utc: ISO 8601 string, e.g. ``"2016-09-12T01:09:42+00:00"``.

    Returns:
        Compact string without separators, e.g. ``"20160912T010942Z"``.
    """
    return pd.Timestamp(snapshot_time_utc).strftime("%Y%m%dT%H%M%SZ")


def load_training_snapshot_index(assembled_root: Path) -> pd.DataFrame:
    """Load canonical snapshot rows for storms present in the training split.

    Args:
        assembled_root: Root directory containing ``index.parquet`` and
            ``train.parquet``.

    Returns:
        DataFrame with one row per source snapshot from training storms.
    """
    train_path = assembled_root / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found at {train_path}. "
            "Run scripts/preprocess/build_splits.py first."
        )

    print(f"Loading training window index from {train_path} …")
    train_samples = pd.read_parquet(train_path)
    if "storm_id" not in train_samples.columns:
        raise ValueError(f"Training split at {train_path} must contain a storm_id column.")

    index_path = assembled_root / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Assembled index not found at {index_path}. Run scripts/preprocess/assemble.py first."
        )

    print(f"Loading canonical snapshot index from {index_path} …")
    snapshot_index = pd.read_parquet(index_path)
    train_storm_ids = set(train_samples["storm_id"].astype(str))
    training_snapshots = snapshot_index[
        snapshot_index["storm_id"].astype(str).isin(train_storm_ids)
    ].copy()

    # Multiple samples from one storm reference the same source snapshots; keep
    # each snapshot exactly once for normalization.
    return training_snapshots.drop_duplicates(
        subset=["storm_id", "source_name", "snapshot_time_utc"]
    ).reset_index(drop=True)


def _flatten_values_and_mask(
    values: np.ndarray,
    mask: np.ndarray,
    kind: SourceKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a source snapshot to values and per-value availability arrays.

    Args:
        values: Raw values array.  Shape ``(C,)`` for SCALAR, ``(L, C)`` for PROFILE,
            ``(H, W, C)`` for FIELD.
        mask: Boolean per-value availability mask. Same shape as ``values``;
            True means available.
        kind: Source dimensionality.

    Returns:
        Tuple ``(flat_values, flat_mask)`` where both arrays have shape ``(N, C)``.
    """
    # Reshape values to (N, C) depending on source kind.
    if kind == SourceKind.SCALAR:
        flat = values.reshape(1, -1).astype(np.float32)
    elif kind == SourceKind.PROFILE:
        flat = values.astype(np.float32)
    else:  # FIELD
        h, w, c = values.shape
        flat = values.reshape(h * w, c).astype(np.float32)

    if mask.shape != values.shape:
        raise ValueError(
            f"mask shape {mask.shape} must match values shape {values.shape} "
            f"for {kind.name} source"
        )

    if kind == SourceKind.SCALAR:
        flat_mask = mask.reshape(1, -1).astype(bool)
    elif kind == SourceKind.PROFILE:
        flat_mask = mask.astype(bool)
    else:  # FIELD
        flat_mask = mask.reshape(flat.shape).astype(bool)

    return flat, flat_mask & np.isfinite(flat)


def _welford_update(
    count: float,
    mean: float,
    m2: float,
    batch: np.ndarray,
) -> tuple[float, float, float]:
    """Update Welford accumulators with a 1-D batch of new values.

    Uses the parallel combination formula so the entire batch is ingested
    in O(1) passes rather than element-by-element.

    Args:
        count: Running count of values seen so far.
        mean: Running mean.
        m2: Running sum of squared deviations from the mean.
        batch: 1-D float array of new values.  Must be non-empty.

    Returns:
        Updated ``(count, mean, m2)`` triple.
    """
    n_b = len(batch)
    if n_b == 0:
        return count, mean, m2

    # Compute statistics of the incoming batch alone.
    mean_b = float(batch.mean())
    m2_b = float(((batch - mean_b) ** 2).sum())

    # Combine existing state with the new batch.
    combined_count = count + n_b
    delta = mean_b - mean
    combined_mean = mean + delta * n_b / combined_count
    combined_m2 = m2 + m2_b + delta**2 * count * n_b / combined_count

    return combined_count, combined_mean, combined_m2


def _reservoir_add(samples: list[float], new_vals: np.ndarray) -> None:
    """Append values to a truncated histogram reservoir in-place.

    Draws up to K_PER_BATCH values at random from ``new_vals`` and appends
    them to ``samples`` until SAMPLE_CAP is reached.  Once the cap is hit no
    further values are added (the resulting sample is biased toward early
    snapshots but sufficient for distribution visualisation).

    Args:
        samples: Mutable list accumulating histogram samples (modified in-place).
        new_vals: 1-D array of candidate new values.
    """
    # Do nothing once the reservoir is full.
    remaining = SAMPLE_CAP - len(samples)
    if remaining <= 0 or len(new_vals) == 0:
        return

    # Randomly subsample the batch to at most K_PER_BATCH values.
    if len(new_vals) > K_PER_BATCH:
        idx = np.random.choice(len(new_vals), K_PER_BATCH, replace=False)
        new_vals = new_vals[idx]

    # Append as many values as will fit.
    take = min(len(new_vals), remaining)
    samples.extend(new_vals[:take].tolist())


def _plot_source(
    source_name: str,
    channels: list[str],
    kind: SourceKind,
    samples_per_channel: list[list[float]],
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    figures_dir: Path,
) -> None:
    """Save a histogram figure for one source with one subplot per channel.

    Args:
        source_name: Source identifier used as figure title and filename stem.
        channels: Ordered channel names (length C).
        kind: Source dimensionality (SCALAR/PROFILE/FIELD).
        samples_per_channel: List of C sample lists for the histograms.
        means: Per-channel means, shape ``(C,)``.
        stds: Per-channel standard deviations, shape ``(C,)``.
        counts: Per-channel valid-value counts, shape ``(C,)``.
        figures_dir: Root figures directory; output is written to
            ``{figures_dir}/normalization/{source_name}.png``.
    """
    c = len(channels)
    # Arrange subplots in a 2-column grid.
    n_cols = min(2, c)
    n_rows = math.ceil(c / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for i, ch in enumerate(channels):
        ax = axes[i // n_cols][i % n_cols]
        samps = np.array(samples_per_channel[i], dtype=np.float32)

        if len(samps) > 0:
            # Plot histogram with log-scale y-axis.
            ax.hist(samps, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
            ax.set_yscale("log")

            # Mark mean ± 1 std.
            mu, sigma = float(means[i]), float(stds[i])
            ax.axvline(mu, color="crimson", linewidth=1.5, label=f"μ = {mu:.4g}")
            ax.axvline(mu - sigma, color="crimson", linewidth=1.0, linestyle="--")
            ax.axvline(mu + sigma, color="crimson", linewidth=1.0, linestyle="--")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        # Subtitle with channel name and key statistics.
        ax.set_title(
            f"{ch}  |  μ={means[i]:.4g}  sigma={stds[i]:.4g}  n={int(counts[i]):,}",
            fontsize=9,
        )
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    # Hide unused subplot axes (when C is odd with 2 columns).
    for j in range(c, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    # Overall figure title.
    fig.suptitle(f"{source_name}  ({kind.name.lower()})", fontsize=12, fontweight="bold")
    fig.tight_layout()

    # Save to figures/normalization/.
    out_dir = figures_dir / "normalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Per-source job (must be a module-level function for submitit pickling)
# ---------------------------------------------------------------------------


def process_source(
    source_name: str,
    rows: pd.DataFrame,
    assembled_root: Path,
    channels_hint: list[str] | None,
    out_dir: Path,
    figures_dir: Path,
) -> dict[str, Any] | None:
    """Compute normalization statistics for one source and save results.

    Iterates over every snapshot of ``source_name`` in the assembled HDF5 files,
    updating per-channel Welford accumulators and collecting histogram samples.
    Writes a per-source YAML file and a histogram PNG at the end.

    This function is the unit of work submitted to submitit.  It must remain a
    top-level, pickleable function (no closures capturing local state).

    Args:
        source_name: Name of the source to process, e.g. ``"pmw_amsr2_gcomw1"``.
        rows: Rows from the assembled ``index.parquet`` for this source; each row
            provides ``storm_id`` and ``snapshot_time_utc``.
        assembled_root: Root directory of assembled storm HDF5 files
            (``cfg.paths.preprocessed_data``).
        channels_hint: Ordered channel names from :class:`MultisourceMetadata`
            (for sources present in ``sources_root``), or ``None`` for sources
            like ``ibtracs_best_track`` that are injected at assembly time.
        out_dir: Directory for per-source YAML outputs.
        figures_dir: Root figures directory.

    Returns:
        Stats dict ``{kind: str, channels: {ch: {mean, std, count}}}`` on success,
        or ``None`` if no valid snapshots were found for this source.
    """
    print(f"\n[{source_name}] {len(rows)} snapshots")

    # --- Discover channels and kind from the first available HDF5 snapshot ---
    channels: list[str] | None = channels_hint
    kind: SourceKind | None = None

    for _, row in rows.iterrows():
        storm_id = str(row["storm_id"])
        snap_time = str(row["snapshot_time_utc"])
        compact = _compact_time(snap_time)
        storm_path = StormData.path(assembled_root, storm_id)

        if not storm_path.exists():
            continue
        try:
            with h5py.File(storm_path, "r") as f:
                if source_name not in f:
                    continue
                src_grp = cast(h5py.Group, f[source_name])
                if compact not in src_grp:
                    continue
                grp = cast(h5py.Group, src_grp[compact])
                kind = SourceKind[str(grp.attrs["kind"])]
                if channels is None:
                    channels = json.loads(str(grp.attrs["channels"]))
            break  # found what we need
        except Exception:
            continue

    if channels is None or kind is None:
        print(f"  [WARN] Could not discover channels/kind for {source_name}. Skipping.")
        return None

    c = len(channels)

    # Welford state: one accumulator per channel.
    counts = np.zeros(c, dtype=np.float64)
    means = np.zeros(c, dtype=np.float64)
    m2s = np.zeros(c, dtype=np.float64)

    # Histogram reservoir: one list per channel.
    samples: list[list[float]] = [[] for _ in range(c)]

    # --- Iterate over all snapshots for this source ---
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=source_name, leave=False):
        storm_id = str(row["storm_id"])
        snap_time = str(row["snapshot_time_utc"])
        compact = _compact_time(snap_time)
        storm_path = StormData.path(assembled_root, storm_id)

        if not storm_path.exists():
            continue

        try:
            with h5py.File(storm_path, "r") as f:
                # Skip if this source/snapshot is absent (can happen if assembly was partial).
                if source_name not in f:
                    continue
                src_grp = cast(h5py.Group, f[source_name])
                if compact not in src_grp:
                    continue
                grp = cast(h5py.Group, src_grp[compact])

                # Read values and optional mask as numpy arrays.
                values: np.ndarray = cast(h5py.Dataset, grp["values"])[:]
                if "mask" not in grp:
                    raise ValueError(
                        f"{storm_id}/{compact}/{source_name} is missing mandatory mask dataset."
                    )
                mask: np.ndarray = cast(h5py.Dataset, grp["mask"])[:]
        except Exception as exc:
            print(f"  [WARN] Failed to read {storm_id}/{compact}: {exc}")
            continue

        # Flatten to (N, C), preserving per-channel availability.
        flat_values, flat_mask = _flatten_values_and_mask(values, mask, kind)
        if flat_values.shape[0] == 0:
            continue

        # Update Welford accumulators and histogram reservoir for each channel.
        for ci in range(c):
            col = flat_values[:, ci][flat_mask[:, ci]]
            if len(col) == 0:
                continue
            counts[ci], means[ci], m2s[ci] = _welford_update(counts[ci], means[ci], m2s[ci], col)
            _reservoir_add(samples[ci], col)

    if counts.max() == 0:
        print(f"  [WARN] No valid values found for {source_name}. Skipping.")
        return None

    # Compute population standard deviation from the m2 accumulators.
    stds = np.sqrt(m2s / np.maximum(counts, 1.0))

    # Generate and save histogram figure.
    _plot_source(source_name, channels, kind, samples, means, stds, counts, figures_dir)

    # Build per-channel stats dict.
    channel_stats = {
        ch: {
            "mean": float(means[ci]),
            "std": float(stds[ci]),
            "count": int(counts[ci]),
        }
        for ci, ch in enumerate(channels)
    }
    source_stats: dict[str, Any] = {
        "kind": kind.name.lower(),
        "channels": channel_stats,
    }

    # Write intermediate per-source YAML.
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{source_name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({source_name: source_stats}, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote stats → {yaml_path}")

    return source_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Compute normalization statistics for all sources in the assembled dataset."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    # Resolve key paths from config.
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    figures_dir = Path("figures")
    out_dir = assembled_root / "normalization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source snapshots belonging to storms present in the training samples.
    index = load_training_snapshot_index(assembled_root)
    groups: dict[str, pd.DataFrame] = {str(k): v for k, v in index.groupby("source_name")}
    print(f"  Found {len(groups)} source(s): {sorted(groups)}")

    # Load channel hints for sources that have metadata.yaml in sources_root.
    # Sources injected at assembly time (e.g. ibtracs_best_track) get None.
    multi_meta = MultisourceMetadata.from_disk(sources_root)
    channels_hints: dict[str, list[str] | None] = {
        sn: (multi_meta[sn].channels if sn in multi_meta else None) for sn in groups
    }

    launch_local = not bool(cfg.get("submitit", False))

    # --- Local execution: process sources sequentially ---
    if launch_local:
        results: dict[str, dict[str, Any] | None] = {}
        for source_name, rows in groups.items():
            results[source_name] = process_source(
                source_name,
                rows,
                assembled_root,
                channels_hints[source_name],
                out_dir,
                figures_dir,
            )

    # --- SLURM execution: one job per source ---
    else:
        from tcfuse.utils.submitit_utils import make_executor

        executor = make_executor(cfg, "compute_normalization")
        jobs: dict[str, Any] = {}

        # Submit one job per source.
        for source_name, rows in groups.items():
            print(f"Submitting job for {source_name} ({len(rows)} snapshots) …")
            jobs[source_name] = executor.submit(
                process_source,
                source_name,
                rows,
                assembled_root,
                channels_hints[source_name],
                out_dir,
                figures_dir,
            )

        # Collect results, blocking until each job finishes.
        results = {}
        for source_name, job in tqdm(jobs.items(), desc="collecting results"):
            try:
                results[source_name] = job.result()
            except Exception as exc:
                print(f"  [ERROR] Job for {source_name} failed: {exc}")
                results[source_name] = None

    # Merge all per-source stats into one final file.
    merged: dict[str, Any] = {sn: stats for sn, stats in results.items() if stats is not None}
    stats_path = assembled_root / "normalization_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=True)
    print(f"\nWrote normalization stats → {stats_path}")

    # Print a brief summary table.
    print(f"\n{'Source':<40}  {'Kind':<8}  {'Channels':>8}  {'Status'}")
    print("-" * 70)
    for sn in sorted(groups):
        stats = results.get(sn)
        if stats is not None:
            nc = len(stats["channels"])
            print(f"{sn:<40}  {stats['kind']:<8}  {nc:>8}  OK")
        else:
            print(f"{sn:<40}  {'—':<8}  {'—':>8}  FAILED")


if __name__ == "__main__":
    main()
