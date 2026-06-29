#!/usr/bin/env python3
"""Windows-setup data profiler — no model, no iteration over the dataset.

Characterizes the train/val/test samples produced by a given windows setup
straight from the long-format windows-index parquets
(``{preprocessed_data}/{windows_setup_name}/{split}_windows.parquet``), where
each row is one ``window_id x source_name x time_utc`` snapshot. Prints a
per-split summary (also saved as CSV) and saves a set of SVG figures describing
sample counts, temporal coverage, source availability, target balance, and
geographic / per-storm distribution.

Usage::

    # Local run
    python scripts/train/profile_data.py paths=local

    # Jean-Zay CPU via submitit
    python scripts/train/profile_data.py paths=jz setup=jz_cpu submitit=true
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Resolve project root so sibling-script imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocess.utils.runner import launch_local_or_slurm
from tcfuse.data.visualization.data_profile import (
    compute_split_summary,
    plot_basin_distribution,
    plot_sample_timeline,
    plot_samples_per_season,
    plot_source_availability,
    plot_sources_per_window_hist,
    plot_target_distribution,
    plot_windows_per_storm,
)

# Splits to profile, in display order.
_SPLITS = ("train", "val", "test")


# ── core run function ─────────────────────────────────────────────────────────


def _run(cfg: dict[str, Any]) -> None:
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    windows_setup_name = str(cfg["windows_setup"]["name"])
    setup_dir = assembled_root / windows_setup_name

    # Load each split's long-format windows index; skip any that are absent.
    split_frames: dict[str, pd.DataFrame] = {}
    for split in _SPLITS:
        index_path = setup_dir / f"{split}_windows.parquet"
        if not index_path.exists():
            print(f"[skip] {split}: {index_path} not found")
            continue
        split_frames[split] = pd.read_parquet(index_path)
        n_windows = split_frames[split]["window_id"].nunique()
        print(f"[load] {split}: {n_windows:,} samples from {index_path.name}")

    # Nothing to do when no split parquet was found.
    if not split_frames:
        print(f"No windows-index parquets found under {setup_dir}")
        return

    # Prepare the output directory for the CSV summary and SVG figures.
    # One output subdirectory per windows setup, under figures/profile/.
    figures_dir = Path(cfg["paths"]["figures"]) / "profile" / windows_setup_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Compute, print, and save the per-split summary table.
    summary = compute_split_summary(split_frames)
    print("\n=== Per-split summary ===")
    print(summary.to_string())
    summary_path = figures_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved -> {summary_path}")

    # Render every profiling figure as an SVG inside the setup directory.
    figure_builders = {
        "timeline": plot_sample_timeline,
        "season": plot_samples_per_season,
        "source_availability": plot_source_availability,
        "target_distribution": plot_target_distribution,
        "sources_per_window": plot_sources_per_window_hist,
        "basin": plot_basin_distribution,
        "windows_per_storm": plot_windows_per_storm,
    }
    for name, builder in figure_builders.items():
        out_path = figures_dir / name
        builder(split_frames, title=f"{windows_setup_name} — {name}", save_path=out_path)
        print(f"Figure saved -> {out_path.with_suffix('.svg')}")


# ── entry point ───────────────────────────────────────────────────────────────


@hydra.main(config_path="../../conf/", config_name="profile_data", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))
    launch_local_or_slurm(cfg, "profile_data", lambda: _run(cfg))


if __name__ == "__main__":
    main()
