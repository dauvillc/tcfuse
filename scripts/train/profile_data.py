#!/usr/bin/env python3
"""Data-loading profiler — mimics training+validation without any model.

Iterates every batch in the train and val splits and counts, for each
``(source_name, source_index)`` slot, how many samples have that source
available (i.e. at least one unmasked value).  Reports aggregate availability
fractions as a side-by-side figure.

Usage::

    # Local run
    python scripts/train/profile_data.py paths=local

    # Jean-Zay CPU via submitit
    python scripts/train/profile_data.py paths=jz setup=jz_cpu submitit=true
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Resolve project root so sibling-script imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocess.utils.runner import launch_local_or_slurm
from tcfuse.data.collate import WindowBatch
from tcfuse.lightning.datamodule import TCWindowDataModule

# ── helpers ──────────────────────────────────────────────────────────────────


def _count_availability(
    dataloader: Any,
    split: str,
) -> dict[tuple[str, int], dict[str, int]]:
    """Iterate ``dataloader`` and count source-slot availability per sample.

    Returns a dict mapping ``(source_name, source_index)`` to
    ``{"available": int, "total": int}``.
    """
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"available": 0, "total": 0}
    )

    for batch in tqdm(dataloader, desc=split):
        batch = cast(WindowBatch, batch)
        B = len(batch.sample_ids)

        for (name, idx), src in batch.sources.items():
            # mask shape: (B, ...) — True where data is present.
            available_per_sample = src.mask.reshape(B, -1).any(dim=-1)  # (B,) bool
            counts[(name, idx)]["available"] += int(available_per_sample.sum().item())
            counts[(name, idx)]["total"] += B

    return dict(counts)


def _plot(
    split_counts: dict[str, dict[tuple[str, int], dict[str, int]]],
    windows_setup_name: str,
    out_path: Path,
) -> None:
    """Render side-by-side horizontal bar charts, one per split, and save."""
    splits = list(split_counts.keys())

    # Collect all source keys across both splits, sorted for stable layout.
    all_keys: list[tuple[str, int]] = sorted(
        {key for counts in split_counts.values() for key in counts},
        key=lambda k: (k[0], k[1]),
    )

    # Assign a consistent color per unique source_name.
    source_names = list(dict.fromkeys(k[0] for k in all_keys))
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(source_names)}

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(7 * len(splits), max(3, 0.4 * len(all_keys) + 2)),
        sharey=True,
    )
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        counts = split_counts[split]

        def _frac(k: tuple[str, int]) -> float:
            slot = counts.get(k, {"available": 0, "total": 0})
            return slot["available"] / slot["total"] if slot["total"] > 0 else 0.0

        fractions = [_frac(k) for k in all_keys]
        labels = [f"{name} [{idx}]" for name, idx in all_keys]
        colors = [color_map[name] for name, _ in all_keys]
        total_samples = max((counts.get(k, {"total": 0})["total"] for k in all_keys), default=0)

        bars = ax.barh(labels, fractions, color=colors)
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Fraction of samples with source available")
        ax.set_title(f"Source availability — {split}\n({total_samples:,} samples)")

        # Annotate bars with percentage text.
        for bar, frac in zip(bars, fractions):
            ax.text(
                min(frac + 0.01, 1.02),
                bar.get_y() + bar.get_height() / 2,
                f"{frac:.1%}",
                va="center",
                ha="left",
                fontsize=8,
            )

    fig.suptitle(f"Windows setup: {windows_setup_name}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── core run function ─────────────────────────────────────────────────────────


def _run(cfg: dict[str, Any]) -> None:
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    windows_setup_name = str(cfg["windows_setup"]["name"])

    dm = TCWindowDataModule(
        assembled_root=assembled_root,
        windows_setup_name=windows_setup_name,
        dataloader_kwargs=cfg["dataloader"],
    )
    dm.setup("fit")

    split_counts: dict[str, dict[tuple[str, int], dict[str, int]]] = {}

    # Disable gradient tracking — no model involved.
    with torch.no_grad():
        split_counts["train"] = _count_availability(dm.train_dataloader(), "train")
        split_counts["val"] = _count_availability(dm.val_dataloader(), "val")

    figures_dir = Path(cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"profile_data_{windows_setup_name}.png"
    _plot(split_counts, windows_setup_name, out_path)


# ── entry point ───────────────────────────────────────────────────────────────


@hydra.main(config_path="../../conf/", config_name="profile_data", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))
    launch_local_or_slurm(cfg, "profile_data", lambda: _run(cfg))


if __name__ == "__main__":
    main()
