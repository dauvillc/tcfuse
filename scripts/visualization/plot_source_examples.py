#!/usr/bin/env python3
"""Plot one example snapshot per Stage 1 preprocessed source (all channels).

Scans ``cfg.paths.preprocessed_sources`` for PMW, infrared, radar, and SAR
directories, loads the first indexed snapshot per source, and writes multi-panel
SVG figures under ``figures/source_examples/``.

Run from the project root::

    TCFUSE_NO_LATEX=1 python scripts/visualization/plot_source_examples.py paths=local

Optional overrides::

    output_dir=figures/source_examples
    sources_root=/path/to/preprocessed/sources
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from tcfuse.data.sources import Source, SourceMetadata
from tcfuse.data.visualization.fields import ChannelPlotSpec, plot_field_source_channels
from tcfuse.data.visualization.style import UNIT_K, UNIT_M_S, UNIT_MM_H, save_fig
from tcfuse.utils.time import to_compact_time

PMW_PREFIX = "pmw_"
IR_NAMES = ("ir_tcirar", "ir_hursat")
RADAR_NAMES = ("radar_gmi", "radar_tmi")
SAR_NAME = "sar_cband"

# Radar channel name fragments (lowercase, as stored in metadata) → plot spec
_RADAR_CHANNEL_SPECS: dict[str, ChannelPlotSpec] = {
    "nearsurfpreciptotrate": ChannelPlotSpec("precip", UNIT_MM_H),
    "nearsurfpreciptotratesigma": ChannelPlotSpec("anomaly", UNIT_MM_H),
    "mainprecipitationtype": ChannelPlotSpec("anomaly", ""),
}


def _is_ready_source_dir(path: Path) -> bool:
    """Return True when a source directory has metadata and a non-empty index."""
    if not path.is_dir():
        return False
    meta = path / "metadata.yaml"
    index = path / "index.parquet"
    return meta.is_file() and index.is_file()


def discover_source_names(sources_root: Path) -> list[str]:
    """Collect PMW, IR, radar, and SAR source directory names present on disk."""
    names: list[str] = []

    # All preprocessed PMW sensors (dynamic list)
    for child in sorted(sources_root.iterdir()):
        if child.name.startswith(PMW_PREFIX) and _is_ready_source_dir(child):
            names.append(child.name)

    # Fixed IR / radar / SAR source names
    for name in (*IR_NAMES, *RADAR_NAMES, SAR_NAME):
        if _is_ready_source_dir(sources_root / name):
            names.append(name)
        else:
            print(f"Skipping missing or empty source: {name}")

    return names


def channel_specs_for_source(meta: SourceMetadata) -> list[ChannelPlotSpec]:
    """Build per-channel plot specs from source type and channel names."""
    name = meta.name

    if name.startswith(PMW_PREFIX) or name.startswith("ir_"):
        return [ChannelPlotSpec("tb", UNIT_K) for _ in meta.channels]

    if name.startswith("radar_"):
        specs: list[ChannelPlotSpec] = []
        for ch in meta.channels:
            ch_lower = ch.lower()
            matched = next(
                (spec for key, spec in _RADAR_CHANNEL_SPECS.items() if key in ch_lower),
                ChannelPlotSpec("precip", ""),
            )
            specs.append(matched)
        return specs

    if name == SAR_NAME:
        return [ChannelPlotSpec("sar_wind", UNIT_M_S)]

    raise ValueError(f"Unsupported source for gallery plotting: {name}")


def example_snapshot_path(sources_root: Path, meta: SourceMetadata) -> Path | None:
    """Return the HDF5 path for the first index row, or None when the index is empty."""
    index_path = sources_root / meta.name / "index.parquet"
    index = pd.read_parquet(index_path)
    if index.empty:
        return None
    row = index.sort_values(["sid", "time_utc"]).iloc[0]
    sid = str(row["sid"])
    time_utc = to_compact_time(str(row["time_utc"]))
    path = Source.path(sources_root, meta.name, sid, time_utc)
    if not path.is_file():
        print(f"WARNING: index points to missing file: {path}")
        return None
    return path


def suptitle_for_source(source: Source) -> str:
    """Compose a figure suptitle from source name and snapshot metadata."""
    snapshot_time = source.meta.get("time_utc", "")
    return f"{source.source_name}  {snapshot_time}"


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Discover sources, plot one example per source, and save SVGs."""
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))

    sources_root = Path(cfg.get("sources_root", cfg["paths"]["preprocessed_sources"]))
    output_dir = Path(cfg.get("output_dir", "figures/source_examples"))

    if not sources_root.is_dir():
        print(f"ERROR: preprocessed sources root not found: {sources_root}")
        return

    source_names = discover_source_names(sources_root)
    if not source_names:
        print(f"ERROR: no plottable sources under {sources_root}")
        return

    print(f"Sources root: {sources_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Plotting {len(source_names)} sources…")

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0

    for source_name in source_names:
        meta = SourceMetadata.from_yaml(sources_root / source_name / "metadata.yaml")
        snapshot_path = example_snapshot_path(sources_root, meta)
        if snapshot_path is None:
            print(f"  SKIP {source_name}: no snapshots")
            continue

        source = Source.from_disk(snapshot_path)
        specs = channel_specs_for_source(meta)
        out_path = output_dir / source_name

        print(f"  {source_name} ← {snapshot_path.name}")
        fig, _axes = plot_field_source_channels(
            source,
            specs,
            suptitle=suptitle_for_source(source),
        )
        save_fig(fig, out_path)
        plt.close(fig)
        plotted += 1

    print(f"Done. Wrote {plotted} figures to {output_dir}/")


if __name__ == "__main__":
    main()
