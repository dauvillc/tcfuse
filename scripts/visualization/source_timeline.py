#!/usr/bin/env python3
"""Plot the temporal availability of each observation source in the assembled index.

Reads ``index.parquet`` from the assembled data directory and produces a
publication-quality SVG timeline saved alongside the index.

Run from the project root:
    python scripts/visualization/source_timeline.py [paths=jz]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from tcfuse.data.visualization.timeline import plot_source_timeline


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Load the assembled index and render the source availability timeline."""
    cfg = cast(dict[str, Any], OmegaConf.to_container(raw_cfg, resolve=True))

    # Locate the assembled index.
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    index_path = assembled_root / "index.parquet"
    if not index_path.exists():
        print(f"ERROR: assembled index not found at {index_path}")
        return

    # Load and summarise the index.
    index_df = pd.read_parquet(index_path)
    n_rows = len(index_df)
    sources = sorted(index_df["source_name"].unique())
    times = pd.to_datetime(index_df["snapshot_time_utc"], utc=True)
    print(
        f"Loaded index: {n_rows:,} rows, {len(sources)} sources, "
        f"{times.min().date()} → {times.max().date()}"
    )
    print("Sources found:")
    for name in sources:
        count = (index_df["source_name"] == name).sum()
        print(f"  {name}: {count:,} snapshots")

    # Render and save the timeline figure.
    output_path = Path("figures") / "source_timeline"
    print(f"Rendering timeline → {output_path}.svg")
    plot_source_timeline(index_df, save_path=output_path)

    print("Done.")


if __name__ == "__main__":
    main()
