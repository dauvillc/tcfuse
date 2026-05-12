#!/usr/bin/env python3
"""Split the assembled index into train/val/test parquet files by season.

Reads ``{preprocessed_data}/index.parquet`` produced by ``assemble.py`` and
writes three parquet files — ``train.parquet``, ``val.parquet``,
``test.parquet`` — to the same directory.  Season assignment uses the
``season`` column of the index, which holds a single value per storm lifetime
even when the storm spans two calendar years.

Split seasons are read from ``cfg.splits`` (``conf/preproc.yaml``):
  - **val**:   seasons listed under ``splits.val``
  - **test**:  seasons listed under ``splits.test``
  - **train**: all remaining seasons

Run from the project root:
    python scripts/preprocess/build_splits.py [paths=jz]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Build season-based train/val/test split parquet files."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    # Resolve paths from config.
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    index_path = assembled_root / "index.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Assembled index not found at {index_path}. Run scripts/preprocess/assemble.py first."
        )

    # Read season lists from config; convert to sets for O(1) lookup.
    val_seasons: set[int] = set(cfg["splits"]["val"])
    test_seasons: set[int] = set(cfg["splits"]["test"])

    # Validate that val and test season sets are disjoint.
    overlap = val_seasons & test_seasons
    if overlap:
        raise ValueError(
            f"Val and test season sets overlap: {sorted(overlap)}. "
            "Fix cfg.splits in conf/preproc.yaml."
        )

    # Load the global assembled index.
    print(f"Loading index from {index_path} …")
    index = pd.read_parquet(index_path)
    print(f"  {len(index)} rows, {index['storm_id'].nunique()} unique storms.")

    # Build boolean masks — test takes priority over val to keep sets disjoint.
    test_mask = index["season"].isin(test_seasons)
    val_mask = index["season"].isin(val_seasons) & ~test_mask
    train_mask = ~test_mask & ~val_mask

    # Slice into split DataFrames.
    splits: dict[str, pd.DataFrame] = {
        "train": index[train_mask].reset_index(drop=True),
        "val": index[val_mask].reset_index(drop=True),
        "test": index[test_mask].reset_index(drop=True),
    }

    # Write each split and print a summary row.
    print(f"\n{'Split':<8}  {'Storms':>7}  {'Rows':>9}  {'Seasons'}")
    print("-" * 55)
    for split_name, df in splits.items():
        out_path = assembled_root / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        seasons_str = ", ".join(str(s) for s in sorted(df["season"].unique()))
        print(f"{split_name:<8}  {df['storm_id'].nunique():>7}  {len(df):>9}  {seasons_str}")

    # Sanity check: every row in the source index appears in exactly one split.
    total = sum(len(df) for df in splits.values())
    assert total == len(index), (
        f"Row count mismatch after splitting: {total} != {len(index)}. "
        "This is a bug in build_splits.py."
    )
    print(f"\nWrote train/val/test parquet files to {assembled_root}")


if __name__ == "__main__":
    main()
