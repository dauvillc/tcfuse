#!/usr/bin/env python3
"""Stage 3A — split the assembled source index by season.

Reads ``{preprocessed_data}/index.parquet`` produced by ``assemble.py`` and
writes three index parquet files — ``train.parquet``, ``val.parquet``,
``test.parquet`` — to the same directory.  Each output file has the same
uniform schema as the assembled index (one row per source snapshot) and can be
consumed directly or used as input for ``build_windows.py``.

Split seasons are read from ``cfg.splits`` (``conf/preproc.yaml``):
  - **val**:   seasons listed under ``splits.val``
  - **test**:  seasons listed under ``splits.test``
  - **train**: all remaining seasons

Run from the project root:
    python scripts/preprocess/build_splits.py [paths=jz]
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from scripts.preprocess.utils.runner import resolve_preproc_cfg


def split_by_season(
    index: pd.DataFrame,
    val_seasons: set[int],
    test_seasons: set[int],
) -> dict[str, pd.DataFrame]:
    """Partition index rows into train/val/test subsets by season."""
    overlap = val_seasons & test_seasons
    if overlap:
        raise ValueError(
            f"Val and test season sets overlap: {sorted(overlap)}. "
            "Fix cfg.splits in conf/preproc.yaml."
        )

    if index.empty:
        return {
            "train": index.copy(),
            "val": index.copy(),
            "test": index.copy(),
        }

    test_mask = index["season"].isin(list(test_seasons))
    val_mask = index["season"].isin(list(val_seasons)) & ~test_mask
    train_mask = ~test_mask & ~val_mask
    return {
        "train": index[train_mask].reset_index(drop=True),
        "val": index[val_mask].reset_index(drop=True),
        "test": index[test_mask].reset_index(drop=True),
    }


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Split the assembled source index into season-based train/val/test files."""
    cfg = resolve_preproc_cfg(raw_cfg)

    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    index_path = assembled_root / "index.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Assembled index not found at {index_path}. Run scripts/preprocess/assemble.py first."
        )

    val_seasons: set[int] = set(cfg["splits"]["val"])
    test_seasons: set[int] = set(cfg["splits"]["test"])

    print(f"Loading index from {index_path} …")
    index = pd.read_parquet(index_path)
    print(f"  {len(index)} rows, {index['sid'].nunique()} unique storms.")

    splits = split_by_season(index, val_seasons, test_seasons)

    print(f"\n{'Split':<8}  {'Storms':>7}  {'Rows':>9}  {'Seasons'}")
    print("-" * 55)
    for split_name, df in splits.items():
        out_path = assembled_root / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        seasons_str = ", ".join(str(s) for s in sorted(df["season"].unique())) if len(df) else ""
        print(f"{split_name:<8}  {df['sid'].nunique():>7}  {len(df):>9}  {seasons_str}")

    total = sum(len(df) for df in splits.values())
    assert total == len(index), (
        f"Row count mismatch after splitting: {total} != {len(index)}. "
        "This is a bug in build_splits.py."
    )
    print(f"\nWrote train/val/test index files to {assembled_root}")


if __name__ == "__main__":
    main()
