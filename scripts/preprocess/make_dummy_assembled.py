#!/usr/bin/env python3
"""Generate a tiny synthetic assembled dataset for local smoke-testing.

Creates the minimum directory structure expected by TCWindowDataModule and
TCWindowDataset under cfg.paths.preprocessed_data / pmw_gmi_reconstruction/:
  - sources_metadata.yaml
  - normalization_stats.yaml
  - storm_data/{sid}.h5  (assembled StormData HDF5 files)
  - pmw_gmi_reconstruction/train_windows.parquet
  - pmw_gmi_reconstruction/val_windows.parquet
  - pmw_gmi_reconstruction/test_windows.parquet

Sources: pmw_gmi_gpm (target) and pmw_amsr2_gcomw1 (context), both FIELD 8x8 with 2 channels.
Storms: 8 for train, 2 for val, 2 for test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

# Allow imports from src/tcfuse regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tcfuse.data.sources.source import Source, SourceKind
from tcfuse.data.sources.storm_data import StormData

# ── Config ────────────────────────────────────────────────────────────────────

ASSEMBLED_ROOT = Path("/home/cdauvill/scratch/tcfuse/data/preprocessed/assembled")
WINDOWS_SETUP_NAME = "pmw_gmi_reconstruction"

# Tiny spatial resolution for fast local testing.
H, W = 8, 8
CHANNELS = ["tb_10.65h", "tb_10.65v"]

# Fake storm metadata organised by split.
STORMS: dict[str, list[dict]] = {
    "train": [
        {"sid": "DUMMY_TRAIN_01", "basin": "AL", "subbasin": "GM", "season": 2019},
        {"sid": "DUMMY_TRAIN_02", "basin": "AL", "subbasin": "GM", "season": 2019},
        {"sid": "DUMMY_TRAIN_03", "basin": "WP", "subbasin": "WP", "season": 2019},
        {"sid": "DUMMY_TRAIN_04", "basin": "EP", "subbasin": "CP", "season": 2019},
        {"sid": "DUMMY_TRAIN_05", "basin": "AL", "subbasin": "GM", "season": 2020},
        {"sid": "DUMMY_TRAIN_06", "basin": "WP", "subbasin": "WP", "season": 2020},
        {"sid": "DUMMY_TRAIN_07", "basin": "EP", "subbasin": "CP", "season": 2020},
        {"sid": "DUMMY_TRAIN_08", "basin": "AL", "subbasin": "GM", "season": 2020},
    ],
    "val": [
        {"sid": "DUMMY_VAL_01", "basin": "AL", "subbasin": "GM", "season": 2021},
        {"sid": "DUMMY_VAL_02", "basin": "WP", "subbasin": "WP", "season": 2021},
    ],
    "test": [
        {"sid": "DUMMY_TEST_01", "basin": "AL", "subbasin": "GM", "season": 2022},
        {"sid": "DUMMY_TEST_02", "basin": "WP", "subbasin": "WP", "season": 2022},
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_field_source(
    source_name: str, time_utc: pd.Timestamp, storm_id: str, basin: str
) -> Source:
    """Create a synthetic FIELD Source with random values and a full mask."""
    rng = np.random.default_rng(seed=abs(hash(source_name + str(time_utc) + storm_id)) % (2**31))
    # Random brightness temperatures in physically plausible range [200, 300] K.
    values = rng.uniform(200.0, 300.0, size=(H, W, len(CHANNELS))).astype(np.float32)
    # Lat/lon grid centered near 20°N, 70°W (Atlantic basin).
    lat_center, lon_center = 20.0, -70.0
    lats = np.linspace(lat_center - 2, lat_center + 2, H, dtype=np.float32)
    lons = np.linspace(lon_center - 2, lon_center + 2, W, dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    coords = np.stack([lat_grid, lon_grid], axis=-1).astype(np.float32)  # (H, W, 2)
    mask = np.ones((H, W, len(CHANNELS)), dtype=bool)
    return Source(
        kind=SourceKind.FIELD,
        values=values,
        coords=coords,
        source_name=source_name,
        channels=CHANNELS,
        mask=mask,
        time_utc=time_utc,
        meta={"storm_id": storm_id, "basin": basin, "time_utc": time_utc.isoformat()},
    )


def make_storm_h5(storm_info: dict, gmi_time: pd.Timestamp, ctx_time: pd.Timestamp) -> None:
    """Write an assembled HDF5 for a storm with one GMI and one AMSR2 snapshot."""
    sid = storm_info["sid"]
    gmi_src = make_field_source("pmw_gmi_gpm", gmi_time, sid, storm_info["basin"])
    amsr2_src = make_field_source("pmw_amsr2_gcomw1", ctx_time, sid, storm_info["basin"])
    storm_data = StormData(
        storm_id=sid,
        basin=storm_info["basin"],
        subbasin=storm_info["subbasin"],
        season=storm_info["season"],
        sources={
            ("pmw_gmi_gpm", gmi_time.isoformat()): gmi_src,
            ("pmw_amsr2_gcomw1", ctx_time.isoformat()): amsr2_src,
        },
        atcf_id=None,
    )
    storm_data.write(ASSEMBLED_ROOT)


def make_window_rows(
    storm_info: dict, gmi_time: pd.Timestamp, ctx_time: pd.Timestamp, window_id: str
) -> list[dict]:
    """Return long-format window-index rows (one per snapshot) for a single window."""
    sid = storm_info["sid"]
    ref_time = gmi_time
    start_time = gmi_time - pd.Timedelta(hours=3)
    end_time = gmi_time + pd.Timedelta(hours=3)
    base = {
        "window_id": window_id,
        "sid": sid,
        "season": storm_info["season"],
        "basin": storm_info["basin"],
        "subbasin": storm_info["subbasin"],
        "usa_atcf_id": float("nan"),
        "window_ref_time_utc": str(ref_time),
        "window_start_time_utc": str(start_time),
        "window_end_time_utc": str(end_time),
    }
    return [
        {**base, "source_name": "pmw_gmi_gpm", "time_utc": str(gmi_time), "is_target": True},
        {**base, "source_name": "pmw_amsr2_gcomw1", "time_utc": str(ctx_time), "is_target": False},
    ]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ASSEMBLED_ROOT.mkdir(parents=True, exist_ok=True)

    # Write sources_metadata.yaml.
    sources_metadata = {
        "pmw_gmi_gpm": {
            "name": "pmw_gmi_gpm",
            "type": "microwave",
            "kind": "field",
            "channels": CHANNELS,
            "num_channels": len(CHANNELS),
            "shape": [H, W],
            "char_vars": {},
        },
        "pmw_amsr2_gcomw1": {
            "name": "pmw_amsr2_gcomw1",
            "type": "microwave",
            "kind": "field",
            "channels": CHANNELS,
            "num_channels": len(CHANNELS),
            "shape": [H, W],
            "char_vars": {},
        },
    }
    with open(ASSEMBLED_ROOT / "sources_metadata.yaml", "w") as f:
        yaml.dump(sources_metadata, f, default_flow_style=False, sort_keys=False)
    print("Wrote sources_metadata.yaml")

    # Write normalization_stats.yaml.
    norm_stats: dict = {}
    for src_name in ("pmw_gmi_gpm", "pmw_amsr2_gcomw1"):
        norm_stats[src_name] = {
            "channels": {
                "tb_10.65h": {"mean": 250.0, "std": 30.0, "count": 1000},
                "tb_10.65v": {"mean": 260.0, "std": 25.0, "count": 1000},
            }
        }
    with open(ASSEMBLED_ROOT / "normalization_stats.yaml", "w") as f:
        yaml.dump(norm_stats, f, default_flow_style=False, sort_keys=False)
    print("Wrote normalization_stats.yaml")

    # Generate HDF5 files and window index rows per split.
    windows_dir = ASSEMBLED_ROOT / WINDOWS_SETUP_NAME
    windows_dir.mkdir(parents=True, exist_ok=True)

    base_time = pd.Timestamp("2019-08-01 12:00:00")

    for split, storm_list in STORMS.items():
        rows: list[dict] = []
        for i, storm_info in enumerate(storm_list):
            # Give each storm a unique observation time.
            gmi_time = cast(pd.Timestamp, base_time + pd.Timedelta(hours=6 * i))
            ctx_time = cast(pd.Timestamp, gmi_time - pd.Timedelta(hours=1))
            window_id = f"{split}_{storm_info['sid']}_w0"

            # Write the assembled HDF5 file.
            make_storm_h5(storm_info, gmi_time, ctx_time)

            # Collect window-index rows.
            rows.extend(make_window_rows(storm_info, gmi_time, ctx_time, window_id))

        df = pd.DataFrame(rows)
        out_path = windows_dir / f"{split}_windows.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows, {len(df['window_id'].unique())} windows)")

    print("Done. Dummy assembled dataset is ready.")


if __name__ == "__main__":
    main()
