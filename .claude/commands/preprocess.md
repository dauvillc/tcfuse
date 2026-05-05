# /preprocess — TC Multi-Source Preprocessing Agent

This skill helps you preprocess any of the project's datasets into the standard HDF5 per-snapshot format.

---

## Dataset inventory

| Dataset | Kind | Sources extracted | Status | Raw path config key |
|---|---|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR geostationary, ERA5 surface, best-track | `best_track` (0D), `pmw_*` (2D), `ir_goes` (2D), `era5_surface` (2D) | Preprocessor implemented | `paths.raw_datasets.tc_primed` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR (Sentinel-1) surface winds | `cyclobs_*` (2D) | Preprocessor not yet written | `paths.raw_datasets.cyclobs` |
| **NOAA AOML Dropsondes** | Vertical profiles P/T/RH/u/v from hurricane recon | `dropsonde` (1D) | Preprocessor not yet written | `paths.raw_datasets.dropsondes` |
| **Argo floats** | T/S profiles 0–2000 m depth | `argo` (1D) | Preprocessor not yet written | `paths.raw_datasets.argo` |

---

## On-disk format (standard)

Preprocessed data is **source-first** under `cfg.paths.preprocessed_sources`:

```
${paths.preprocessed_sources}/
├── pmw_amsr2_gcomw1/
│   ├── index.parquet           # fast lookup; one row per snapshot
│   └── snapshots/
│       └── {storm_id}_{YYYYMMDDTHHMMSSZ}.h5   # contains pmw_amsr2_gcomw1 only
├── pmw_ssmis_f18/
│   └── ...
├── era5_surface/
│   └── ...
└── best_track/
    └── ...
```

Each HDF5 file holds **exactly one source**. Use
`source_snapshot_path(sources_root, source_name, storm_id, snapshot_time_utc)` from
`src/tcfuse/utils/io.py` to compute canonical file paths.

**HDF5 file structure per snapshot:**
```
/
├── attrs: {storm_id, basin, snapshot_time_utc, lat, lon, vmax_kt, mslp_hpa}
├── scalar/
│   └── {source_name}/
│       ├── values    float32 (C,)
│       ├── coords    float64 (3,)     [time_unix_s, lat_deg, lon_deg]
│       └── attrs:    {source_name, channels: [...]}
├── profile/
│   └── {source_name}/
│       ├── values    float32 (L, C)
│       ├── coords    float64 (L, 4)   [time_unix_s, lat, lon, alt_m]
│       ├── mask      bool    (L, C)   (present only when missing values exist)
│       └── attrs:    {source_name, channels: [...]}
└── field/
    └── {source_name}/
        ├── values    float32 (H, W, C)  H, W vary per snapshot
        ├── coords    float32 (H, W, 3)  [time_unix_s broadcast, lat, lon]
        ├── mask      bool    (H, W)     (present only when missing values exist)
        └── attrs:    {source_name, channels: [...]}
```

**Missing sources:** simply absent from the HDF5 file — no empty groups.

**Index parquet schema (per source):**
```
storm_id, basin, snapshot_time_utc, lat, lon, vmax_kt, mslp_hpa,
development_level, storm_speed_ms, storm_heading_deg,
source_name (str), file_path (str)
```

No `split` column — splitting is handled separately (see Step 5 below).

---

## Running the preprocessor

### Step 1 — Verify paths

```bash
# Check that raw data is present
ls $SCRATCH/tcfuse/data/raw/tc_primed/
```

For local runs, `$SCRATCH` must be set; or override `paths.scratch` on the command line.

### Step 2 — Local debug run (no SLURM)

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py submitit=false
```

To limit to a subset during development, pass `include_seasons=[2020]` or similar via the
config override.

### Step 3 — Jean-Zay production run

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py paths=jz setup=jz_cpu
```

Use `/jz` for preflight checks (quota, env, W&B mode) before submitting.

### Step 4 — Validate output

After the run, open a Python shell and verify:

```python
from pathlib import Path
from tcfuse.utils.io import read_snapshot, source_snapshot_path
import pandas as pd

sources_root = Path("$SCRATCH/tcfuse/data/preprocessed/sources")

# Spot-check one source
source_dir = sources_root / "pmw_amsr2_gcomw1"
path = next((source_dir / "snapshots").glob("*.h5"))
sources = read_snapshot(path)
print({k: v.values.shape for k, v in sources.items()})

# Check per-source index
df = pd.read_parquet(source_dir / "index.parquet")
print(df.head())
print(df["source_name"].value_counts())
```

### Step 5 — Build train/val/test splits

After all desired sources are preprocessed, run:

```bash
python scripts/preprocess/build_splits.py
```

This reads each source's `index.parquet`, merges them on `(storm_id, snapshot_time_utc)`,
applies the year-based split rule (`year % 10 == 0` → test; `year % 5 == 0` → val;
else → train), and writes:

```
${paths.preprocessed_sources}/
├── train.parquet
├── val.parquet
└── test.parquet
```

Each file has all columns from the per-source indexes.

---

## Adding a new dataset preprocessor

1. Add source definitions to a new `conf/data/<name>.yaml` following the TC-PRIMED template.
2. Create `scripts/preprocess/<name>.py` modelling it on `scripts/preprocess/tc_primed.py`.
3. Add the dataset to the table at the top of this skill.
4. Update the `CLAUDE.md` dataset table with the confirmed `$SCRATCH` path once known.

---

## I/O API reference

All read/write operations go through `src/tcfuse/utils/io.py`:

| Function | Purpose |
|---|---|
| `source_snapshot_path(sources_root, source_name, storm_id, snapshot_time_utc)` | Compute canonical path for a source snapshot |
| `write_snapshot(path, meta, sources)` | Write one snapshot to HDF5 (pass a single-key dict for one source per file) |
| `read_snapshot(path, source_names=None)` | Read all or selected sources from HDF5 |
| `read_snapshot_meta(path)` | Read only root attrs (for index building) |
| `write_source(group, source)` | Low-level: write one Source to an open group |
| `read_source(group, kind)` | Low-level: read one Source from an open group |
