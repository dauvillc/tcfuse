# /preprocess — TC Multi-Source Preprocessing Agent

This skill helps you preprocess any of the project's datasets into the standard two-stage
format: per-source HDF5 snapshots (Stage 1), then assembled per-storm HDF5 files (Stage 2).

---

## Dataset inventory

| Dataset | Kind | Sources extracted | Status | Raw path config key |
|---|---|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR geostationary, ERA5 surface, best-track | `best_track` (0D), `pmw_*` (2D), `ir_goes` (2D), `era5_surface` (2D) | Preprocessor implemented | `paths.raw_datasets.tc_primed` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR (Sentinel-1) surface winds | `cyclobs_*` (2D) | Preprocessor not yet written | `paths.raw_datasets.cyclobs` |
| **NOAA AOML Dropsondes** | Vertical profiles P/T/RH/u/v from hurricane recon | `dropsonde` (1D) | Preprocessor not yet written | `paths.raw_datasets.dropsondes` |
| **Argo floats** | T/S profiles 0–2000 m depth | `argo` (1D) | Preprocessor not yet written | `paths.raw_datasets.argo` |

---

## Stage 1 — Per-source format

Config key: `cfg.paths.preprocessed_sources`

```
${paths.preprocessed_sources}/
├── pmw_amsr2_gcomw1/
│   ├── metadata.yaml           # source kind, channels, char_vars
│   ├── index.parquet           # one row per snapshot
│   └── snapshots/
│       └── {storm_id}_{YYYYMMDDTHHMMSSZ}.h5
├── pmw_ssmis_f18/
│   └── ...
├── ir_tcirar/
│   └── ...
└── radar_gmi/
    └── ...
```

Each HDF5 file holds **exactly one source**, written by `Source.write(path)`.
Use `Source.path(sources_root, source_name, storm_id, snapshot_time_utc)` to compute canonical paths.

**HDF5 file structure per snapshot:**
```
/
├── attrs: {storm_id, basin, snapshot_time_utc, lat, lon, vmax_kt, …}  ← Source.meta
├── scalar/
│   └── {source_name}/
│       ├── values    float32 (C,),       gzip-4
│       ├── coords    float64 (3,),       gzip-4  [time_unix_s, lat, lon]
│       ├── [mask]    bool    (C,)        (optional)
│       └── attrs:    {source_name, channels (JSON), char_vars (JSON)}
├── profile/
│   └── {source_name}/
│       ├── values    float32 (L, C),     gzip-4
│       ├── coords    float64 (L, 4),     gzip-4  [time_unix_s, lat, lon, alt_m]
│       ├── [mask]    bool    (L, C)      (optional)
│       └── attrs:    {source_name, channels (JSON), char_vars (JSON)}
└── field/
    └── {source_name}/
        ├── values    float32 (H, W, C),  gzip-4
        ├── coords    float32 (H, W, 3),  gzip-4  [time_unix_s broadcast, lat, lon]
        ├── [mask]    bool    (H, W)      (optional)
        └── attrs:    {source_name, channels (JSON), char_vars (JSON)}
```

**Missing sources:** simply absent from the HDF5 file — no empty groups.

**metadata.yaml schema (per source):**
```yaml
name: pmw_amsr2_gcomw1
type: microwave      # physical category
kind: field          # scalar | profile | field
channels: [tb_36.5h, tb_36.5v, tb_a89.0h, tb_a89.0v]
num_channels: 4
char_vars:
  ifov: {tb_36.5h: [7.2, 4.4, 7.2, 4.4], …}
```

**index.parquet schema (per source):**
```
storm_id, basin, snapshot_time_utc, lat, lon, vmax_kt, mslp_hpa,
development_level, storm_speed_ms, storm_heading_deg,
source_name (str), file_path (str)
```
Columns vary by source; the listed ones are always present.

---

## Stage 2 — Assembled format

Config key: `cfg.paths.preprocessed_data`

```
${paths.preprocessed_data}/
├── storm_data/
│   └── {ibtracs_sid}.h5    # one file per storm, e.g. 2016228N14275.h5
└── index.parquet           # global assembled index (USA ATCF–tracked storms only)
```

Each HDF5 file holds **all sources for one storm**, written by `StormData.write(assembled_root)`.
Use `StormData.path(assembled_root, storm_id)` to compute canonical paths.

**HDF5 file structure (assembled):**
```
/
├── attrs: {storm_id (IBTrACS SID), basin, season, atcf_id}
└── {source_name}/
    └── {compact_snapshot_time}/     # e.g., "20160912T010942Z"
        ├── values    float32, gzip-4
        ├── coords    float32 (FIELD) or float64 (SCALAR/PROFILE), gzip-4
        ├── [mask]    bool (only when present in Source)
        └── attrs:
            ├── source_name        str
            ├── channels           JSON list
            ├── char_vars          JSON list
            ├── kind               "SCALAR" | "PROFILE" | "FIELD"
            ├── snapshot_time_utc  isoformat str (for round-trip key recovery)
            └── [lat, lon, vmax_kt, …]  from Source.meta
```

**ATCF-only filtering:** `assemble.py` requires the IBTrACS CSV at
`cfg.paths.raw_datasets.ibtracs` and assembles only storms whose Stage-1 `storm_id`
has a non-empty `USA_ATCF_ID` in IBTrACS (USA agency tracking). Storms without an
ATCF match are skipped entirely.

**IBTrACS injection:** For each retained storm, `assemble.py` injects one SCALAR source per best-track
observation time:
- `source_name = "ibtracs_best_track"`
- `kind = SCALAR`, channels: `[usa_vmax_kt, wmo_vmax_kt, usa_mslp_hpa, wmo_mslp_hpa, usa_rmw_nm, usa_r34_ne_nm, usa_r34_se_nm, usa_r34_sw_nm, usa_r34_nw_nm]`
- USA and WMO quantities are kept as distinct channels; missing values remain NaN.

**`StormData` sources dict key:** `(source_name, snapshot_time_utc)` where
`snapshot_time_utc` is the isoformat string as it appears in per-source `index.parquet`.

**Assembled index.parquet schema:**
```
storm_id, basin, season, atcf_id,
source_name, snapshot_time_utc, lat, lon, usa_vmax_kt, wmo_vmax_kt
```
One row per (storm, source_name, snapshot). `ibtracs_best_track` rows are included.

---

## Running the preprocessor

### Step 1 — Verify paths

```bash
# Check that raw data is present
ls $SCRATCH/tcfuse/data/raw/tc_primed/
ls $SCRATCH/tcfuse/data/raw/ibtracs/    # needed by the assembly step
```

For local runs, `$SCRATCH` must be set; or override `paths.scratch` on the command line.

### Step 2 — Local debug run of source preprocessors (no SLURM)

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py submitit=false
python scripts/preprocess/tc_primed/prepare_infrared.py submitit=false
python scripts/preprocess/tc_primed/prepare_radar.py submitit=false
```

To limit to a subset during development, pass `include_seasons=[2020]` or similar via the
config override.

### Step 3 — Jean-Zay production run of source preprocessors

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py paths=jz setup=jz_cpu
```

Use `/jz` for preflight checks (quota, env, W&B mode) before submitting.

### Step 4 — Validate Stage 1 output

```python
from pathlib import Path
from tcfuse.data.sources import Source
import pandas as pd

sources_root = Path("$SCRATCH/tcfuse/data/preprocessed/sources")

# Spot-check one per-source snapshot
p = next((sources_root / "pmw_amsr2_gcomw1" / "snapshots").glob("*.h5"))
src = Source.from_disk(p)
print(src.kind, src.values.shape, src.channels)

# Check per-source index
df = pd.read_parquet(sources_root / "pmw_amsr2_gcomw1" / "index.parquet")
print(df.head())
print(df["source_name"].value_counts())
```

### Step 5 — Run assembly

Requires IBTrACS at `cfg.paths.raw_datasets.ibtracs`. Only storms with a USA ATCF ID
in IBTrACS are assembled; others are skipped.

```bash
# Local debug run (no SLURM)
python scripts/preprocess/assemble.py submitit=false num_workers=4

# Jean-Zay production run
python scripts/preprocess/assemble.py paths=jz setup=jz_cpu
```

Key options:
- `skip_existing=true` — resume a partial run without rewriting existing storm files
- `chunk_size=200` — number of storms per SLURM job (SLURM mode only)

Re-running assembly does not delete previously assembled storm files that no longer
pass the ATCF filter; use `skip_existing=false` or remove stale `storm_data/*.h5`
before `build_splits.py` for a clean refresh.

Validate Stage 2 output:

```python
from pathlib import Path
from tcfuse.data.sources import StormData
import pandas as pd

assembled_root = Path("$SCRATCH/tcfuse/data/preprocessed/assembled")

# Spot-check one assembled storm
storm_id = next((assembled_root / "storm_data").glob("*.h5")).stem
sd = StormData.from_disk(assembled_root, storm_id)
print(sd.storm_id, sd.basin, sd.season)
print({k: v.values.shape for k, v in sd.sources.items()})

# Check global assembled index
df = pd.read_parquet(assembled_root / "index.parquet")
print(df["source_name"].value_counts())
```

### Step 6 — Build train/val/test splits

After all desired sources are preprocessed and assembled, run:

```bash
# Local run
python scripts/preprocess/build_splits.py

# Jean-Zay
python scripts/preprocess/build_splits.py paths=jz
```

This reads the assembled `index.parquet` and builds one model sample per valid
`ibtracs_best_track` window. By default, each sample spans lead hours
`[0, 6, 12, 18, 24, 30]`; finite `usa_vmax_kt`, `lat`, and `lon` are required at
`+0h`, `+6h`, and `+30h`, while intermediate leads may be missing or NaN.

The resulting sample rows are assigned to splits based on the season lists in
`conf/preproc.yaml`:
- **val**:   seasons in `cfg.splits.val`
- **test**:  seasons in `cfg.splits.test`
- **train**: all remaining seasons

The `season` column holds a single value per storm lifetime (not per-snapshot), so a
storm that crosses a calendar year boundary is always assigned to one split.

Output files:

```
${paths.preprocessed_data}/
├── index.parquet       # canonical source-snapshot index
├── train.parquet
├── val.parquet
└── test.parquet
```

`train.parquet`, `val.parquet`, and `test.parquet` are model-sample window
indices, not source-snapshot indices.

### Step 7 — Compute normalization statistics

After splits are built, compute per-channel mean and std for every source using an online
Welford algorithm.  `compute_normalization.py` reads the training window index,
derives the training storms, then filters the canonical assembled `index.parquet` to
source snapshots from those storms. This prevents leakage from val/test seasons.
Results are used by the data transforms layer during training.

```bash
# Local run (sequential, one source at a time)
python scripts/preprocess/compute_normalization.py submitit=false

# Jean-Zay production run (one SLURM job per source, parallel)
python scripts/preprocess/compute_normalization.py paths=jz setup=jz_cpu
```

Outputs:
- `{preprocessed_data}/normalization_stats.yaml` — merged stats for all sources
- `{preprocessed_data}/normalization/{source_name}.yaml` — intermediate per-source stats
- `figures/normalization/{source_name}.png` — channel distribution histograms

Stats YAML structure:
```yaml
pmw_amsr2_gcomw1:
  kind: field
  channels:
    tb_36.5h:
      mean: 234.567
      std: 12.345
      count: 1234567
ibtracs_best_track:
  kind: scalar
  channels:
    vmax_kt:
      mean: 62.1
      std: 28.4
      count: 85000
```

Masked pixels (where `mask == False`) and non-finite values are excluded from both
statistics and histogram samples.  `ibtracs_best_track` is included even though it has
no `metadata.yaml` in `sources_root`; its channels are discovered from the HDF5 attrs.

---

## Adding a new dataset preprocessor

1. Add source definitions to a new `conf/data/<name>.yaml` following the TC-PRIMED template.
2. Create `scripts/preprocess/<name>.py` modelling it on `scripts/preprocess/tc_primed/prepare_pmw.py`.
3. Add the dataset to the table at the top of this skill.
4. Update the `CLAUDE.md` dataset table with the confirmed `$SCRATCH` path once known.

---

## I/O API reference

All read/write operations go through `src/tcfuse/data/sources/`:

| Class / function | Location | Purpose |
|---|---|---|
| `Source.write(path)` | `sources/source.py` | Write one Source to a self-contained HDF5 file |
| `Source.from_disk(path)` | `sources/source.py` | Load Source + meta from HDF5 |
| `Source.read_meta(path)` | `sources/source.py` | Read root attrs only (no tensors) |
| `Source.path(sources_root, source_name, storm_id, snapshot_time_utc)` | `sources/source.py` | Compute canonical per-source snapshot path |
| `Source.to_hdf5_group(group)` | `sources/source.py` | Low-level: write tensors into an open HDF5 group |
| `Source.from_hdf5_group(group, kind)` | `sources/source.py` | Low-level: read tensors from an open HDF5 group |
| `StormData.write(assembled_root)` | `sources/storm_data.py` | Write assembled per-storm HDF5 (all sources) |
| `StormData.from_disk(assembled_root, storm_id)` | `sources/storm_data.py` | Load all sources for a storm |
| `StormData.read_meta(assembled_root, storm_id)` | `sources/storm_data.py` | Read root attrs only (no tensors) |
| `StormData.path(assembled_root, storm_id)` | `sources/storm_data.py` | Canonical assembled file path |
| `SourceMetadata.from_disk(source_dir)` | `sources/metadata.py` | Load metadata.yaml + index for one source |
| `MultisourceMetadata.from_disk(sources_root)` | `sources/metadata.py` | Scan sources_root and load all SourceMetadata |
