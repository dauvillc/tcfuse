---
name: tcfuse-preprocess
description: >-
  TC-Fuse dataset preprocessing pipeline — Stage 0 IBTrACS prep + ATCF→SID
  translation table, Stage 1 per-source HDF5 snapshots, Stage 2 assembled
  per-storm HDF5 + concatenated index, Stage 3 best-track window splits,
  normalization statistics, the I/O API in `src/tcfuse/data/sources/`. Use
  when preparing any dataset (TC-PRIMED, CyclObs, dropsondes, Argo), running
  `prepare_ibtracs.py` / `assemble.py` / `build_splits.py` /
  `compute_normalization.py`, working with `Source` or `StormData`, or
  extending the per-source HDF5 / assembled formats.
---

# TC-Fuse preprocessing pipeline

Claude Code: invoke `/preprocess` (reads this skill).

## Pipeline at a glance

The pipeline runs in four explicit stages. Each stage reads only the outputs
of previous stages — never raw inputs from earlier ones.

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| 0 | `prepare_ibtracs.py` | raw IBTrACS CSV | `ibtracs/ibtracs.parquet`, `ibtracs/atcf_to_sid.csv` |
| 1 | `prepare_pmw.py`, `prepare_infrared.py`, `prepare_radar.py`, `sar/prepare_sar.py` | raw TC-PRIMED / CyclObs files, Stage 0 `atcf_to_sid.csv` | per-source HDF5 snapshots + `index.parquet` |
| 2 | `assemble.py` | Stage 0 `ibtracs.parquet`, Stage 1 indices + snapshots | `storm_data/{sid}.h5`, `index.parquet` (concatenated) |
| 3 | `build_splits.py`, `compute_normalization.py` | Stage 2 `index.parquet` | `train.parquet`, `val.parquet`, `test.parquet`, normalization stats |

## When to use

- Writing or modifying a per-source preprocessor under `scripts/preprocess/`.
- Running `prepare_ibtracs.py`, `assemble.py`, `build_splits.py`, or `compute_normalization.py`.
- Reading or writing `Source` / `StormData` HDF5 files.
- Understanding the on-disk Stage 0 / Stage 1 / Stage 2 layouts and the assembled `index.parquet`.
- Adding a new dataset to the project.

For Jean-Zay submission, preflight, or SLURM setup, see [tcfuse-jz](../tcfuse-jz/SKILL.md).
For forecast output storage (predictions, not preprocessing), see [tcfuse-predictions](../tcfuse-predictions/SKILL.md).

## Dataset inventory

| Dataset | Kind | Sources extracted | Status | Raw path config key |
|---|---|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR geostationary, ERA5 surface, best-track | `best_track` (0D), `pmw_*` (2D), `ir_goes` (2D), `era5_surface` (2D) | Preprocessor implemented | `paths.raw_datasets.tc_primed` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR (Sentinel-1) surface winds | `cyclobs_*` (2D) | Preprocessor not yet written | `paths.raw_datasets.cyclobs` |
| **NOAA AOML Dropsondes** | Vertical profiles P/T/RH/u/v from hurricane recon | `dropsonde` (1D) | Preprocessor not yet written | `paths.raw_datasets.dropsondes` |
| **Argo floats** | T/S profiles 0–2000 m depth | `argo` (1D) | Preprocessor not yet written | `paths.raw_datasets.argo` |

## Stage 0 — IBTrACS preprocessing

Config keys: `cfg.paths.raw_datasets.ibtracs` (input) and `cfg.paths.preprocessed_sources` (output).
Script: `scripts/preprocess/prepare_ibtracs.py`.

`prepare_ibtracs.py` reads the raw IBTrACS CSV once, filters to `TRACK_TYPE == "MAIN"`,
keeps a trimmed lowercase column set, and writes two artifacts under
`${paths.preprocessed_sources}/ibtracs/`:

```
${paths.preprocessed_sources}/ibtracs/
├── ibtracs.parquet     # one row per (sid, iso_time)
└── atcf_to_sid.csv     # translation table consumed by every Stage 1 preprocessor
```

**`ibtracs.parquet` columns (all lowercased):**

```
sid, season, basin, subbasin, name, number, iso_time, nature, lat, lon,
usa_atcf_id, usa_wind, usa_pres, usa_sshs,
usa_r34_ne, usa_r34_se, usa_r34_sw, usa_r34_nw,
usa_r50_ne, usa_r50_se, usa_r50_sw, usa_r50_nw,
usa_r64_ne, usa_r64_se, usa_r64_sw, usa_r64_nw
```

`iso_time` is a naive-UTC isoformat string. `season` is `int64`; `usa_sshs`
and `number` are nullable `Int64`; everything else numeric is `float64`.

**`atcf_to_sid.csv` columns:** `sid, season, basin, subbasin, name, usa_atcf_id`,
deduplicated on `[sid, usa_atcf_id]`. A `ValueError` is raised if any SID maps
to more than one `USA_ATCF_ID` (the per-SID check; a single ATCF mapping to
multiple SIDs across seasons is allowed).

**Loader API** in `src/tcfuse/data/ibtracs.py`:

| Function | Purpose |
|---|---|
| `ibtracs_paths(sources_root)` | Return `(ibtracs.parquet, atcf_to_sid.csv)` paths. |
| `load_ibtracs_snapshots(sources_root)` | Load the snapshots parquet as a DataFrame. |
| `load_atcf_to_sid(sources_root)` | Load the translation table as a DataFrame. |
| `load_atcf_to_sid_dict(sources_root)` | `{usa_atcf_id: sid}` mapping. |
| `ibtracs_rows_to_sources(rows, sid, basin)` | Convert per-storm rows into 16-channel SCALAR `Source` snapshots. |
| `group_ibtracs_by_sid(snapshots)` | Group snapshots by SID for fast per-storm access. |

`IBTRACS_SOURCE_NAME = "ibtracs_best_track"` and `IBTRACS_CHANNELS` (length 16) are
the canonical names of the injected best-track Source — see Stage 2 below.

## Stage 1 — Per-source format

Config key: `cfg.paths.preprocessed_sources`

```
${paths.preprocessed_sources}/
├── ibtracs/                    # Stage 0 outputs (see above)
├── pmw_amsr2_gcomw1/
│   ├── metadata.yaml           # source kind, channels, char_vars
│   ├── index.parquet           # one row per snapshot
│   └── snapshots/
│       └── {sid}_{YYYYMMDDTHHMMSSZ}.h5
├── pmw_ssmis_f18/
│   └── ...
├── ir_tcirar/
│   └── ...
└── radar_gmi/
    └── ...
```

Each per-source HDF5 file holds **exactly one source**, written by `Source.write(path)`.
Use `Source.path(sources_root, source_name, sid, snapshot_time_utc)` to compute canonical paths.

**ATCF→SID translation at Stage 1:** every per-source preprocessor calls
`load_translation(sources_root)` to load the Stage 0 ATCF→SID lookup, then in the
per-file worker:

1. Reads the dataset-native ATCF ID (e.g. `storm_id` in TC-PRIMED, `sid` in CyclObs).
2. Looks up the IBTrACS SID; if missing, the worker emits `warnings.warn(...)` and
   **discards** the file — no snapshot is written.
3. Uses the SID in the HDF5 filename and writes `Source.meta = {"storm_id": sid,
   "snapshot_time_utc": <iso>}`. **`storm_lat` and `storm_lon` are no longer written**
   to `Source.meta` — per-pixel coordinates already live in `Source.coords`.

**Per-source `index.parquet` schema (canonical for every source):**

```
sid, source_name, snapshot_time_utc, season, basin, subbasin
```

The index is rebuilt at the end of every Stage 1 run by `finalize_source` in
`scripts/preprocess/utils/runner.py`, which scans `snapshots/*.h5`, reads each
file's `storm_id` (=SID) and `snapshot_time_utc` root attrs, and left-joins
the Stage 0 translation table to populate `season / basin / subbasin`.

**HDF5 file structure per snapshot:**
```
/
├── attrs: {storm_id (=SID), snapshot_time_utc, …}     ← Source.meta
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

## Stage 2 — Assembled format

Config key: `cfg.paths.preprocessed_data`

```
${paths.preprocessed_data}/
├── storm_data/
│   └── {sid}.h5            # one file per IBTrACS storm, e.g. 2016228N14275.h5
└── index.parquet           # concatenated assembled index (Stage 1 + IBTrACS rows)
```

Each HDF5 file holds **all sources for one storm**, written by `StormData.write(assembled_root)`.
Use `StormData.path(assembled_root, sid)` to compute canonical paths.

**HDF5 file structure (assembled):**
```
/
├── attrs: {storm_id (=SID), basin, subbasin, season, atcf_id?}
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
            └── […]                from Source.meta
```

`StormData.subbasin` is **mandatory**. Older assembled files written before
the four-stage refactor are missing this root attr and must be regenerated.

**Storm selection:** `assemble.py` drives the storm set straight from the
Stage 0 `ibtracs.parquet` (already filtered to `TRACK_TYPE == "MAIN"`). It
never re-reads the raw IBTrACS CSV. Stage 1 indices already key on SID, so
no ATCF translation happens at this stage.

**IBTrACS injection:** For each storm, `assemble.py` injects one SCALAR
`ibtracs_best_track` Source per best-track observation time via
`ibtracs_rows_to_sources`:
- `source_name = "ibtracs_best_track"`
- `kind = SCALAR`, `coords = [time_unix_s, lat, lon]`
- channels (length 16) — `IBTRACS_CHANNELS` in `src/tcfuse/data/ibtracs.py`:
  ```
  usa_wind, usa_pres, lat, lon,
  usa_r34_ne, usa_r34_se, usa_r34_sw, usa_r34_nw,
  usa_r50_ne, usa_r50_se, usa_r50_sw, usa_r50_nw,
  usa_r64_ne, usa_r64_se, usa_r64_sw, usa_r64_nw
  ```
  `lat` and `lon` are intentionally duplicated as values so the embedding
  layer can treat storm position as a feature, not only as a coordinate.

**`StormData` sources dict key:** `(source_name, snapshot_time_utc)` where
`snapshot_time_utc` is the isoformat string as it appears in per-source `index.parquet`.

**Assembled `index.parquet` schema:** the concatenation of two row blocks
(satellite snapshots first, IBTrACS rows second), sharing a union schema with
columns in this order:

```
sid, source_name, snapshot_time_utc, season, basin, subbasin,
name, number, nature, lat, lon,
usa_atcf_id, usa_wind, usa_pres, usa_sshs,
usa_r34_*, usa_r50_*, usa_r64_*
```

Satellite rows leave the IBTrACS-specific columns (`usa_wind`, radii, etc.)
NaN; IBTrACS rows carry their full Stage 0 schema with `source_name =
"ibtracs_best_track"` and the IBTrACS `iso_time` column renamed to
`snapshot_time_utc`.

## Running the preprocessor

### Step 0 — Verify paths

```bash
# Check that raw data is present
ls $SCRATCH/tcfuse/data/raw/tc_primed/
ls $SCRATCH/tcfuse/data/raw/ibtracs/    # raw CSV used by Stage 0
```

For local runs, `$SCRATCH` must be set; or override `paths.scratch` on the command line.

### Step 1 — Stage 0 IBTrACS preprocessing

```bash
python scripts/preprocess/prepare_ibtracs.py
# Optional Jean-Zay run (no SLURM submission needed; it's a single-process job)
python scripts/preprocess/prepare_ibtracs.py paths=jz
```

This must run **before** any Stage 1 preprocessor: every per-source preprocessor
loads `atcf_to_sid.csv` from disk to discard files with unknown ATCF IDs.

### Step 2 — Stage 1 per-source preprocessors

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py submitit=false
python scripts/preprocess/tc_primed/prepare_infrared.py submitit=false
python scripts/preprocess/tc_primed/prepare_radar.py submitit=false
python scripts/preprocess/sar/prepare_sar.py submitit=false
```

To limit to a subset during development, pass `include_seasons=[2020]` or similar via the
config override.

### Step 3 — Jean-Zay production run of source preprocessors

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py paths=jz setup=jz_cpu
```

See [tcfuse-jz](../tcfuse-jz/SKILL.md) for preflight checks (quota, env, W&B mode) before submitting.

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
print(df.head())  # sid, source_name, snapshot_time_utc, season, basin, subbasin
print(df["source_name"].value_counts())
```

### Step 5 — Stage 2 assembly

Reads the Stage 0 artifacts and every Stage 1 `index.parquet`; the storm set
is exactly `set(ibtracs_snapshots["sid"])`. The raw IBTrACS CSV is **not** read.

```bash
# Local debug run (no SLURM)
python scripts/preprocess/assemble.py submitit=false num_workers=4

# Jean-Zay production run
python scripts/preprocess/assemble.py paths=jz setup=jz_cpu
```

Key options:
- `skip_existing=true` — resume a partial run without rewriting existing storm files
- `chunk_size=200` — number of storms per SLURM job (SLURM mode only)

Validate Stage 2 output:

```python
from pathlib import Path
from tcfuse.data.sources import StormData
import pandas as pd

assembled_root = Path("$SCRATCH/tcfuse/data/preprocessed/assembled")

# Spot-check one assembled storm
sid = next((assembled_root / "storm_data").glob("*.h5")).stem
sd = StormData.from_disk(assembled_root, sid)
print(sd.storm_id, sd.basin, sd.subbasin, sd.season)
print({k: v.values.shape for k, v in sd.sources.items()})

# Check concatenated assembled index
df = pd.read_parquet(assembled_root / "index.parquet")
print(df["source_name"].value_counts())
# IBTrACS-specific columns (usa_wind, …) are populated on best-track rows only.
print(df.loc[df["source_name"] == "ibtracs_best_track", "usa_wind"].describe())
```

### Step 6 — Stage 3 train/val/test splits

After all desired sources are preprocessed and assembled, run:

```bash
# Local run
python scripts/preprocess/build_splits.py

# Jean-Zay
python scripts/preprocess/build_splits.py paths=jz
```

This reads the assembled `index.parquet` and builds one model sample per valid
`ibtracs_best_track` assimilation window. By default, each sample is centred on
`init_time_utc` (t₀) with lead hours `[-6, 0, 6, 12, 18, 24]` relative to t₀;
finite `usa_wind`, `usa_sshs`, `lat`, and `lon` are required at `-6h`, `0h`, and
`+24h`, while intermediate leads may be missing or NaN. Per-lead columns use
signed prefixes such as `lead_-006h_usa_wind` and `lead_+000h_lat`.

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
    usa_wind:
      mean: 62.1
      std: 28.4
      count: 85000
```

Masked pixels (where `mask == False`) and non-finite values are excluded from both
statistics and histogram samples. `ibtracs_best_track` is included even though it has
no `metadata.yaml` in `sources_root`; its channels are discovered from the HDF5 attrs.

## Adding a new dataset preprocessor

1. Add source definitions to a new `conf/data/<name>.yaml` following the TC-PRIMED template.
2. Create `scripts/preprocess/<name>.py` modelling it on `scripts/preprocess/tc_primed/prepare_pmw.py`.
3. Add the dataset to the table at the top of this skill.
4. Update the dataset stack table in `.cursor/rules/tcfuse-core.mdc` with the confirmed `$SCRATCH` path once known.

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

## Maintenance

When changing any file under `src/tcfuse/data/sources/` or any script under `scripts/preprocess/`,
update this skill in the same PR. If triggers or behavior rules change, also update
`.claude/commands/preprocess.md` and the dataset table in `.cursor/rules/tcfuse-core.mdc`.
