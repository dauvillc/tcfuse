# TC-Fuse preprocessing pipeline

Claude Code: invoke `/preprocess` (reads this skill).

## Pipeline at a glance

The pipeline runs in four explicit stages. Each stage reads only the outputs
of previous stages — never raw inputs from earlier ones.

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| 0 | `prepare_ibtracs.py` | raw IBTrACS CSV | `ibtracs/ibtracs.parquet`, `ibtracs/atcf_to_sid.csv` |
| 1 | `prepare_pmw.py`, `prepare_infrared.py`, `prepare_radar.py`, `prepare_era5.py`, `sar/prepare_sar.py` | raw TC-PRIMED / CyclObs files, Stage 0 `atcf_to_sid.csv` | per-source HDF5 snapshots + `index.parquet` |
| 2 | `assemble.py` | Stage 0 `ibtracs.parquet`, Stage 1 indices + snapshots | `storm_data/{sid}.h5`, `index.parquet` (uniform schema, all sources) |
| 3A | `build_splits.py` | Stage 2 `index.parquet` | `train.parquet`, `val.parquet`, `test.parquet` (source-snapshot rows, split by season) |
| 3B | `build_windows.py` | Stage 3A split parquets | `{windows_name}/train_windows.parquet` etc. (long-format window index) |
| 4 | `compute_normalization.py` | Stage 3A `train.parquet` + Stage 2 `index.parquet` | normalization stats |

## When to use

- Writing or modifying a per-source preprocessor under `scripts/preprocess/`.
- Running `prepare_ibtracs.py`, `assemble.py`, `build_splits.py`, `build_windows.py`, or `compute_normalization.py`.
- Reading or writing `Source` / `StormData` HDF5 files.
- Understanding the on-disk Stage 0 / Stage 1 / Stage 2 layouts and the assembled `index.parquet`.
- Adding a new dataset to the project.

For Jean-Zay submission, preflight, or SLURM setup, see [jz.md](jz.md).
For forecast output storage (predictions, not preprocessing), see [predictions/skill.md](predictions/skill.md).

**Coding style:** follow project-wide rules in [`.agents/context.md`](context.md) § Human-readable code (priority). Preprocess-specific layout below.

## Preprocess file layout

- Entry script → `process_*_file` worker (multiprocessing boundary) → `main`.
- Shared infra: `utils/runner.py`, `utils/regridding.py`, `utils/field_grid.py`, `tc_primed/utils.py`.
- Model new preprocessors on `prepare_pmw.py` after the readability refactor.

**Pipeline invariants** (keep; do not add other validation without asking):

- IBTrACS ATCF→SID resolution and NaN lat/lon skip when building SCALAR sources.
- Train-only normalization (no val/test leakage).
- Preprocessing scripts must use non-batched `Source` snapshots (`batched=False`); batched `Source` is reserved for ML dataset/collate/model flow.

## Dataset inventory

| Dataset | Kind | Sources extracted | Status | Raw path config key |
|---|---|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR geostationary, ERA5 surface, best-track | `best_track` (0D), `pmw_*` (2D), `ir_goes` (2D), `era5_surface` (2D) | Preprocessor implemented | `paths.raw_datasets.tc_primed` |
| **CyclObs** | C-band SAR (Sentinel-1) surface winds | `sar_cband` (2D FIELD) | `scripts/preprocess/sar/prepare_sar.py` | `paths.raw_datasets.cyclobs` |
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
deduplicated on `[sid, usa_atcf_id]`. When a SID maps to multiple `USA_ATCF_ID`
values, the mapping keeps the ATCF ID with the highest max `USA_WIND`.

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
Use `Source.path(sources_root, source_name, sid, time_utc)` to compute canonical paths.

**TC-PRIMED ERA5 surface (native rectilinear grid):** `prepare_era5.py` outputs
`era5_surface` snapshots at 121×121 (30° patch, ±15°, 0.25°/px). One file per storm
covers ~70 synoptic times; each time step becomes its own HDF5 snapshot. Channels:
`precipitable_water, rain_large_scale, rain_convective, sst, pressure_msl,
temperature_2m, dewpoint_2m, u_wind_10m, v_wind_10m` (all 2D surface fields from the
`rectilinear` group; 3D pressure-level fields are not extracted).

**TC-PRIMED infrared (native grid fixed size):** `prepare_infrared.py` outputs a
storm-centered square on each source's regular lat/lon grid (no regridding):
`ir_tcirar` always 401×401 (±200 px), `ir_hursat` always 201×201 (±100 px). Native
grids larger than the target are center-cropped; smaller axes get symmetric NaN padding.

**CyclObs SAR (native grid fixed size):** `sar/prepare_sar.py` always outputs 401×401
(±200 px) with the same center-crop / symmetric NaN-pad rules on the native grid.

**TC-PRIMED PMW and radar (storm-centered equiangular grids):** `prepare_pmw.py` and
`prepare_radar.py` bilinearly resample swaths onto a fixed Plate Carrée grid centered on
`overpass_storm_metadata` `storm_latitude` / `storm_longitude`, with half-width
`cfg.tc_primed.storm_grid_extent_half_km` (default 750 km) along each axis and pixel
spacing equal to the minimum IFOV (km) for that sensor/swath (`tc_primed_ifovs.yaml`).
Grid shape is `(2 × round(extent_half_km / resolution_km), …)`, stored as `SourceMetadata.shape`
and also recorded in source `char_vars` as `grid_shape_yx` together with `target_resolution_km`
and `storm_grid_extent_half_km`. Implementation: `scripts/preprocess/utils/regridding.py`.

**ATCF→SID translation at Stage 1:** every per-source preprocessor calls
`load_translation(sources_root)` to load the Stage 0 ATCF→SID lookup, then in the
per-file worker:

1. Reads the dataset-native ATCF ID (e.g. `storm_id` in TC-PRIMED, `sid` in CyclObs).
2. Looks up the IBTrACS SID; if missing, the worker emits `warnings.warn(...)` and
   **discards** the file — no snapshot is written.
3. Uses the SID in the HDF5 filename and writes `Source.meta = {"storm_id": sid,
   "time_utc": <iso>}`. **`storm_lat` and `storm_lon` are no longer written**
   to `Source.meta` — per-pixel coordinates already live in `Source.coords`.

**Per-source `index.parquet` schema (canonical for every source):**

```
sid, source_name, time_utc, season, basin, subbasin
```

The index is rebuilt at the end of every Stage 1 run by `finalize_source` in
`scripts/preprocess/utils/runner.py`, which scans `snapshots/*.h5`, reads each
file's `storm_id` (=SID) and `time_utc` root attrs, and left-joins
the Stage 0 translation table to populate `season / basin / subbasin`. Metadata
is written via `SourceMetadata.to_yaml`; the index is written separately as
``index.parquet`` in the same source directory.

**HDF5 file structure per snapshot:**
```
/
├── attrs: {storm_id (=SID), time_utc, …}     ← Source.meta
├── scalar/
│   └── {source_name}/
│       ├── values    float32 (C,),       gzip-4
│       ├── coords    float64 (3,),       gzip-4  [time_unix_s, lat, lon]
│       ├── mask      bool    (C,)
│       └── attrs:    {source_name, batched (bool), channels (JSON), char_vars (JSON)}
├── profile/
│   └── {source_name}/
│       ├── values    float32 (L, C),     gzip-4
│       ├── coords    float64 (L, 4),     gzip-4  [time_unix_s, lat, lon, alt_m]
│       ├── mask      bool    (L, C)
│       └── attrs:    {source_name, batched (bool), channels (JSON), char_vars (JSON)}
└── field/
    └── {source_name}/
        ├── values    float32 (H, W, C),  gzip-4
        ├── coords    float32 (H, W, 3),  gzip-4  [time_unix_s broadcast, lat, lon]
        ├── mask      bool    (H, W, C)
        └── attrs:    {source_name, batched (bool), channels (JSON), char_vars (JSON)}
```

**Missing sources:** simply absent from the HDF5 file — no empty groups.

**metadata.yaml schema (per source):**
```yaml
name: pmw_amsr2_gcomw1
type: microwave      # physical category
kind: field          # scalar | profile | field
channels: [tb_36.5h, tb_36.5v, tb_a89.0h, tb_a89.0v]
num_channels: 4
shape: [400, 400]    # spatial dims shared by every snapshot: [] SCALAR, [L] PROFILE, [H, W] FIELD
char_vars:
  ifov: {tb_36.5h: [7.2, 4.4, 7.2, 4.4], …}
```

`SourceMetadata.shape` is the canonical way to know a snapshot's spatial dimensions without
loading any HDF5 file. `SourceMetadata.num_tokens` derives from it (`math.prod(shape)`, 1 for
SCALAR). For PMW/radar, `shape` is consistent with `char_vars["grid_shape_yx"]`.

## Stage 2 — Assembled format

Config key: `cfg.paths.preprocessed_data`

```
${paths.preprocessed_data}/
├── storm_data/
│   └── {sid}.h5            # one file per IBTrACS storm, e.g. 2016228N14275.h5
├── index.parquet           # concatenated assembled index (Stage 1 + IBTrACS rows)
└── sources_metadata.yaml   # merged source descriptors (channels, shape, kind)
```

`SourceMetadata` and `MultisourceMetadata` hold static descriptors only — no snapshot
index. Stage-1 per-source indices live in ``index.parquet`` under each source directory;
the assembled snapshot index is ``index.parquet`` at this root. Load descriptors for
training via ``MultisourceMetadata.from_yaml(assembled_root / "sources_metadata.yaml")``.
Produced by ``assemble.py`` from Stage-1 ``metadata.yaml`` files.

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
        ├── mask      bool (always present; same shape as values)
        └── attrs:
            ├── source_name        str
            ├── channels           JSON list
            ├── char_vars          JSON list
            ├── kind               "SCALAR" | "PROFILE" | "FIELD"
            ├── time_utc  isoformat str (for round-trip key recovery)
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

**`StormData` sources dict key:** `(source_name, time_utc)` where
`time_utc` is the isoformat string as it appears in per-source `index.parquet`.

**Assembled `index.parquet` schema:** the concatenation of two row blocks
(satellite snapshots first, IBTrACS rows second), sharing a union schema with
columns in this order:

```
sid, source_name, time_utc, season, basin, subbasin,
name, number, nature, lat, lon,
usa_atcf_id, usa_wind, usa_pres, usa_sshs,
usa_r34_*, usa_r50_*, usa_r64_*
```

Satellite rows leave the IBTrACS-specific columns (`usa_wind`, radii, etc.)
NaN; IBTrACS rows carry their full Stage 0 schema with `source_name =
"ibtracs_best_track"` and the IBTrACS `iso_time` column renamed to
`time_utc`.

## Running the preprocessor

### Pixi tasks

Preprocessing steps are exposed as local-only Pixi tasks in `pixi.toml` (Jean-Zay
uses direct `python` invocations with `paths=jz`; see [jz.md](jz.md)).
All tasks pass `paths=local setup=local submitit=false`. Extra Hydra overrides append
to any task, e.g. `pixi run preprocess-pmw include_seasons=[2020]`.

Local paths resolve via `paths.scratch` in `conf/paths/local.yaml`
(default `${paths.scratch}/data/...` under `/home/cdauvill/scratch/tcfuse`); ensure raw
data exists there or override `paths.scratch` on the command line.

| Task | Stage / script |
|---|---|
| `preprocess-ibtracs` | 0 — `prepare_ibtracs.py` |
| `preprocess-pmw` | 1 — `tc_primed/prepare_pmw.py` (depends on ibtracs) |
| `preprocess-infrared` | 1 — `tc_primed/prepare_infrared.py` |
| `preprocess-radar` | 1 — `tc_primed/prepare_radar.py` |
| `preprocess-era5` | 1 — `tc_primed/prepare_era5.py` |
| `preprocess-sar` | 1 — `sar/prepare_sar.py` |
| `preprocess-assemble` | 2 — `assemble.py` (`num_workers=4`) |
| `preprocess-splits` | 3A — `build_splits.py` (season split) |
| `preprocess-windows` | 3B — `build_windows.py` (window index, configurable via `windows_setup=`) |
| `preprocess-normalization` | 4 — `compute_normalization.py` |
| `preprocess-stage0` … `preprocess-stage3` | Composites for stages 0–3 |
| `preprocess` | Full pipeline (stage0 → … → stage3) |

Examples:

```bash
pixi run preprocess                    # full pipeline
pixi run preprocess-stage1             # IBTrACS + all Stage 1 sources
pixi run preprocess-pmw include_seasons=[2020]
```

Raw downloads (`download_tc_primed.py`, `download_sar_cyclobs.py`) are not wired
as Pixi tasks; invoke them manually when needed.

### Step 0 — Verify paths

Confirm raw data exists at `$SCRATCH/tcfuse/data/raw/{tc_primed,ibtracs}/`. For local runs, `$SCRATCH` must be set or override `paths.scratch` on the command line.

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
python scripts/preprocess/tc_primed/prepare_era5.py submitit=false
python scripts/preprocess/sar/prepare_sar.py submitit=false
```

To limit to a subset during development, pass `include_seasons=[2020]` or similar via the
config override.

### Step 3 — Jean-Zay production run of source preprocessors

```bash
python scripts/preprocess/tc_primed/prepare_pmw.py paths=jz setup=jz_cpu
```

See [jz.md](jz.md) for preflight checks (quota, env, W&B mode) before submitting.

### Step 4 — Validate Stage 1 output

Spot-check a snapshot with `Source.from_disk(p)` and the index with `pd.read_parquet(sources_root / "pmw_amsr2_gcomw1" / "index.parquet")` — see I/O API reference below.

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

Spot-check with `StormData.from_disk(assembled_root, sid)` and the assembled index with `pd.read_parquet(assembled_root / "index.parquet")` — see I/O API reference below.

### Step 6 — Stage 3A train/val/test splits

After all desired sources are preprocessed and assembled, run:

```bash
# Local run
python scripts/preprocess/build_splits.py

# Jean-Zay
python scripts/preprocess/build_splits.py paths=jz
```

This reads the assembled `index.parquet` and partitions rows by season.
IBTrACS is treated identically to any other source — no windowing happens here.

Split assignment is controlled by `conf/preproc.yaml`:
- **val**:   seasons in `cfg.splits.val`
- **test**:  seasons in `cfg.splits.test`
- **train**: all remaining seasons

Output files (same uniform schema as `index.parquet`):

```
${paths.preprocessed_data}/
├── index.parquet       # canonical source-snapshot index (all seasons)
├── train.parquet       # source-snapshot rows for training seasons
├── val.parquet
└── test.parquet
```

### Step 7 — Stage 3B window indexes

Build training samples (windows) from the split files using a named window
configuration from `conf/windows_setup/`:

```bash
# Default config (ibtracs_forecast_24h)
python scripts/preprocess/build_windows.py

# Jean-Zay / custom config
python scripts/preprocess/build_windows.py paths=jz windows_setup=ibtracs_forecast_24h
```

Each snapshot from one of the `target_sources` defines a window.  Input
snapshots are governed by the `input_sources` specification: a snapshot from the
same storm is collected only when its source *type* is listed AND its `time_utc`
falls within one of that type's periods.  A window is discarded entirely when any
listed period contains fewer than its `min_required` snapshots (use `0` to
include a source without requiring it).  The window's overall span is the union
of all periods.  The target snapshot is always present as `is_target = True`
even when it falls outside that span.

Window config keys (see `conf/windows_setup/*.yaml`):
- `name` — subdirectory name for outputs
- `target_sources` — list of source names whose snapshots anchor windows
- `input_sources` — mapping `{source_type: [[start_offset, end_offset, min_required], ...]}`.
  Offsets are `pd.Timedelta`-parseable strings (typically negative). Type keys
  match `source_name` by prefix (`"era5"` → `era5_surface`; `"pmw"` → `pmw_gmi`
  / `pmw_tmi`, with `min_required` counted across all matching sources).

Output (long-format, one row per window × source snapshot):

```
${paths.preprocessed_data}/{windows_name}/
├── train_windows.parquet
├── val_windows.parquet
└── test_windows.parquet
```

Output schema: `window_id | sid | basin | subbasin | season | usa_atcf_id |
window_start_time_utc | window_end_time_utc | window_ref_time_utc |
source_name | time_utc | is_target`

### Step 8 — Compute normalization statistics

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
2. Create `scripts/preprocess/<name>.py` modelling it on `scripts/preprocess/tc_primed/prepare_pmw.py` (post-refactor layout; see "Preprocess file layout" above).
3. Add the dataset to the table at the top of this skill.
4. Update the dataset stack table in [`.agents/context.md`](context.md) with the confirmed `$SCRATCH` path once known.

## I/O API reference

All read/write operations go through `src/tcfuse/data/sources/`:

| Class / function | Location | Purpose |
|---|---|---|
| `Source.write(path)` | `sources/source.py` | Write one Source to a self-contained HDF5 file |
| `Source.from_disk(path)` | `sources/source.py` | Load Source + meta from HDF5 |
| `Source.read_meta(path)` | `sources/source.py` | Read root attrs only (no tensors) |
| `Source.path(sources_root, source_name, storm_id, time_utc)` | `sources/source.py` | Compute canonical per-source snapshot path |
| `Source.to_hdf5_group(group)` | `sources/source.py` | Low-level: write tensors into an open HDF5 group |
| `Source.from_hdf5_group(group, kind)` | `sources/source.py` | Low-level: read tensors from an open HDF5 group |
| `StormData.write(assembled_root)` | `sources/storm_data.py` | Write assembled per-storm HDF5 (all sources) |
| `StormData.from_disk(assembled_root, storm_id)` | `sources/storm_data.py` | Load all sources for a storm |
| `StormData.read_meta(assembled_root, storm_id)` | `sources/storm_data.py` | Read root attrs only (no tensors) |
| `StormData.path(assembled_root, storm_id)` | `sources/storm_data.py` | Canonical assembled file path |
| `SourceMetadata.to_yaml(yaml_path)` | `sources/metadata.py` | Write one source's metadata.yaml |
| `SourceMetadata.from_yaml(yaml_path)` | `sources/metadata.py` | Load metadata.yaml (descriptors only, no index) |
| `MultisourceMetadata.from_yaml(yaml_path)` | `sources/metadata.py` | Load assembled sources_metadata.yaml |
| `MultisourceMetadata.from_multiple_yaml(paths)` | `sources/metadata.py` | Union several per-source metadata.yaml files |
| `MultisourceMetadata.to_yaml(yaml_path)` | `sources/metadata.py` | Write merged sources_metadata.yaml |

## Maintenance

When changing any file under `src/tcfuse/data/sources/` or any script under `scripts/preprocess/`,
update this skill in the same PR. If triggers or behavior rules change, also update
`.claude/commands/preprocess.md` and the dataset table in [`.agents/context.md`](context.md).
