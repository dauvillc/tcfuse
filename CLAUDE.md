# CLAUDE.md — TC Multi-Source Fusion Project

> Read this file at the start of every session before doing anything else.
> Update it whenever a new architectural decision, data convention, or workflow rule is established.

---

## Project overview

This project develops a machine learning framework for **tropical cyclone (TC) analysis and forecasting** by fusing heterogeneous observation sources. The central scientific idea is that each observation source — regardless of its physical nature — can be represented as a set of measurements associated with explicit spatio-temporal coordinates. This coordinate-aware multi-source representation is the backbone of the entire framework.

**Target tasks (in order of priority):**
1. Rapid intensification (RI) forecasting at lead times of +6 h to +48 h
2. High-resolution inner-core wind field and pressure reconstruction
3. Microwave image interpolation from sparse multi-source satellite passes
4. Track forecasting (downstream, via fine-tuning)

This is a standalone research project. It is not a version or extension of any prior system — it is designed from first principles with efficiency, modularity, and multi-architecture support in mind.

---

## Core data abstraction

The fundamental unit of this framework is a **source**: a collection of measurements of the same physical quantity, acquired by the same instrument or model, at the same nominal time, sharing a common coordinate system.

Every source, regardless of its dimensionality, is represented as a set of **(value, coordinate)** pairs:

| Source type | Example sources | Value shape | Coordinate channels |
|---|---|---|---|
| **0D** (scalar) | Best-track, buoy point obs | `(1,)` | time, lat, lon |
| **1D** (vertical profile) | Dropsonde, Argo float | `(L, C)` — L levels, C channels | time, lat, lon, altitude/depth |
| **2D** (image / field) | PMW satellite, IR geostationary, ERA5 field | `(H, W, C)` | time (scalar), lat grid `(H, W)`, lon grid `(H, W)` |

**Hard constraints — never violate these:**
- Coordinates are always **continuous and physical** (degrees, seconds since epoch, meters). No learned bin embeddings for coordinates.
- Coordinates are stored **alongside** values, not inferred from array indices.
- A source may have **missing values** (NaN-masked); the framework must handle this gracefully.
- The number of sources per sample is **variable**. No fixed-size source list.

---

## Dataset stack

### Active datasets

| Dataset | Content | Format | Location |
|---|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors, 37 & 89 GHz), IR geostationary, ERA5 surface fields, DPR radar, best-track. 1987–2024, 3,552 storms, 242k+ overpasses | NetCDF, AI-ready | `TODO: $SCRATCH/tc_primed/` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR (Sentinel-1) surface wind speeds | NetCDF | `TODO: $SCRATCH/cyclobs/` |
| **NOAA AOML Dropsondes** | Vertical profiles (P, T, RH, u, v) from hurricane reconnaissance. ~13k sondes, North Atlantic + East Pacific | WMO TEMP DROP ASCII or NetCDF | `TODO: $SCRATCH/dropsondes/` |
| **Argo floats** | T/S profiles 0–2000 m depth, ~100k profiles/year globally. Key for upper ocean heat content | NetCDF (per-profile or gridded) | `TODO: $SCRATCH/argo/` |

### Deferred datasets

| Dataset | Reason |
|---|---|
| GFS / ECMWF HRES NWP fields | High value but requires careful train/test temporal split; integrate after baseline is stable |
| GPM DPR full 3D swaths | Partially in TC-PRIMED; full 3D integration deferred |
| TC-OBS, Saildrones | Limited coverage, derived labels — defer |

---

## On-disk preprocessed format

Preprocessed data is organised **source-first** under `cfg.paths.preprocessed_sources`:

```
<preprocessed_sources>/          ← cfg.paths.preprocessed_sources
├── pmw_amsr2_gcomw1/
│   ├── index.parquet            ← one row per snapshot; fast lookup without opening HDF5 files
│   └── snapshots/
│       └── {storm_id}_{YYYYMMDDTHHMMSSZ}.h5
├── pmw_ssmis_f18/
│   ├── index.parquet
│   └── snapshots/
│       └── ...
├── era5_surface/
│   └── ...
└── best_track/
    └── ...
```

Each HDF5 file holds **exactly one source** plus storm-level metadata in root attributes.
Use `source_snapshot_path(sources_root, source_name, storm_id, snapshot_time_utc)` from
`src/tcfuse/utils/io.py` to compute canonical paths.

**HDF5 per-snapshot structure:**

```
/
├── attrs: {storm_id, basin, snapshot_time_utc, lat, lon, vmax_kt, mslp_hpa, …}
├── scalar/
│   └── {source_name}/  values float32 (C,)       coords float64 (3,)      [time_unix_s, lat, lon]
├── profile/
│   └── {source_name}/  values float32 (L, C)      coords float64 (L, 4)   [time_unix_s, lat, lon, alt_m]
└── field/
    └── {source_name}/  values float32 (H, W, C)   coords float32 (H, W, 3) [time_unix_s broadcast, lat, lon]
```

Each source sub-group also has an optional `mask` bool dataset (same leading shape as `values`) and attrs `{source_name, channels}`.

**Key conventions:**
- Each HDF5 contains exactly one source; the source group is absent when data is unavailable (never empty groups).
- `source_name` identifies the source (e.g. `"best_track"`, `"era5_surface"`) — not tied to a raw dataset name.
- Index `source_name` column is a single string matching the source directory name.
- Per-source `index.parquet` files carry **no** split column. Train/val/test assignment is done separately by `scripts/preprocess/build_splits.py`, which merges all source indexes and writes `{preprocessed_sources}/train.parquet`, `val.parquet`, `test.parquet`.

All I/O goes through `src/tcfuse/utils/io.py` (`write_snapshot`, `read_snapshot`, `read_snapshot_meta`, `source_snapshot_path`).
Use `/preprocess` for the full preprocessing workflow and dataset inventory.

---

## Repository structure

```
project_root/
├── CLAUDE.md                  ← this file
├── conf/                      ← Hydra configuration tree
│   ├── config.yaml            ← top-level defaults
│   ├── data/                  ← dataset and source configs
│   ├── model/                 ← architecture configs
│   ├── paths/                 ← environment-specific path configs (local.yaml, jz.yaml)
│   ├── training/              ← optimizer, scheduler, loss configs
│   └── experiment/            ← named experiment overrides
├── src/
│   └── tcfuse/
│       ├── data/
│       │   ├── sources/           ← one module per source type (tcprimed, argo, dropsonde, …)
│       │   ├── collocation.py     ← spatiotemporal window queries
│       │   ├── transforms.py      ← normalization, coordinate encoding
│       │   └── dataset.py         ← PyTorch Dataset / LightningDataModule
│       ├── model/
│       │   ├── embeddings/        ← value embedders per source type (0D, 1D, 2D)
│       │   ├── encoders/          ← interchangeable backbone architectures
│       │   ├── decoders/          ← task heads (regression, generative, classification)
│       │   └── model.py           ← top-level LightningModule
│       ├── training/
│       │   ├── losses.py
│       │   └── callbacks.py
│       └── utils/
│           ├── coords.py          ← coordinate utilities (projections, normalization)
│           └── io.py              ← NetCDF / HDF5 helpers
├── scripts/
│   ├── preprocess/            ← data download and preprocessing scripts
│   └── slurm/                 ← Jean-Zay job submission scripts (see section below)
├── tests/
│   ├── test_sources.py
│   ├── test_embeddings.py
│   └── test_model.py
└── notebooks/                 ← exploration only, never imported by src/
```

**Rules:**
- Nothing in `notebooks/` is ever imported by `src/`.
- Each source embedder must be unit-testable with a **synthetic tensor** — no real data required for tests.
- `conf/experiment/` overrides are the only place where full run configurations are assembled. Never hardcode experiment parameters in `src/`.
- Path resolution is handled by `conf/paths/`. Select the environment at launch: `paths=local` (default) for local debugging, `paths=jz` on Jean-Zay. All code must reference paths via `cfg.paths.*` — never hardcode filesystem paths.

---

## Tech stack and coding conventions

**Core stack:** Python 3.10+, PyTorch, PyTorch Lightning, Hydra (config), Weights & Biases (logging).

**Coding rules — always follow these:**
- Type hints on all function signatures.
- Docstrings on all public classes and functions (one-line summary + args/returns for non-trivial ones).
- No magic numbers anywhere in `src/`. Use named constants or config values.
- Tensor shapes must be documented in docstrings as comments: `# (B, L, C)`.
- All configurable hyperparameters live in `conf/`. Code reads from config; config does not live in code.

**W&B conventions:**
- Project name: `TODO`
- Entity: `TODO`
- Run naming: `{architecture}_{sources}_{task}_{date}` — e.g. `perceiver_pmw-era5-argo_ri_20250901`
- Always log: source types present, number of training samples, GPU memory peak, val metrics per task head.

---

## Architecture philosophy

The framework is **architecture-agnostic at the backbone level**. The embedding layer (value + coordinate → token) and the task heads (decoder) are fixed interfaces; the encoder between them is swappable. This makes it straightforward to benchmark multiple architectures without rewriting data loading or training logic.

```
[Source 1: values + coords] ──┐
[Source 2: values + coords] ──┼──► [Source Embeddings] ──► [Encoder (swappable)] ──► [Task Head]
[Source N: values + coords] ──┘
```

**The encoder interface:**
- Input: a list of token sequences, one per source, each of shape `(B, N_i, D)` where `N_i` is the number of tokens for source `i` and `D` is the embedding dimension.
- Output: a representation that the task head can query — exact form depends on architecture (latent array for Perceiver, CLS token for ViT-style, etc.).
- The encoder must be instantiable from a Hydra config node.

**Candidate architectures to benchmark** (add to this list as needed):
- Perceiver / Perceiver IO
- Cross-attention Transformer (queries from anchor points or task positions)
- Hierarchical windowed attention (Swin-style, per source + cross-source)

**Self-supervised training task:** randomly mask one source at training time; reconstruct its values from all remaining sources, using only its coordinates and instrument metadata as queries. This is the default pre-training objective. Supervised fine-tuning follows for specific tasks.

---

## Jean-Zay cluster

Use the `/jz` skill for all cluster operations (storage layout, environment setup, W&B sync, SLURM parameters, checkpoint/resume, preflight checks).

SLURM parameters live in `conf/setup/jz_<hw>.yaml`; job submission uses `submitit.AutoExecutor` in `scripts/train.py` and `scripts/preprocess/<source>.py`. There is no manual bash SLURM template.

---

## Workflow rules for Claude Code

1. **Read this file first** at the start of every session, before reading any other file or writing any code.
2. **Plan before implementing.** For any non-trivial task, propose a plan (module structure, interface, test strategy) and wait for approval before writing code.
3. **One module at a time.** Implement and test one component fully before moving to the next.
4. **Update this file** when any of the following happens:
   - A new source type or embedding convention is decided → update the data abstraction table
   - A new architecture is added → update the architecture section
   - A new SLURM script is written → add it to the scripts table
   - A new dataset path is confirmed → update the dataset table
   - A new W&B convention is established → update the logging section
5. **Ask, don't guess** on design decisions not covered by this file. Especially: tensor layout, coordinate normalization strategy, masking implementation, task head interface.
6. **Prefer explicit over implicit.** If a function's behavior depends on the presence or absence of a source, make that conditional explicit in the code and documented in the docstring.
7. **Tests are not optional.** Every new embedding module ships with a unit test using a synthetic `(B, N, C)` tensor. Every data loader change ships with a test that runs on a 10-sample subset.
