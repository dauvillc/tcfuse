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
| **0D** (scalar) | Best-track, buoy point obs | `(C,)` | time, lat, lon |
| **1D** (vertical profile) | Dropsonde, Argo float | `(L, C)` — L levels, C channels | time, lat, lon, altitude/depth |
| **2D** (image / field) | PMW satellite, IR geostationary, ERA5 field | `(H, W, C)` | time (scalar), lat grid `(H, W)`, lon grid `(H, W)` |

**Hard constraints — never violate these:**
- Coordinates are always **continuous and physical** (degrees, seconds since epoch, meters). No learned bin embeddings for coordinates.
- Coordinates are stored **alongside** values, not inferred from array indices.
- A source may have **missing values** (NaN-masked); the framework must handle this gracefully.
- The number of sources per sample is **variable**. No fixed-size source list.

---

## Dataset stack

Use `/preprocess` for the full dataset inventory (including deferred datasets) and preprocessing workflow.

| Dataset | Content | Location |
|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR, ERA5, DPR, best-track. 1987–2024, 3,552 storms | `TODO: $SCRATCH/tc_primed/` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR surface winds | `TODO: $SCRATCH/cyclobs/` |
| **NOAA AOML Dropsondes** | Vertical profiles (P, T, RH, u, v), ~13k sondes | `TODO: $SCRATCH/dropsondes/` |
| **Argo floats** | T/S profiles 0–2000 m, ~100k profiles/year | `TODO: $SCRATCH/argo/` |

---

## On-disk preprocessed format

Use `/preprocess` for the full format spec and dataset inventory. Quick reference:
- **Stage 1** (`cfg.paths.preprocessed_sources`): one HDF5 per source snapshot — `Source.path(sources_root, source_name, storm_id, time)`
- **Stage 2** (`cfg.paths.preprocessed_data`): one HDF5 per storm — `StormData.path(assembled_root, storm_id)`
- I/O API: `Source`, `StormData`, `SourceMetadata`, `MultisourceMetadata` in `src/tcfuse/data/sources/`
- Pipeline order: `assemble.py` → `build_splits.py` → `compute_normalization.py` (splits before normalization to prevent leakage)

---

## Repository structure

```
project_root/
├── conf/                      ← Hydra config tree (data/, model/, paths/, training/, experiment/)
├── src/tcfuse/
│   ├── data/
│   │   ├── sources/           ← Source, SourceKind, SourceMetadata, MultisourceMetadata, StormData
│   │   ├── collocation.py     ← spatiotemporal window queries
│   │   ├── transforms.py      ← normalization, coordinate encoding
│   │   └── dataset.py         ← PyTorch Dataset / LightningDataModule
│   ├── model/
│   │   ├── embeddings/        ← value embedders per source type (0D, 1D, 2D)
│   │   ├── encoders/          ← interchangeable backbone architectures
│   │   ├── decoders/          ← task heads (regression, generative, classification)
│   │   └── model.py           ← top-level LightningModule
│   ├── training/              ← losses.py, callbacks.py
│   └── utils/                 ← coords.py, archive.py
├── scripts/preprocess/        ← prepare_*.py, assemble.py, build_splits.py, compute_normalization.py
├── tests/
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
- Use inline comments liberally: add a short `# comment` above every logical code block (even small ones) to explain what it does.

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

Use `/jz` for all cluster operations (storage layout, environment setup, W&B sync, SLURM, checkpoint/resume, preflight checks). GPU configs (`jz_gpu_v100` / `jz_gpu_a100` / `jz_gpu_h100` / `jz_cpu`) are defined in `conf/setup/jz_<hw>.yaml`; job submission uses `submitit.AutoExecutor`. Archival to STORE is automatic when `archive: true` (all `jz_*` configs) — see `src/tcfuse/utils/archive.py`.

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
