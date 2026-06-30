# TC-Fuse Core Context

This project develops a machine learning framework for **tropical cyclone (TC) analysis and forecasting** by fusing heterogeneous observation sources. The central scientific idea is that each observation source — regardless of its physical nature — can be represented as a set of measurements associated with explicit spatio-temporal coordinates. This coordinate-aware multi-source representation is the backbone of the entire framework.

This is a standalone research project. It is not a version or extension of any prior system — it is designed from first principles with efficiency, modularity, and multi-architecture support in mind.

**Primary goal:** train a **self-supervised multi-source foundation model** for tropical cyclones — a single encoder that learns from heterogeneous TC observations (PMW, IR, ERA5, profiles, best-track, etc.) without task-specific labels.

The default pre-training objective is **masked-source reconstruction**: randomly mask one source at training time and reconstruct its values from all remaining sources, using only its coordinates and instrument metadata as queries (see Architecture philosophy below). Downstream supervised tasks — RI forecasting, track prediction, wind-field reconstruction, microwave interpolation — are future fine-tuning applications of this foundation model, not the current focus.

## On-demand skills (`.agents/`)

Read the matching skill file before doing work in that area. The Claude slash commands in `.claude/commands/` are thin redirects to these skills.

| Topic | Skill file | Claude slash command |
|---|---|---|
| Dataset preprocessing pipeline (per-source HDF5, assembled storms, splits, normalization) | [`.agents/preprocess.md`](preprocess.md) | `/preprocess` |
| Jean-Zay cluster operations (rsync, SLURM, monitoring, W&B sync, checkpoints) | [`.agents/jz.md`](jz.md) | `/jz` |
| CLEPS cluster operations (pixi, W&B online, persistent scratch, SLURM, monitoring) | [`.agents/cleps.md`](cleps.md) | `/cleps` |
| Publication-quality figures (style.py, SVG output, thematic plotting modules) | [`.agents/visualize.md`](visualize.md) | `/visualize` |
| Model backbone architecture (embedding/encoder/decoder design, candidate backbones, pre-training task) | [`.agents/architecture.md`](architecture.md) | `/architecture` |
| Inference and prediction pipeline (checkpoint loading, task masking, PredictionRun output, metrics) | [`.agents/inference.md`](inference.md) | `/inference` |
| Basedpyright diagnostics workflow | [`.agents/pyright-fixer.md`](pyright-fixer.md) | (none) |

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
- A source may have **missing values**; every `Source` carries a per-value availability mask with the same shape as `values`, where `True` means finite/available and `False` means NaN/missing.
- `Source` may be **batched** (leading `B` axis) only inside ML pipeline internals (dataset/collate/model flow). Preprocessing, evaluation, and visualization code paths must use non-batched `Source` objects.
- The number of sources per sample is **variable**. No fixed-size source list.
- All snapshots within a given source share the same **spatial shape** `(H, W)` for FIELD sources (or `()` / `(L,)` for SCALAR / PROFILE). The shape is stored in `SourceMetadata.shape` and therefore knowable without loading any HDF5 file.
- IBTrACS USA and WMO best-track quantities are **distinct definitions**; store them in separate channels/columns and preserve NaN rather than falling back or coalescing across providers.

## Dataset stack

| Dataset | Content | Location |
|---|---|---|
| **TC-PRIMED v01r01** | PMW (11 sensors), IR, ERA5, DPR, best-track. 1987–2024, 3,552 storms | `TODO: $SCRATCH/tc_primed/` |
| **CyclObs** | L-band (SMOS, SMAP) + C-band SAR surface winds | `TODO: $SCRATCH/cyclobs/` |
| **NOAA AOML Dropsondes** | Vertical profiles (P, T, RH, u, v), ~13k sondes | `TODO: $SCRATCH/dropsondes/` |
| **Argo floats** | T/S profiles 0–2000 m, ~100k profiles/year | `TODO: $SCRATCH/argo/` |

## On-disk preprocessed format

On-disk layouts (Stage 0–3, HDF5 schemas, index.parquet columns, I/O API) → [`.agents/preprocess.md`](preprocess.md).

## Repository structure

```
project_root/
├── conf/                      ← Hydra config tree (dataloader/, datamodule/, lightning_module/, model/, trainer/, optimizer/, lr_scheduler/, setup/, paths/, windows_setup/, experiment/)
├── src/tcfuse/
│   ├── data/
│   │   ├── sources/           ← Source, SourceKind, SourceMetadata, MultisourceMetadata, StormData
│   │   ├── collocation.py     ← spatiotemporal window queries
│   │   ├── transforms.py      ← normalization, coordinate encoding
│   │   ├── collate.py         ← collate_window_samples → WindowBatch
│   │   ├── dataset.py         ← TCWindowDataset (PyTorch map-style Dataset)
│   │   └── predictions/       ← SamplePrediction, PredictionRun (per-window pred+target HDF5 + index + metrics) — see /inference
│   ├── evaluation/            ← Evaluation plugins (base.py, flatten.py, quantitative/) run over a PredictionRun — see /evaluate
│   ├── lightning/
│   │   ├── datamodule.py           ← TCWindowDataModule (LightningDataModule)
│   │   ├── base_module.py          ← BaseLightningModule (general WindowBatch→WindowBatch, normalization, AdamW+cosine-LR; accepts backbone as nn.Module or Hydra partial)
│   │   ├── masked_reconstruction.py ← MaskedReconstructionLightningModule (general masked-source reconstruction; targets from WindowBatch.is_target, NaN masking, MSE loss; predict_step masks targets before reconstructing)
│   │   ├── prediction_writer.py     ← PredictionWriter (BasePredictionWriter → PredictionRun; single-GPU only) — see /inference
│   │   └── lr_scheduler.py         ← CosineAnnealingWarmupRestarts
│   ├── models/
│   │   └── encoders/               ← source embedding layer: EmbeddedSource/EmbeddedBatch (embedded.py), SourceEncoder (base.py), Scalar/Profile/Field patch-embed encoders (patch_embed.py), MultiSourceEncoder dispatcher WindowBatch→EmbeddedBatch (multisource.py)
│   ├── training/              ← losses.py, callbacks.py
│   └── utils/                 ← coords.py, archive.py
├── scripts/preprocess/        ← prepare_*.py, assemble.py, build_splits.py, compute_normalization.py
├── scripts/train/             ← train.py (Hydra+submitit, Checkpointable, ModelCheckpoint), profile_data.py
├── scripts/inference/         ← infer.py (Hydra checkpoint → split prediction → PredictionRun + metrics.csv) — see /inference
├── scripts/evaluation/        ← evaluate.py (Hydra PredictionRun → enabled Evaluation plugins → paths.results) — see /evaluate
├── tests/
└── notebooks/                 ← exploration only, never imported by src/
```

## Repository conventions

- Use `conf/` for Hydra configuration and `cfg.paths.*` for all paths.
- Assemble full run configs only in `conf/experiment/`.
- Path resolution is handled by `conf/paths/`. Select the environment at launch: `paths=local` (default) for local debugging, `paths=jz` on Jean-Zay, `paths=cleps` on CLEPS. All code must reference paths via `cfg.paths.*` — never hardcode filesystem paths.
- Never import anything from `notebooks/` into `src/`.
- Keep source embedders unit-testable with synthetic tensors only (no real data required for unit tests).
- Preserve the preprocessing order: `assemble.py`, `build_splits.py`, then `compute_normalization.py`.
- Pixi is the source of truth for local development and CI dependencies. Use `pixi add` or `pixi add --pypi` for dependency changes, and run checks through Pixi tasks (`pixi run typecheck`, `pixi run lint`, `pixi run test`, `pixi run format-check`).
- `requirements-jz.txt` is generated from `pixi.toml` for Jean-Zay only (which uses site modules, not Pixi). Regenerate it with `pixi run export-jz-requirements`; do not edit it by hand.
- Do not recreate `requirements.txt`, `requirements-ci.txt`, or `requirements-dev.txt`; CI uses Pixi directly.

## Coding rules

### Human-readable code (priority)

Human readability beats clever abstraction. A reader should understand any file top-to-bottom without jumping across many modules. This applies to **all** project code: `src/tcfuse/`, `scripts/`, `tests/`, and config with logic.

- **Factorization:** extract a function or module only if (a) reused in **2+ files**, or (b) the block is **~40+ lines** and naming it clarifies the main flow. Do **not** create one-liner helpers, docstring-heavy wrappers, or extra files used from a single call site.
- **Inline comments:** put a `# comment` **before every logical step** — aim for roughly **one comment line per 1–2 code lines** at non-trivial sites (reads, transforms, tensor ops, writes, control flow). Docstrings stay one-line summaries (+ Args/Returns when needed); detailed narration lives in **inline comments**, not in docstrings.
- **Validation policy:** this is a research codebase. Do **not** add `raise`/`assert`/defensive guards unless (i) the user asked, or (ii) it is a **documented invariant** below or in a skill. If unsure, **ask first**.

**Documented invariants** (keep; not defensive fluff):

- **Data abstraction:** `Source` tensor shape checks in `src/tcfuse/data/sources/source.py` — core API contract.
- **Preprocessing pipeline** (details in [`.agents/preprocess.md`](preprocess.md)): IBTrACS ATCF→SID resolution, NaN lat/lon skip for SCALAR sources, train-only normalization (no split leakage).

See also [`.agents/coding-style.md`](coding-style.md) for concrete do/don't examples.

### Python typing discipline

See [`.agents/python-typing.md`](python-typing.md) (read before editing any Python file). Key points:

- Preserve basedpyright correctness for changed code; run `pixi run typecheck` after substantive edits.
- Preserve linting correctness for changed code; run `pixi run lint` after any edit and fix all reported errors before finishing.
- Use Python 3.12 typing style; prefer precise types over `Any`.
- No `# type: ignore` / `# pyright: ignore` without explicit user approval.
- Warn the user before applying a cosmetic typing fix that might mask a real bug.
- Isolate third-party unknown-type noise (pandas/numpy/torch) at boundaries with small annotations or local casts; don't weaken project code.

### Other coding rules

- Type hints on all function signatures.
- Docstrings on all public classes and functions (one-line summary + Args/Returns for non-trivial ones).
- Put configurable hyperparameters in `conf/`, not in code.
- No magic numbers anywhere in `src/`; use named constants or config values.
- Document tensor shapes in docstrings or comments, for example `# (B, L, C)`.
- Ask before guessing unresolved design choices such as tensor layout, coordinate normalization, masking, or task head interfaces.

## Architecture philosophy

The model backbone's design — embedding/un-embedding layer contracts, the swappable encoder interface, candidate architectures to benchmark, and the self-supervised pre-training task — lives in [`.agents/architecture.md`](architecture.md).

## Tech stack

Python 3.10+, PyTorch, PyTorch Lightning, Hydra (config), Weights & Biases (logging).

## W&B conventions

- Project name: `tcfuse`.
- Entity: `arches`.
- Run naming: `{experiment_name}-{run_id}`, where `experiment_name` is the `name` field of the experiment config (`conf/experiment/*.yaml`) and `run_id` is the W&B run id (the Hydra output-dir timestamp, stable across SLURM requeues). Set in `BaseLightningModule.on_train_start` (`src/tcfuse/lightning/base_module.py`) — only the display name, never the run `id`.
- Run grouping: `{experiment_name}-{run_id}`. All segment runs spawned by a logical training run (initial launch + SLURM requeues) share this group. Set in `_build_trainer` (`scripts/train/train.py`).
- Always log: source types present, number of training samples, GPU memory peak, val metrics per task head.
- Full config: the complete resolved Hydra config is logged to the run config under `hydra_cfg` and embedded into every checkpoint (`checkpoint["hydra_cfg"]`) by `ConfigArchiveCallback` (`src/tcfuse/training/callbacks.py`). Inference rebuilds the model architecture from the embedded copy, so it never has to be re-specified.

## Jean-Zay cluster quick reference

Full workflow and hardware configs (rsync, preflight, submission, monitoring, W&B sync, checkpoints, `jz_v100` / `jz_4xv100` / `jz_h100` / `jz_4xh100` / `jz_cpu` / `jz_prepost`) → [`.agents/jz.md`](jz.md).

## CLEPS cluster quick reference

Second launch target (Inria Paris). Key differences from Jean-Zay: **pixi** (no modules), **W&B online** (no offline sync), **persistent scratch** (no `$STORE`/archive), internet on all compute nodes, and **`cpus_per_gpu`** for GPU jobs. Code lives in `$HOME/tcfuse`, data/checkpoints/wandb on `$SCRATCH/tcfuse`. `rsynctf` already syncs to both clusters. Configs: `cleps_arches` / `cleps_arches_x4` (1×/4× H200, training) / `cleps_rtx8000` / `cleps_rtx8000_x3` (1×/3× RTX8000, debug) / `cleps_cpu` (preprocessing, eval, downloads). Full workflow → [`.agents/cleps.md`](cleps.md).

## Workflow rules

1. **Plan before implementing.** For any non-trivial task, propose a plan (module structure, interface, test strategy) and wait for approval before writing code.
2. **One module at a time.** Implement and test one component fully before moving to the next.
3. **Ask, don't guess** on design decisions not covered by the rules and skills. Especially: tensor layout, coordinate normalization strategy, masking implementation, task head interface.
4. **Prefer explicit over implicit.** If a function's behavior depends on the presence or absence of a source, make that conditional explicit in the code and documented in the docstring.
5. **Tests are not optional.** Every new embedding module ships with a unit test using a synthetic `(B, N, C)` tensor. Every data loader change ships with a test that runs on a 10-sample subset.
6. **Update the rules and skills** whenever:
   - A new source type or embedding convention is decided → update the data abstraction table.
   - A new architecture is added → update [`.agents/architecture.md`](architecture.md).
   - A new SLURM script is written → update [`.agents/jz.md`](jz.md) (Jean-Zay) or [`.agents/cleps.md`](cleps.md) (CLEPS).
   - A new dataset path is confirmed → update the dataset stack table and [`.agents/preprocess.md`](preprocess.md).
   - A new W&B convention is established → update the W&B section.
   - A new skill area is added or an existing one is renamed/removed → update `.agents/<topic>.md`, then update **both** `.cursor/skills/<topic>/SKILL.md` (thin pointer) and `.claude/commands/<topic>.md` (slash-command redirect); update the skills table in `context.md`, `CLAUDE.md`, and `AGENTS.md`.
