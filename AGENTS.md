# AGENTS.md — TC Multi-Source Fusion Project

This project develops a machine learning framework for tropical cyclone (TC) analysis and forecasting by fusing heterogeneous observation sources. It is designed from first principles with efficiency, modularity, and multi-architecture support in mind.

## Read first

Always start by reading the core project context:

- [`.agents/context.md`](.agents/context.md) — project overview, data abstraction, repo structure, coding rules, architecture, W&B conventions, workflow rules.

## On-demand skills

When a task touches one of these areas, read the matching skill file before making changes.

| Topic | Skill file |
|---|---|
| Dataset preprocessing (per-source HDF5, assembled storms, splits, normalization) | [`.agents/preprocess.md`](.agents/preprocess.md) |
| Jean-Zay cluster operations (rsync, SLURM, monitoring, W&B sync, checkpoints) | [`.agents/jz.md`](.agents/jz.md) |
| CLEPS cluster operations (pixi, W&B online, persistent scratch, SLURM, monitoring) | [`.agents/cleps.md`](.agents/cleps.md) |
| Publication-quality figures (style.py, SVG output, thematic plotting modules) | [`.agents/visualize.md`](.agents/visualize.md) |
| Model backbone architecture (embedding/encoder/decoder design, candidate backbones, pre-training task) | [`.agents/architecture.md`](.agents/architecture.md) |
| Inference and prediction pipeline (checkpoint loading, task masking, PredictionRun output, metrics) | [`.agents/inference.md`](.agents/inference.md) |
| Hyperparameter search (Hydra Optuna sweeper, parallel SLURM trials, search-space/divisibility rules, objective contract) | [`.agents/sweep.md`](.agents/sweep.md) |
| Basedpyright diagnostics workflow | [`.agents/pyright-fixer.md`](.agents/pyright-fixer.md) |

## Update protocol

See [`.agents/context.md`](.agents/context.md) § Workflow rules, item 6 for the full update protocol. Never duplicate content across this file, `CLAUDE.md`, and `.agents/` files.
