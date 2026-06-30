---
name: tcfuse-sweep
description: >-
  TC-Fuse hyperparameter search — Hydra Optuna sweeper + submitit launcher running
  parallel single-GPU SLURM trials (offline-safe, works on Jean-Zay). Search space in
  conf/hydra/sweeper/*.yaml; launcher SLURM spec in conf/hydra/launcher/jz_*_sweep.yaml;
  short-budget experiment in conf/experiment/pmw_gmi_sweep.yaml. With submitit=false,
  scripts/train/train.py returns the best val/loss as the objective Optuna minimizes.
  Respect the Perceiver's head-divisibility invariants when defining the space. Use when
  running a sweep, adding a search space, or changing the optimized metric.
---

> **Content has moved.** The full skill documentation is in [`.agents/sweep.md`](../../.agents/sweep.md).
> Read that file for the sweep invocation, the search-space / divisibility rules, the objective contract in `train.py`, and how to add a new sweep.
