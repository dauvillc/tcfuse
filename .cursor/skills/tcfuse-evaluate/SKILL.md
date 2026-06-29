---
name: tcfuse-evaluate
description: >-
  TC-Fuse evaluation pipeline — computing metrics and building figures from
  saved predictions with `scripts/evaluation/evaluate.py` (Hydra,
  `conf/evaluation.yaml`). Plugin-based: each `Evaluation` (`src/tcfuse/evaluation/`)
  is enabled via the `conf/evaluation/` config group and writes into its own
  subfolder under `paths.results/<run_id>/<experiment_name>/`. The base contract
  imposes no data shape (flattening is plugin-dependent); the
  `quantitative_metrics` plugin computes RMSE/MAE/R2/MAPE per source/channel with
  numpy/scikit-learn, independent of torchmetrics. Use when evaluating a
  prediction run, adding a new evaluation plugin, or changing the results layout.
---

> **Content has moved.** The full skill documentation is in [`.agents/evaluate.md`](../../.agents/evaluate.md).
> Read that file for the evaluation invocation, the plugin contract (`Evaluation.run(run, output_dir)`), the on-disk results layout, and how to add a new evaluation plugin.
