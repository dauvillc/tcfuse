---
name: tcfuse-inference
description: >-
  TC-Fuse inference and prediction pipeline — running a trained checkpoint
  over a split with `scripts/inference/infer.py` (Hydra, `conf/inference.yaml`),
  the single-GPU `trainer.predict` flow with task-specific target masking, the
  `PredictionWriter` callback, and the `SamplePrediction` / `PredictionRun`
  output format (per-window HDF5 + `index.parquet` + `manifest.yaml`) with its
  `compute_metrics` evaluation (bias, RMSE, MAE, R2 per channel). Use when
  running inference, evaluating a checkpoint, changing the prediction output
  format, or working in `src/tcfuse/data/predictions/` or
  `src/tcfuse/lightning/prediction_writer.py`.
---

> **Content has moved.** The full skill documentation is in [`.agents/inference.md`](../../.agents/inference.md).
> Read that file for the inference invocation, the end-to-end flow, the on-disk `PredictionRun` format, the read/evaluation API, and the predict-time masking caveat.
