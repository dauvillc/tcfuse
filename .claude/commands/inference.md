# /inference — TC-Fuse Inference and Prediction Agent

Source of truth: [`.agents/inference.md`](../../.agents/inference.md).

This command activates the TC-Fuse inference skill. **Before running `scripts/inference/infer.py`, changing `src/tcfuse/data/predictions/`, `src/tcfuse/lightning/prediction_writer.py`, or `conf/inference.yaml`, or evaluating a checkpoint**, read the skill file. The end-to-end flow, single-GPU constraint, on-disk `PredictionRun` format, and the `compute_metrics` API are defined there.

Model internals: [`/architecture`](architecture.md). Cluster submission: [`/jz`](jz.md), [`/cleps`](cleps.md).

Keep docs in sync: when the inference flow, the `PredictionRun` / `SamplePrediction` output format, or the metrics set changes, update `.agents/inference.md` and this file together; update the prediction tree lines + skills table in `.agents/context.md` if the layout changes.

---

## Quick pointer

| Need | Start here (in inference.md) |
|---|---|
| Run a checkpoint over a split (`infer.py` invocation, config fields) | "Running inference" |
| Config → predict_step → writer → run → metrics chain | "End-to-end flow" |
| Why single-GPU only / `shuffle=False` / physical units | "Agent behavior rules" |
| `{run_dir}/` layout, `index.parquet` columns, manifest keys, HDF5 groups | "Output format" |
| Read predictions back, group metrics by basin/season/… | "Reading a run" |
| Why `predict_step` masks targets at inference | "Masking caveat" |
