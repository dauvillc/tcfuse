# TC-Fuse inference and prediction pipeline

Claude Code: invoke `/inference` (reads this skill).

Read this before touching `scripts/inference/`, `src/tcfuse/data/predictions/`, `src/tcfuse/lightning/prediction_writer.py`, or `conf/inference.yaml`. The end-to-end flow, on-disk output format, and evaluation API are defined here.

## When to use
- Run a trained checkpoint over a split and save predictions.
- Evaluate a checkpoint — compute per-source / per-channel metrics (bias, RMSE, MAE, R2).
- Change the prediction output format (`SamplePrediction` / `PredictionRun` HDF5 / index layout).
- Add or change an evaluation metric.
- Debug the `PredictionWriter` (how `trainer.predict` output is paired with ground truth).

## Agent behavior rules
1. **Single-GPU only.** `infer.py` pins `devices=1`. The `PredictionWriter` and `PredictionRun` index are **not DDP-safe** (per-process index buffer, unconditional `finalize`). Do not introduce multi-device prediction without first solving rank-0 index gather + dedup of the padded tail.
2. **`shuffle=False` is mandatory** in the predict dataloader — the writer recovers each window's ground truth via `batch_indices`, and reproducible order keeps the index valid.
3. **Predictions are in physical units.** `predict_step` de-normalizes before the writer sees them; everything stored on disk and all metrics are in physical units (`manifest["units"] == "physical"`).
4. **Reuse `build_source_metric_collection`** (`src/tcfuse/metrics/collection.py`) for any evaluation — never redefine metrics inline. This keeps offline metrics identical to online validation.
5. **`predicted` and `target` always share the same key set** `(source_name, source_index)` in a `SamplePrediction`. Only target slots are stored; non-target snapshots are discarded.
6. **Masking is required at predict time** for reconstruction tasks — see § Masking caveat. Without it the model trivially echoes the ground-truth targets.

## Running inference

```bash
python scripts/inference/infer.py \
    experiment=pmw_gmi_reconstruction_dummy \
    run_id=0627015132 \
    split=test
```

`run_id` is the training run identifier (the directory name under
`paths.checkpoints`); inference resolves that run's `best-*.ckpt` under
`paths.checkpoints/<run_id>/checkpoints/` via `best_checkpoint`
(`src/tcfuse/utils/checkpoint.py`). The `experiment=` chosen here defines the
**inference** experiment (sources, datamodule, lightning_module, windows_setup)
and need not match the one used for training.

`conf/inference.yaml` fields (override on the CLI):

| Field | Default | Meaning |
|---|---|---|
| `experiment` | `???` | Required. Defines `name` + sources + mounts `lightning_module` and `windows_setup`. |
| `run_id` | `???` | Required. Training run id (dir name under `paths.checkpoints`); resolves to `{paths.checkpoints}/{run_id}/checkpoints/best-*.ckpt` via `best_checkpoint`. Weights loaded into the rebuilt module (`checkpoint["state_dict"]`). |
| `split` | `test` | Split to run over: `train` / `val` / `test`. |
| `limit_samples` | `null` | Cap on windows processed (smoke tests). Translated to `limit_predict_batches`. |
| `compute_metrics` | `true` | Write `metrics.csv` next to the run after prediction. |
| `paths` | (set by `setup`) | Owned by the setup config; `setup=local` → `paths=local`, `setup=jz_*` → `paths=jz`, etc. Override explicitly with `paths=jz` if needed. |

`optimizer`, `lr_scheduler`, and `setup` are carried only to satisfy the module constructor / experiment resolution — unused at inference.

## End-to-end flow

`scripts/inference/infer.py:main()`:

1. Instantiate the `datamodule`; `dm.setup("predict")` loads `sources_metadata` + `normalization_stats` from the metadata YAML.
2. Rebuild the task-specific Lightning module via a `_partial_` factory, passing `sources_metadata` / `normalization_stats` / `experiment_name` directly (these are **deliberately omitted** from checkpoint hparams), then `module.load_state_dict(checkpoint["state_dict"])`.
3. Build `TCWindowDataset(preprocessed_data, windows_setup.name, split)` and a `DataLoader` with `collate_window_samples`, `shuffle=False`.
4. `PredictionRun.create(run_dir, manifest=...)` opens the output dir; `PredictionWriter(run, dataset)` is registered as a Trainer callback (`write_interval="batch"`).
5. `pl.Trainer(accelerator=..., devices=1, logger=False, callbacks=[writer], limit_predict_batches=...)`; `trainer.predict(module, dataloaders=loader, return_predictions=False)` owns the loop, device placement, `eval()`/`no_grad`.
6. Per batch, `module.predict_step` normalizes → masks targets (task-specific) → forward → de-normalizes; `PredictionWriter.write_on_batch_end` recovers each window's ground-truth `WindowSample` from the dataset (via `batch_indices`), builds one `SamplePrediction` per window (target slots only), and `run.append()`s it.
7. `run.finalize()` flushes `index.parquet` + `manifest.yaml` (adds `num_samples`).
8. If `compute_metrics`: `run.compute_metrics()` → `metrics.csv`.

## Output format

```
{paths.predictions}/{run_id}/{experiment_name}/
├── manifest.yaml          # run-level metadata
├── index.parquet          # one row per (sample_id, source_name, source_index)
├── samples/{sample_id}.h5 # one SamplePrediction per window
└── metrics.csv            # optional; per (source, channel, metric) row
```

Re-running a different `split` for the same `run_id` + inference `experiment` overwrites this directory (split is recorded only in the manifest); use a distinct inference `experiment=` to keep them separate.

**`manifest.yaml` keys:** `run_id`, `checkpoint_path`, `experiment_name`, `windows_setup_name`, `split`, `units` (`"physical"`), `created_utc` (ISO 8601), `num_samples` (added by `finalize`).

**`index.parquet` columns** (long-format, one row per predicted source slot): `sample_id`, `sid`, `season`, `basin`, `subbasin`, `window_ref_time_utc`, `source_name`, `source_index`, `kind`, `time_utc`, `n_channels`.

**`samples/{sample_id}.h5` layout** (`SamplePrediction.write`, `src/tcfuse/data/predictions/sample.py`):
```
/
├── attrs: {sample_id, sid, season, basin, subbasin, window_ref_time_utc}
├── predicted/{source_name}/{source_index}/   # Source group (model output)
└── target/{source_name}/{source_index}/      # Source group (ground truth)
```
Predicted values are the model output; coords / mask / channels / time are copied from the ground truth.

## Reading a run

```python
from tcfuse.data.predictions.run import PredictionRun

run = PredictionRun.open(run_dir)          # loads manifest + index
run.sample_ids                             # distinct window ids, in index order
sample = run.load_sample(sample_id)        # one SamplePrediction
for sample in run.iter_samples(): ...      # lazy streaming over all windows

# Per-source / per-channel metrics (bias, RMSE, MAE, R2), physical units.
df = run.compute_metrics()                         # global
df = run.compute_metrics(group_by=["basin"])       # grouped
# group_by ∈ {"sid", "season", "basin", "subbasin"}; returns tidy DataFrame
# with columns [source_name, channel, metric, value] (+ one col per group field).
```

A spatial position is counted only where **every** target channel is available (`target.mask.all(axis=-1)`), mirroring `BaseLightningModule._update_val_metrics`. Each metric is computed independently, so one failing metric (e.g. R2 with a single sample) does not suppress the rest.

## Masking caveat

`BaseLightningModule.predict_step` is the generic path: `denormalize(self(normalize(batch)))`. `MaskedReconstructionLightningModule.predict_step` **overrides** it to re-apply `_mask_targets` (values → NaN, mask → all-False at target batch indexes) on the normalized batch *before* the forward pass — exactly as in training. Skipping this would let the backbone see the ground-truth target values and reconstruct them trivially, invalidating the evaluation. Any new task module that uses target masking must mirror this in its `predict_step`.

Related: model internals → [`.agents/architecture.md`](architecture.md); dataset / `WindowSample` / `Source` I/O → [`.agents/preprocess.md`](preprocess.md); cluster submission → [`.agents/jz.md`](jz.md) / [`.agents/cleps.md`](cleps.md).
