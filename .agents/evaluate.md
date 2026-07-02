# TC-Fuse evaluation pipeline

Claude Code: invoke `/evaluate` (reads this skill).

Read this before touching `scripts/evaluation/`, `src/tcfuse/evaluation/`, `conf/evaluation.yaml`, or `conf/evaluation/`. The plugin contract, on-disk output layout, and how to add a new evaluation are defined here.

## When to use
- Compare several models' quantitative metrics or build comparison figures **from saved predictions** (one `PredictionRun` per model).
- Add a new evaluation plugin (a new metric family, a diagnostic figure set, a spectral comparison).
- Change the evaluation output layout or the metrics computed by an existing plugin.

Evaluation consumes the predictions written by inference; it never loads the model or datamodule. To (re)generate predictions, see [`.agents/inference.md`](inference.md).

## Agent behavior rules
1. **Evaluation reads predictions from disk only and compares models.** Each model is opened as a `PredictionRun` via `PredictionRun.open()`; no checkpoint, datamodule, or GPU is involved. A model's `run_id` + `experiment_name` are plain strings that locate `paths.predictions/<run_id>/<experiment_name>/`. The set of models comes from the `models` config mapping `{model_name: {run_id, experiment_name}}`.
2. **The base contract imposes no data shape.** `Evaluation.run(runs, output_dir)` hands a plugin a `dict[str, PredictionRun]` (model name → run, in config order) of structured `Source` objects. Flattening to `(N, C)` is a *plugin choice* — point-wise metrics use `flatten_valid` (`src/tcfuse/evaluation/flatten.py`); spatial metrics (e.g. power spectra) keep the 2-D `FIELD` structure and read `source.values` / `source.mask` directly.
3. **Each plugin owns its subfolder.** A plugin must write everything under the `output_dir` it is handed (named after its `name`) and never reach outside it.
4. **Quantitative metrics are torchmetrics-independent** by design (numpy / scikit-learn), so the offline suite can evolve without touching training. They are *not* expected to be bit-identical to the online `build_source_metric_collection` values, only close.

## Running evaluation

The model set is supplied by a **required** `comparison` config (`conf/comparison/`),
which fills the top-level `eval_name` and `models` keys. Select one on the CLI:

```bash
python scripts/evaluation/evaluate.py comparison=debug
```

Add a new comparison by dropping a file in `conf/comparison/` (packaged
`# @package _global_`); see `conf/comparison/debug.yaml` for the shape:

```yaml
# @package _global_
eval_name: baseline-vs-fusion
models:
  baseline: { run_id: "0627015132", experiment_name: pmw-gmi-dummy }
  fusion:   { run_id: "0628231045", experiment_name: pmw-gmi-fusion }
```

Quote `run_id` so leading zeros survive YAML parsing.

`conf/evaluation.yaml` fields (override on the CLI):

| Field | Default | Meaning |
|---|---|---|
| `comparison` | `???` | Required. Selects the `conf/comparison/` config that provides `eval_name` + `models`. |
| `eval_name` | (from `comparison`) | Names the comparison; results go to `paths.results/<eval_name>/`. |
| `models` | (from `comparison`) | Mapping `{model_name: {run_id, experiment_name}}`; each pair locates `paths.predictions/<run_id>/<experiment_name>/`. Declaration order is preserved as column / plot order. |
| `evaluations` | `{quantitative_metrics}` | Mapping of enabled plugins, composed from the `conf/evaluation/` config group. Add/remove via the `defaults` list or CLI. |
| `paths` | (set by `setup`) | Owned by the setup config; `setup=local` → `paths=local`, etc. |

## Output layout

```
{paths.results}/{eval_name}/
├── manifest.yaml             # eval_name, models (per-model run_id/experiment_name/predictions_dir), evaluations, created_utc
└── {plugin.name}/            # one subfolder per enabled plugin
    └── ...                   # plugin-defined outputs
```

`quantitative_metrics/` contents:
- `metrics.csv` columns: `model, source_name, channel, metric, value` (+ one column per `group_by` field). Metrics: `rmse`, `mae`, `r2`, `mape`.
- `<metric>_<source_name>.svg` — one grouped bar chart per (metric, source): x = channel, one coloured bar per model. With `group_by` set, figures show the global result only; the per-group breakdown stays in the CSV.

`visual/` contents (not enabled by default — add `evaluation/visual` to the `defaults` list or CLI):
- `<sample_id>_<source_name>_<source_index>.svg` — one figure per window shared by every model (a sample id present in all runs) and FIELD-kind target source, up to `max_samples` windows. One row per channel: Target | Pred | Diff panels repeated per model, grouped by model. Target/prediction panels share a per-channel color scale; diff panels (prediction minus target) share one symmetric diverging scale across all models. SCALAR / PROFILE targets are skipped.

## Plugin contract

A plugin subclasses `Evaluation` (`src/tcfuse/evaluation/base.py`) and compares models:

```python
class MyEvaluation(Evaluation):
    name = "my_evaluation"            # filesystem-safe; also its output subfolder + config key

    def run(self, runs: dict[str, PredictionRun], output_dir: Path) -> None:
        for model_name, run in runs.items():        # config order preserved
            for sample in run.iter_samples():        # streaming; values in physical units
                ...                                  # write results under output_dir
```

## Adding a new evaluation

1. Create a module under `src/tcfuse/evaluation/<x>/` with an `Evaluation` subclass.
2. Add `conf/evaluation/<x>.yaml` with `# @package evaluations.<x>` and a `_target_` pointing at the class (plus any params).
3. Enable it by adding `evaluation/<x>` to the `defaults` list in `conf/evaluation.yaml` (or via CLI). The script instantiates every entry in `cfg.evaluations` and calls `run`.

Related: prediction format / `PredictionRun` API → [`.agents/inference.md`](inference.md); `Source` / `WindowSample` I/O → [`.agents/preprocess.md`](preprocess.md); figures style → [`.agents/visualize.md`](visualize.md).
