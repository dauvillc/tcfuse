# TC-Fuse evaluation pipeline

Claude Code: invoke `/evaluate` (reads this skill).

Read this before touching `scripts/evaluation/`, `src/tcfuse/evaluation/`, `conf/evaluation.yaml`, or `conf/evaluation/`. The plugin contract, on-disk output layout, and how to add a new evaluation are defined here.

## When to use
- Compute quantitative metrics or build figures **from saved predictions** (a `PredictionRun`).
- Add a new evaluation plugin (a new metric family, a diagnostic figure set, a spectral comparison).
- Change the evaluation output layout or the metrics computed by an existing plugin.

Evaluation consumes the predictions written by inference; it never loads the model or datamodule. To (re)generate predictions, see [`.agents/inference.md`](inference.md).

## Agent behavior rules
1. **Evaluation reads predictions from disk only.** It opens a `PredictionRun` via `PredictionRun.open()`; no checkpoint, datamodule, or GPU is involved. `run_id` + `experiment_name` are plain strings that locate `paths.predictions/<run_id>/<experiment_name>/`.
2. **The base contract imposes no data shape.** `Evaluation.run(run, output_dir)` hands a plugin the whole run of structured `Source` objects. Flattening to `(N, C)` is a *plugin choice* — point-wise metrics use `flatten_valid` (`src/tcfuse/evaluation/flatten.py`); spatial metrics (e.g. power spectra) keep the 2-D `FIELD` structure and read `source.values` / `source.mask` directly.
3. **Each plugin owns its subfolder.** A plugin must write everything under the `output_dir` it is handed (named after its `name`) and never reach outside it.
4. **Quantitative metrics are torchmetrics-independent** by design (numpy / scikit-learn), so the offline suite can evolve without touching training. They are *not* expected to be bit-identical to the online `build_source_metric_collection` values, only close.

## Running evaluation

```bash
python scripts/evaluation/evaluate.py \
    run_id=0627015132 experiment_name=pmw-gmi-dummy
```

`conf/evaluation.yaml` fields (override on the CLI):

| Field | Default | Meaning |
|---|---|---|
| `run_id` | `???` | Required. Training run id; with `experiment_name` locates `paths.predictions/<run_id>/<experiment_name>/`. |
| `experiment_name` | `???` | Required. The inference experiment `name` used by `infer.py`. |
| `evaluations` | `{quantitative_metrics}` | Mapping of enabled plugins, composed from the `conf/evaluation/` config group. Add/remove via the `defaults` list or CLI. |
| `paths` | (set by `setup`) | Owned by the setup config; `setup=local` → `paths=local`, etc. |

## Output layout

```
{paths.results}/{run_id}/{experiment_name}/
├── manifest.yaml             # run_id, experiment_name, predictions_dir, evaluations, created_utc
└── {plugin.name}/            # one subfolder per enabled plugin
    └── ...                   # plugin-defined outputs
```

`quantitative_metrics/metrics.csv` columns: `source_name, channel, metric, value` (+ one column per `group_by` field). Metrics: `rmse`, `mae`, `r2`, `mape`.

## Plugin contract

A plugin subclasses `Evaluation` (`src/tcfuse/evaluation/base.py`):

```python
class MyEvaluation(Evaluation):
    name = "my_evaluation"            # filesystem-safe; also its output subfolder + config key

    def run(self, run: PredictionRun, output_dir: Path) -> None:
        for sample in run.iter_samples():   # streaming; values in physical units
            ...                             # write results under output_dir
```

## Adding a new evaluation

1. Create a module under `src/tcfuse/evaluation/<x>/` with an `Evaluation` subclass.
2. Add `conf/evaluation/<x>.yaml` with `# @package evaluations.<x>` and a `_target_` pointing at the class (plus any params).
3. Enable it by adding `evaluation/<x>` to the `defaults` list in `conf/evaluation.yaml` (or via CLI). The script instantiates every entry in `cfg.evaluations` and calls `run`.

Related: prediction format / `PredictionRun` API → [`.agents/inference.md`](inference.md); `Source` / `WindowSample` I/O → [`.agents/preprocess.md`](preprocess.md); figures style → [`.agents/visualize.md`](visualize.md).
