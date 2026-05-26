# Predictions interface — adaptation guide

Use this when a downstream task (eval script, notebook, leaderboard, viz) may not fit the current API. **Ask the user before implementing Tier C changes** (project design rule).

## 1. Classify the task

| Type | Typical needs |
|------|----------------|
| Tabular metrics only | IBTrACS RMSE/MAE, RI flags, lead-time curves |
| Per-source tensors | PMW fields, profiles at specific leads |
| Multi-run comparison | Several `run_id`s, same split |
| Streaming at scale | Many windows; memory-bound |
| Cross-run leaderboard | Aggregate metrics over manifests |

## 2. Map to existing artefacts

| Artefact | Best for |
|----------|----------|
| `run.index` | Filtering windows, source coverage, paths to HDF5 |
| `run.ibtracs` | Scalar channel metrics without loading fields |
| `run.manifest` | Model metadata, lead list, channel list |
| `SamplePrediction` HDF5 | Full `Source` pred/target for one window |
| `long_to_pivot` | Wide tables aligned with split-style column names |

## 3. Known limitations

| Limitation | Symptom for downstream app |
|------------|---------------------------|
| No multi-run API | Comparing many `run_id`s needs ad-hoc pandas loops |
| `run.ibtracs` loads entire parquet | Large runs + memory-bound analytics |
| `iter_samples` / `load_sample` always load tensors | Need stats without reading H×W fields |
| Index tracks predicted sources only | Asymmetric pred/target invisible in catalog |
| `manifest` unstructured | No schema validation for model metadata |
| IBTrACS helpers are scalar channels only | Field-level metrics need HDF5 or custom sidecar |
| `long_to_pivot` fixed pivot | Different wide layout needs wrapper or extension |

## 4. Action tiers

**Tier A — use as-is**

- Filter `run.index`, aggregate `run.ibtracs` with pandas.
- `load_sample` only for rows that need tensors.
- `read_meta` to probe before loading HDF5.

**Tier B — thin wrapper (app/notebook code)**

- `load_samples(sample_ids: list[str])` batching HDF5 reads.
- `filter_index(run, storm_id=..., has_ibtracs=True)`.
- TypedDict or dataclass for manifest fields you rely on.
- PyArrow filters on `ibtracs.parquet` outside `PredictionRun` if full load is too heavy.

**Tier C — propose API change**

Open a short design note (or ask the user) with:

- Downstream task and scale
- Which tier failed and why
- Minimal extension (new method, lazy iterator, optional sidecar parquet, manifest schema)
- Files to touch: `run.py`, `sample.py`, `ibtracs.py`, tests, this skill

Do not implement Tier C without approval.

## 5. Suggestion template

```markdown
## Prediction interface fit: [task name]

- **Need**: ...
- **Current path**: index / ibtracs / SamplePrediction / manifest / ...
- **Gap**: ...
- **Recommendation**: Tier A|B|C — [concrete change or workaround]
- **Files to touch**: ...
```

## Quick decision tree

```
Need only usa_vmax / mslp per lead?
  → run.ibtracs (+ Tier B pyarrow filter if huge)

Need one PMW field at +6h for 10 windows?
  → filter index → load_sample for those ids only

Need metrics over 20 training runs?
  → Tier B loop over run roots (Tier C if this becomes core library API)

Need field RMSE without storing in IBTrACS parquet?
  → compute from pred/target Sources in HDF5 (Tier B) or propose sidecar (Tier C)
```
