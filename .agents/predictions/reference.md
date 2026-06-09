# Predictions interface — reference

Source of truth: `src/tcfuse/data/predictions/`. Examples: `tests/data/predictions/`.

## Paths

| Concept | Resolution |
|---------|------------|
| Runs root | `cfg.paths.predictions` (`conf/paths/local.yaml`, `jz.yaml`) |
| Run directory | `{cfg.paths.predictions}/{run_id}/` |
| Sample HDF5 | `SamplePrediction.path(run_root, sample_id)` → `{run_root}/samples/{sample_id}.h5` |

## On-disk run layout

```
{run_root}/
├── manifest.yaml       # run metadata (model, split, leads, …)
├── index.parquet       # one row per forecast window (catalog)
├── ibtracs.parquet     # tidy-long IBTrACS preds + targets (all samples)
└── samples/
    └── {sample_id}.h5  # per-window pred + target Sources
```

### `manifest.yaml`

Free-form dict written by the caller. Writer injects/updates at `close()`:

- `created_at_utc` (set if absent at `create`)
- `n_samples`
- `predicted_sources` (sorted union of source names across samples)

Typical caller fields: `run_id`, `model`, `split`, `leads_hours`, `ibtracs_channels`, `deterministic`.

### `index.parquet` columns

| Column | Type / notes |
|--------|----------------|
| `sample_id` | Window id |
| `storm_id` | e.g. `2016AL10` |
| `season` | int |
| `basin` | e.g. `AL` |
| `atcf_id` | optional ATCF id |
| `init_time_utc` | Window anchor (repository ISO) |
| `window_start_time_utc` | Defaults to `init_time_utc` if not overridden in `add_sample` |
| `window_end_time_utc` | Optional; may be null |
| `n_predicted_sources` | Count of unique predicted source names |
| `predicted_source_names` | list of str |
| `has_ibtracs_prediction` | bool — non-empty IBTrACS block appended for this sample |
| `sample_path` | Relative path, e.g. `samples/2016AL10_20160912T000000Z.h5` |

### `ibtracs.parquet` — tidy-long schema

One row per `(sample_id, valid_time_utc, channel)` (plus denormalized keys).

| Column | Dtype | Meaning |
|--------|-------|---------|
| `sample_id` | string | Window id |
| `storm_id` | string | |
| `season` | int32 | For cheap group-bys |
| `basin` | string (dictionary in parquet) | |
| `init_time_utc` | string | Window anchor |
| `valid_time_utc` | string | Valid time for this lead |
| `lead_hour` | int32 | Hours from init |
| `channel` | string (dictionary) | e.g. `usa_vmax_kt` |
| `pred` | float64 | Predicted value; NaN if missing |
| `target` | float64 | Target value; NaN if missing |
| `mask` | bool | `True` iff both `pred` and `target` are finite |

Canonical column order: `IBTRACS_LONG_COLUMNS` in `ibtracs.py`. PyArrow schema: `ibtracs_long_schema()`.

Empty runs still get an empty `ibtracs.parquet` (so readers can rely on the file existing).

## Per-sample HDF5 (`SamplePrediction`)

### Root attributes

`sample_id`, `storm_id`, `init_time_utc`, `basin`, `season`, optional `atcf_id`, optional `run_id`.

### Groups

```
/
├── pred/
│   └── {source_name}/{compact_valid_time}/   # Source HDF5 group + attrs
└── target/
    └── {source_name}/{compact_valid_time}/
```

`compact_valid_time` from `to_compact_time(time_utc)` (same as preprocessed sources).

Per-snapshot group attrs (in addition to `Source` datasets):

- `kind` — `SourceKind` name (`SCALAR`, `PROFILE`, `FIELD`)
- `time_utc` — ISO string (round-trip key)
- `lead_hour` — int, when derivable from init vs snapshot
- Forwarded keys from `Source.meta` (except `source_name`, `channels`, `kind`, `time_utc`, `lead_hour`)

Tensor layout inside each group: delegated to `Source.to_hdf5_group` / `from_hdf5_group` (same as preprocessing).

### In-memory `SamplePrediction`

| Field | Type |
|-------|------|
| `pred_sources` | `dict[tuple[str, str], Source]` |
| `target_sources` | `dict[tuple[str, str], Source]` |
| `predicted_source_names` | property — sorted unique names in `pred_sources` |

## `build_long_rows` contract

```python
build_long_rows(
    sample_id: str,
    storm_id: str,
    season: int,
    basin: str,
    init_time_utc: str,
    leads: Sequence[Mapping[str, Any]],
    channels: Sequence[str],
) -> pd.DataFrame
```

Each element of `leads`:

| Key | Type | Required |
|-----|------|----------|
| `lead_hour` | int | yes |
| `valid_time_utc` | str (ISO) | yes |
| `pred` | mapping channel → float, or omit | no — missing → NaN |
| `target` | mapping channel → float, or `None` | no — `None` → all NaN, `mask=False` |

- `channels` order is preserved in output rows.
- Channels absent from a lead's `pred`/`target` map → NaN, `mask=False`.
- Returns `empty_long_frame()` if `leads` is empty.

## `long_to_pivot`

Pivots tidy-long → wide frame indexed by `sample_id`, `storm_id`, `season`, `basin`, `init_time_utc`.

Default wide columns: `lead_{lead_hour:03d}h_{channel}_{pred|target}` (e.g. `lead_006h_usa_vmax_kt_pred`).

Returns empty `DataFrame` if input is empty.

## API cheat sheet

| Class / function | Role |
|------------------|------|
| `PredictionRun.create(run_root, manifest)` | New writer; creates `samples/`, writes manifest |
| `PredictionRun.from_disk(run_root)` | Reader; manifest eager, parquet lazy |
| `run.add_sample(sample, ibtracs_long_rows=None, *, window_start_time_utc=..., window_end_time_utc=...)` | Write HDF5 + index row + append IBTrACS |
| `run.close()` | Finalise index + ibtracs + manifest; idempotent |
| `run.index` | Catalog DataFrame |
| `run.ibtracs` | Full tidy-long table |
| `run.load_sample(sample_id)` | One `SamplePrediction` |
| `run.iter_samples()` | All samples in index order |
| `SamplePrediction.write(run_root)` | Write one HDF5 |
| `SamplePrediction.from_disk(run_root, sample_id)` | Load one HDF5 |
| `SamplePrediction.read_meta(run_root, sample_id)` | Root attrs only |

## Reader / writer pitfalls

- `add_sample` validates IBTrACS columns before append; missing columns → `ValueError` (reindex would hide bugs).
- `iter_samples` and `load_sample` always load full tensor payloads.
- `run.ibtracs` loads the entire parquet on first property access.
- Index does **not** list target-only sources or per-target coverage — only predicted source names.
- `PredictionRun.create` does not delete existing files; use unique `run_id` values.

## StormData comparison

| | `StormData` | `SamplePrediction` |
|---|-------------|-------------------|
| Granularity | One storm | One forecast window `(storm_id, init_time)` |
| Path | `{assembled}/storm_data/{storm_id}.h5` | `{run_root}/samples/{sample_id}.h5` |
| Source map | `sources` | `pred_sources` + `target_sources` |
| HDF5 layout | `{source_name}/{compact_time}/` at root | Same under `pred/` and `target/` |
| Root attrs | `storm_id`, `basin`, `season`, `atcf_id?` | + `sample_id`, `init_time_utc`, `run_id?` |

Preprocessed inputs: `StormData.path(cfg.paths.preprocessed_data, storm_id)`.
