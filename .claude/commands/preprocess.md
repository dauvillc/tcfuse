# /preprocess — TC-Fuse Preprocessing Agent

Source of truth: [`.cursor/skills/tcfuse-preprocess/SKILL.md`](../../.cursor/skills/tcfuse-preprocess/SKILL.md).

This command activates the TC-Fuse preprocessing skill. **Before running any
preprocessor, modifying `src/tcfuse/data/sources/`, or working with Stage 1 /
Stage 2 HDF5 layouts**, read the SKILL.md.

Jean-Zay submission lives in [`/jz`](jz.md) (→ [`.cursor/skills/tcfuse-jz/`](../../.cursor/skills/tcfuse-jz/SKILL.md)).
Forecast output storage (predictions, not preprocessing) lives in [`/predictions`](predictions.md).

---

## Agent behavior rules

1. **Read the skill file first.** Do not guess Stage 1 / Stage 2 layouts, IBTrACS injection rules, or the assembled `index.parquet` schema.
2. **Pipeline order is fixed:** per-source preprocessors → `assemble.py` → `build_splits.py` → `compute_normalization.py`. Build splits before normalization to avoid validation/test leakage.
3. **Use `cfg.paths.*` for all paths.** Never hardcode filesystem paths; `paths.raw_datasets.<name>` for raw, `paths.preprocessed_sources` for Stage 1, `paths.preprocessed_data` for Stage 2.
4. **Preserve missing-data semantics.** Sources may have NaN values; rely on `Source.mask` and never silently fill USA/WMO best-track quantities across providers.
5. **Keep docs in sync:** when a Stage 1/Stage 2 schema, the assembled index, or a script under `scripts/preprocess/` changes, update `.cursor/skills/tcfuse-preprocess/SKILL.md` and this command file together; update the dataset table in `.cursor/rules/tcfuse-core.mdc` when a new dataset path is confirmed.

---

## Quick pointer

| Need | Start here (in SKILL.md) |
|---|---|
| Dataset inventory + status | "Dataset inventory" |
| Stage 1 per-source HDF5 layout | "Stage 1 — Per-source format" |
| Stage 2 assembled per-storm layout | "Stage 2 — Assembled format" |
| Run a per-source preprocessor (local or JZ) | "Running the preprocessor" steps 2–3 |
| Run assembly | "Running the preprocessor" step 5 |
| Build train/val/test window splits | "Running the preprocessor" step 6 |
| Compute normalization stats | "Running the preprocessor" step 7 |
| Add a new dataset | "Adding a new dataset preprocessor" |
| I/O API (`Source`, `StormData`, `SourceMetadata`) | "I/O API reference" |
