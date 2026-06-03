# /preprocess — TC-Fuse Preprocessing Agent

Source of truth: [`.cursor/skills/tcfuse-preprocess/SKILL.md`](../../.cursor/skills/tcfuse-preprocess/SKILL.md).

This command activates the TC-Fuse preprocessing skill. **Before running any preprocessor, modifying `src/tcfuse/data/sources/`, or working with Stage 1 / Stage 2 HDF5 layouts**, read the SKILL.md. All behavior rules, pipeline invariants, schema details, and coding-style requirements are defined there.

Jean-Zay submission: [`/jz`](jz.md). Forecast output storage: [`/predictions`](predictions.md).

Keep docs in sync: when a Stage 0/1/2 schema or preprocess script changes, update SKILL.md and this file together; update the dataset table in `tcfuse-core.mdc` when a new dataset path is confirmed.

---

## Quick pointer

| Need | Start here (in SKILL.md) |
|---|---|
| Run via Pixi (`pixi run preprocess`, stage composites) | "Running the preprocessor" → "Pixi tasks" |
| Pipeline overview | "Pipeline at a glance" |
| Dataset inventory + status | "Dataset inventory" |
| Stage 0 IBTrACS prep + ATCF→SID translation | "Stage 0 — IBTrACS preprocessing" |
| Stage 1 per-source HDF5 layout | "Stage 1 — Per-source format" |
| Stage 2 assembled per-storm layout | "Stage 2 — Assembled format" |
| Run Stage 0 IBTrACS prep | "Running the preprocessor" step 1 |
| Run a per-source preprocessor (local or JZ) | "Running the preprocessor" steps 2–3 |
| Run assembly | "Running the preprocessor" step 5 |
| Build train/val/test window splits | "Running the preprocessor" step 6 |
| Compute normalization stats | "Running the preprocessor" step 7 |
| Add a new dataset | "Adding a new dataset preprocessor" |
| I/O API (`Source`, `StormData`, `SourceMetadata`) | "I/O API reference" |
