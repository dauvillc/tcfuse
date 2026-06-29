# /evaluate — TC-Fuse Evaluation Agent

Source of truth: [`.agents/evaluate.md`](../../.agents/evaluate.md).

This command activates the TC-Fuse evaluation skill. **Before running `scripts/evaluation/evaluate.py`, changing `src/tcfuse/evaluation/`, or editing `conf/evaluation.yaml` / `conf/evaluation/`**, read the skill file. The plugin contract (`Evaluation.run(run, output_dir)`), the on-disk results layout, and how to add a new evaluation plugin are defined there.

Predictions come from [`/inference`](inference.md). Cluster submission: [`/jz`](jz.md), [`/cleps`](cleps.md). Figures style: [`/visualize`](visualize.md).

Keep docs in sync: when the plugin contract, the results output layout, or the set of plugins changes, update `.agents/evaluate.md` and this file together; update the skills table in `.agents/context.md` and `CLAUDE.md` if the layout changes.

---

## Quick pointer

| Need | Start here (in evaluate.md) |
|---|---|
| Run evaluation over a prediction run (`evaluate.py` invocation, config fields) | "Running evaluation" |
| What a plugin receives / must do | "Plugin contract" |
| Why flattening is plugin-dependent / torchmetrics-independent | "Agent behavior rules" |
| `{paths.results}/` layout, `metrics.csv` columns | "Output layout" |
| Add a new metric family or figure set | "Adding a new evaluation" |
