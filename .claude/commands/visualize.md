# /visualize — Publication-Quality Visualization Agent

Source of truth: [`.cursor/skills/tcfuse-visualize/SKILL.md`](../../.cursor/skills/tcfuse-visualize/SKILL.md).

This command activates the TC-Fuse visualization skill. **Before writing or editing any plotting function, adding a new thematic module, or modifying `src/tcfuse/data/visualization/style.py`**, read the SKILL.md. All behavior rules, style API, and module conventions are defined there.

Figures target LaTeX articles and preprints (AMS / AGU style), saved as SVG via `save_fig`.

Keep docs in sync: when `style.py` or a thematic module changes, update SKILL.md and this file together.

---

## Quick pointer

| Need | Start here (in SKILL.md) |
|---|---|
| Style API (`setup_style`, `save_fig`, `COL1/COL2/AR_GOLDEN`) | "Style conventions" |
| Colormaps (`get_cmap("tb" \| "wind" \| …)`) | "Colormaps" |
| Categorical colors (`INTENSITY_COLORS`, `SOURCE_COLORS`) | "Categorical colors" |
| Module layout (`tracks.py`, `fields.py`, `profiles.py`, …) | "Visualization module conventions" |
| Function signature pattern (`plot_<thing>(data, ax=None, *, ...) -> (fig, ax)`) | "Function signature pattern" |
| TC track map example | "Priority plot types" → 1. TC track map |
| 2D satellite field example | "Priority plot types" → 2. 2D satellite field |
| Adding a new visualization module | "Adding a new visualization module" |
