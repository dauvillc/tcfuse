# /visualize — Publication-Quality Visualization Agent

Source of truth: [`.cursor/skills/tcfuse-visualize/SKILL.md`](../../.cursor/skills/tcfuse-visualize/SKILL.md).

This command activates the TC-Fuse visualization skill. **Before writing or
editing any plotting function, adding a new thematic module, or modifying
`src/tcfuse/data/visualization/style.py`**, read the SKILL.md.

Figures target LaTeX articles and preprints (AMS / AGU style), saved as SVG via
`save_fig`.

---

## Agent behavior rules

1. **Ask before assuming.** When in doubt about color scale, map projection, domain extent, or which channels to display — stop and ask the user.
2. **Consistency first.** Check `style.py` for existing constants (colormaps, colors, figure widths) before defining new ones; extend `style.py` only for genuinely reusable values.
3. **Always use `setup_style()` and `save_fig()`.** Call `setup_style()` at module import; save with `save_fig(fig, path)` for SVG output and consistent margins.
4. **No hardcoded paths.** Output paths come from function arguments or `cfg.paths.*`.
5. **Install missing packages with Pixi.** `pixi add <package>` (conda-forge). Never use `pip` to add visualization deps.
6. **Keep docs in sync:** when `style.py` or a thematic module changes, update `.cursor/skills/tcfuse-visualize/SKILL.md` and this command file together.

---

## Quick pointer

| Need | Start here (in SKILL.md) |
|---|---|
| Style API (`setup_style`, `save_fig`, `COL1/COL2/AR_GOLDEN`) | "Style conventions" |
| Colormaps (`get_cmap("tb" | "wind" | …)`) | "Colormaps" |
| Categorical colors (`INTENSITY_COLORS`, `SOURCE_COLORS`) | "Categorical colors" |
| Module layout (`tracks.py`, `fields.py`, `profiles.py`, …) | "Visualization module conventions" |
| Function signature pattern (`plot_<thing>(data, ax=None, *, ...) -> (fig, ax)`) | "Function signature pattern" |
| TC track map example | "Priority plot types" → 1. TC track map |
| 2D satellite field example | "Priority plot types" → 2. 2D satellite field |
| Adding a new visualization module | "Adding a new visualization module" |
