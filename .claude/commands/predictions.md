# /predictions — Forecast Output Storage Agent

Canonical tests: `tests/data/predictions/`.

This command activates the TC-Fuse prediction I/O skill. **Before reading or editing `src/tcfuse/data/predictions/` or writing inference/evaluation scripts**, read these files:

1. `.agents/predictions/skill.md`
2. `.agents/predictions/reference.md` — schemas, paths, HDF5 layout
3. `.agents/predictions/adaptation.md` — API change proposals

Keep docs in sync: when changing `src/tcfuse/data/predictions/`, update the skill files in the same PR; update this file and `CLAUDE.md` if triggers or behavior rules change.

---

## Quick pointer

| Need | Start here |
|------|------------|
| Write a forecast run | `PredictionRun.create` + `add_sample` + `close` — see SKILL.md |
| Read catalog / IBTrACS table | `PredictionRun.from_disk` → `.index`, `.ibtracs` |
| Load one window's fields | `load_sample` or `SamplePrediction.from_disk` |
| Propose API extension | `adaptation.md` template |

Public imports: `from tcfuse.data.predictions import PredictionRun, SamplePrediction, build_long_rows, ...`
