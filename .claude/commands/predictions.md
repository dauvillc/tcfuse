# /predictions — Forecast Output Storage Agent

Canonical tests: `tests/data/predictions/`.

This command activates the TC-Fuse prediction I/O skill. **Before reading or editing**
`src/tcfuse/data/predictions/` or writing inference/evaluation scripts that emit or consume
forecast runs, read these files in order:

1. `.cursor/skills/tcfuse-predictions/SKILL.md`
2. `.cursor/skills/tcfuse-predictions/reference.md` — if schemas, paths, or HDF5 layout matter
3. `.cursor/skills/tcfuse-predictions/adaptation.md` — if proposing API changes for a downstream app

Do **not** read the Python modules under `src/tcfuse/data/predictions/` unless the skill
directs you to a specific symbol or the skill is insufficient for the task.

---

## Agent behavior rules

1. **Read the skill files first.** Do not guess tensor layout, on-disk layout, or IBTrACS schema.
2. **Use `cfg.paths.predictions` for all paths.** Never hardcode run directories.
3. **Do not confuse modules:** `tcfuse.data.ibtracs` (CSV → `Source` for preprocessing) vs `tcfuse.data.predictions.ibtracs` (tidy-long pred/target table).
4. **Interface fit:** When a downstream task does not match the storage layout, follow `adaptation.md` (Tier A/B first). Ask before implementing Tier C API changes.
5. **Keep docs in sync:** When changing `src/tcfuse/data/predictions/`, update `.cursor/skills/tcfuse-predictions/` in the same PR; update this command and `CLAUDE.md` if triggers or behavior rules change.

---

## Quick pointer

| Need | Start here |
|------|------------|
| Write a forecast run | `PredictionRun.create` + `add_sample` + `close` — see SKILL.md |
| Read catalog / IBTrACS table | `PredictionRun.from_disk` → `.index`, `.ibtracs` |
| Load one window's fields | `load_sample` or `SamplePrediction.from_disk` |
| Propose API extension | `adaptation.md` template |

Public imports: `from tcfuse.data.predictions import PredictionRun, SamplePrediction, build_long_rows, ...`
