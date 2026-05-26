# CLAUDE.md — TC Multi-Source Fusion Project

> This project's agent rules and skills live in Cursor's `.cursor/` directory.
> Treat this file as the session bootstrap pointer for Claude Code.

## Read first

Always start by reading the project's core context rule:

- [`.cursor/rules/tcfuse-core.mdc`](.cursor/rules/tcfuse-core.mdc) — project overview, data abstraction, repo structure, coding rules, architecture, W&B conventions, workflow rules.

`tcfuse-core.mdc` is the alwaysApply Cursor rule and the single source of truth for project-wide context. Do not duplicate its content here.

## On-demand skills (`.cursor/skills/`)

When a task touches one of these areas, read the matching `SKILL.md` before making changes. The Claude slash commands in `.claude/commands/` are thin redirects to these skills.

| Topic | Cursor skill | Claude slash command |
|---|---|---|
| Dataset preprocessing (per-source HDF5, assembled storms, splits, normalization) | [`.cursor/skills/tcfuse-preprocess/SKILL.md`](.cursor/skills/tcfuse-preprocess/SKILL.md) | `/preprocess` |
| Jean-Zay cluster operations (rsync, SLURM, monitoring, W&B sync, checkpoints) | [`.cursor/skills/tcfuse-jz/SKILL.md`](.cursor/skills/tcfuse-jz/SKILL.md) | `/jz` |
| Forecast output storage (`PredictionRun`, `SamplePrediction`, `ibtracs.parquet`) | [`.cursor/skills/tcfuse-predictions/SKILL.md`](.cursor/skills/tcfuse-predictions/SKILL.md) | `/predictions` |
| Publication-quality figures (style.py, SVG output, thematic plotting modules) | [`.cursor/skills/tcfuse-visualize/SKILL.md`](.cursor/skills/tcfuse-visualize/SKILL.md) | `/visualize` |
| Basedpyright diagnostics workflow | [`.cursor/skills/pyright-typing-fixer/SKILL.md`](.cursor/skills/pyright-typing-fixer/SKILL.md) | (none) |

## Update protocol

Cursor is the source of truth. When changing project conventions:

- Always-apply context → edit [`.cursor/rules/tcfuse-core.mdc`](.cursor/rules/tcfuse-core.mdc).
- Topic-specific docs → edit the matching `.cursor/skills/<topic>/SKILL.md`.
- Slash-command triggers or behavior rules → also update the matching `.claude/commands/<topic>.md` thin redirect.
- Never duplicate content in this file; this file only points at the Cursor source of truth.
