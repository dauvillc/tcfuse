# CLAUDE.md — TC Multi-Source Fusion Project

> This project uses an agent-agnostic documentation layout under `.agents/`.
> Treat this file as the session bootstrap pointer for Claude Code.

## Read first

Always start by reading the project's core context:

- [`.agents/context.md`](.agents/context.md) — project overview, data abstraction, repo structure, coding rules, architecture, W&B conventions, workflow rules.

`.agents/context.md` is the single source of truth for project-wide context. **Coding style** (human readability, inline comments, factorization, validation policy) lives in the § Human-readable code section there, with examples in [`.agents/coding-style.md`](.agents/coding-style.md). Do not duplicate its content here.

## On-demand skills (`.agents/`)

When a task touches one of these areas, read the matching skill file before making changes. The Claude slash commands in `.claude/commands/` are thin redirects to these skills.

| Topic | Skill file | Claude slash command |
|---|---|---|
| Dataset preprocessing (per-source HDF5, assembled storms, splits, normalization) | [`.agents/preprocess.md`](.agents/preprocess.md) | `/preprocess` |
| Jean-Zay cluster operations (rsync, SLURM, monitoring, W&B sync, checkpoints) | [`.agents/jz.md`](.agents/jz.md) | `/jz` |
| CLEPS cluster operations (pixi, W&B online, persistent scratch, SLURM, monitoring) | [`.agents/cleps.md`](.agents/cleps.md) | `/cleps` |
| Publication-quality figures (style.py, SVG output, thematic plotting modules) | [`.agents/visualize.md`](.agents/visualize.md) | `/visualize` |
| Model backbone architecture (embedding/encoder/decoder design, candidate backbones, pre-training task) | [`.agents/architecture.md`](.agents/architecture.md) | `/architecture` |
| Basedpyright diagnostics workflow | [`.agents/pyright-fixer.md`](.agents/pyright-fixer.md) | (none) |

## Update protocol

See [`.agents/context.md`](.agents/context.md) § Workflow rules, item 6 for the full update protocol. Never duplicate content in this file; this file only points at the source of truth.
