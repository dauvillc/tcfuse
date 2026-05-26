# /jz — Jean-Zay Cluster Operations Agent

Source of truth: [`.cursor/skills/tcfuse-jz/SKILL.md`](../../.cursor/skills/tcfuse-jz/SKILL.md).

This command activates the TC-Fuse Jean-Zay skill. **Before running, submitting,
monitoring, or rsync'ing anything on Jean-Zay**, read the SKILL.md. The cluster
quick reference (setup configs, partitions, walltimes) also lives in
[`.cursor/rules/tcfuse-core.mdc`](../../.cursor/rules/tcfuse-core.mdc).

---

## Agent behavior rules

1. **Sync before every job.** Always rsync local → JZ before any SLURM submission (full command in SKILL.md).
2. **Preflight before submission.** Run `bash scripts/slurm/preflight_check.sh` on the login node; abort on failure.
3. **Verify the job launched.** After `submit`, confirm the job ID appears in `squeue` and report partition/state.
4. **Format monitoring output.** Never dump raw `squeue`/`sacct`; render a readable table (see SKILL.md monitoring section).
5. **Ask before cancelling.** Never call `scancel` without explicit user confirmation.
6. **Never hardcode paths.** Use `$WORK/tcfuse`, `$SCRATCH/tcfuse`, `$STORE` — `$HOME` is a separate linkhome dir.
7. **Keep docs in sync:** when SLURM conventions, storage layout, or rsync filters change, update `.cursor/skills/tcfuse-jz/SKILL.md` and this command file together.

---

## Quick pointer

| Need | Start here (in SKILL.md) |
|---|---|
| Sync local → JZ | "Rsync — sync local → JZ" |
| Submit a job (V100 / A100 / H100 / CPU) | "Job submission workflow" |
| Monitor queue / live logs | "Job monitoring" |
| Cancel a job | "Cancel a job" |
| Environment setup (modules) | "Environment setup (login node only)" |
| W&B offline mode + sync | "W&B offline mode" |
| One-time JZ setup | "One-time setup" |
| Setup configs and walltimes | "SLURM hardware configs" |
| Resume from checkpoint | "Checkpoint and resume" |
| Storage layout ($WORK, $SCRATCH, $STORE) | "Storage layout" |
