# /jz — Jean-Zay Cluster Operations Agent

Source of truth: [`.cursor/skills/tcfuse-jz/SKILL.md`](../../.cursor/skills/tcfuse-jz/SKILL.md).

This command activates the TC-Fuse Jean-Zay skill. **Before running, submitting, monitoring, or rsync'ing anything on Jean-Zay**, read the SKILL.md. All behavior rules, storage layout, rsync command, SLURM configs, and walltimes are defined there.

Keep docs in sync: when SLURM conventions, storage layout, or rsync filters change, update SKILL.md and this file together.

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
| CPU partition routing (downloads / preprocessing / archive) | "Partition routing (CPU jobs)" |
| Resume from checkpoint | "Checkpoint and resume" |
| Storage layout ($WORK, $SCRATCH, $STORE) | "Storage layout" |
