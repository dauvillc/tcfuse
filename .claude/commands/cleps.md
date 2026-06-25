# /cleps — CLEPS Cluster Operations Agent

Source of truth: [`.agents/cleps.md`](../../.agents/cleps.md).

This command activates the TC-Fuse CLEPS skill. **Before running, submitting, monitoring, or rsync'ing anything on CLEPS**, read the skill file. All behavior rules, storage layout, rsync command, SLURM configs, and walltimes are defined there.

CLEPS is a second cluster alongside Jean-Zay (`/jz`). Key differences: pixi (no modules), W&B **online** (no offline sync), persistent scratch (no archive), internet on all compute nodes, and `cpus_per_gpu` for GPU jobs.

Keep docs in sync: when SLURM conventions, storage layout, or rsync filters change, update `.agents/cleps.md` and this file together.

---

## Quick pointer

| Need | Start here (in `.agents/cleps.md`) |
|---|---|
| Sync local → CLEPS | "Local CLI tools" (`rsynctf` covers both clusters) |
| Submit a job (training / debug / CPU) | "Job submission workflow" |
| Monitor queue / live logs | "Job monitoring (raw SSH)" |
| Cancel a job | "Job monitoring" → Cancel a job |
| Environment (pixi, no modules) | "Environment (pixi, no modules)" |
| W&B online | "W&B online mode" |
| One-time CLEPS setup | "One-time setup" |
| Setup configs and walltimes | "SLURM hardware configs" |
| Partition routing (arches / gpu / cpu_devel) | "SLURM hardware configs" → Partition routing |
| Resume from checkpoint | "Checkpoint and resume" |
| Storage layout ($HOME, $SCRATCH, /local) | "Storage layout" |
| How CLEPS differs from Jean-Zay | "Key differences from Jean-Zay" |
