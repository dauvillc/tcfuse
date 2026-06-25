---
name: tcfuse-cleps
description: >-
  Drive the CLEPS (Inria Paris) cluster from the local machine — rsync sync,
  preflight, SLURM submission via submitit, job monitoring, checkpoint resume,
  and storage quota. Use whenever a task involves `ssh cleps`, $SCRATCH,
  CLEPS SLURM partitions (cpu_devel / gpu / arches), pixi on the cluster,
  `submitit`, `setup=cleps_*`, `paths=cleps`, or running jobs on CLEPS.
  CLEPS uses pixi (no modules) and W&B online (no offline sync), and its
  scratch is persistent (no archive step).
---

> **Content has moved.** The full skill documentation is in [`.agents/cleps.md`](../../.agents/cleps.md).
> Read that file for the key differences from Jean-Zay, agent behavior rules, storage layout, submission workflow, hardware configs, and checkpoint resume.
