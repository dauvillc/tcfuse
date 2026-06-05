---
name: tcfuse-jz
description: >-
  Drive the Jean-Zay cluster from the local machine — rsync sync, preflight,
  SLURM submission via submitit, job monitoring, W&B offline sync, checkpoint
  resume, and storage quota. Use whenever a task involves `ssh jz`, $WORK,
  $SCRATCH, $STORE, SLURM partitions (gpu_p13 / gpu_p5 / gpu_p6 / cpu /
  prepost / archive),
  module loads, `submitit`, `setup=jz_*`, `paths=jz`, `requirements-jz.txt`,
  or running jobs on Jean-Zay.
---

# TC-Fuse Jean-Zay cluster operations

Claude Code: invoke `/jz` (reads this skill).

**Coding style:** follow [`.cursor/rules/tcfuse-core.mdc`](../../rules/tcfuse-core.mdc) § Human-readable code (priority).

## When to use

- Syncing the local project to Jean-Zay before any cluster job.
- Submitting, monitoring, or cancelling SLURM jobs.
- Managing W&B in offline mode and syncing runs from the login node.
- Picking the right `setup=jz_*` config for a given hardware target.
- Resuming a training run from a submitit checkpoint.
- Checking quotas, environment, or doing one-time setup on JZ.

## Local CLI tools

These commands are available in the local shell (defined in `~/.bash_aliases` and `~/.local/bin`). They SSH into Jean-Zay and, where noted, pipe results through Claude. Prefer them over raw SSH for the matching tasks.

**JZ username:** `ute68qj` (stored as `_JZ_USER` in the shell environment).

### Rsync

| Alias | What it does |
|---|---|
| `rsynctf` | `rsync -avzP --exclude-from='.rsyncignore' ~/inria/tcfuse jz:work` — syncs the local project to `$WORK/tcfuse/` on JZ |

### Queue and monitoring (shell functions, `~/.bash_aliases`)

| Command | Usage | What it does |
|---|---|---|
| `jzq` | `jzq` | One-shot `squeue` for `ute68qj` with a standard format. |
| `jzwatch` | `jzwatch` | Live-refresh the queue every 30 s (Ctrl-C to exit). |
| `jzpipe <id> …` | `jzpipe 111 222 333` | Live-refresh only the listed job IDs every 30 s — useful for watching a multi-step pipeline. |
| `jzlog <jobid>` | `jzlog 12345678` | Finds the stdout log (same search order as below) and `tail -f`s it live. |
| `jzinfo <jobid>` | `jzinfo 12345678` | Runs `scontrol show job` — full SLURM metadata (partition, timelimit, nodes, gres, …). |
| `jzeff <jobid>` | `jzeff 12345678` | Runs `seff` — CPU/GPU efficiency report (completed jobs only). |
| `jzcancel <jobid>` | `jzcancel 12345678` | Runs `scancel`. **Always ask the user for confirmation before calling this.** |

### AI-assisted analysis (shell functions, `~/.bash_aliases`)

| Command | Usage | What it does |
|---|---|---|
| `jzask <jobid> "<q>"` | `jzask 111 "why is val_loss plateauing?"` | Fetches the last 150 log lines and asks Claude a one-shot free-form question about the job. |
| `jzchat <jobid>` | `jzchat 12345678` | Fetches the last 300 stdout + 50 stderr lines and opens an **interactive** Claude session pre-loaded with the log — for open-ended investigation. |

### AI-assisted analysis (standalone scripts, `~/.local/bin`)

| Command | Usage | What it does |
|---|---|---|
| `jzstatus [-n N]` | `jzstatus` or `jzstatus --lines 120` | Fetches SLURM queue + last N log lines for every active job; Claude summarises progress, metrics, health. |
| `jzdebug <jobid>` | `jzdebug 12345678` | Pulls stdout, stderr, and `sacct` for a failed job; Claude gives root cause + fix. |
| `jzreport <jobid> [--save]` | `jzreport 12345678 --save` | Generates a structured markdown report (summary, config, results, efficiency, issues). `--save` writes to `~/.jz_reports/<jobid>.md`. |
| `jzcompare <id1> <id2> …` | `jzcompare 111 222 333` | Compares 2–5 runs side-by-side: config diffs, metrics table, best-run verdict, what to try next. |

**Log search order** (used by `jzlog`, `jzask`, `jzchat`, `jzdebug`, `jzreport`, `jzstatus`):
1. `scontrol StdOut` field
2. `$WORK/tcfuse/submitit/<jobid>_*_log.out`
3. `$WORK/motif/submitit/<jobid>_*_log.out`
4. `$WORK/motif/slurm/jz/logs/slurm-<jobid>.out`

## Agent behavior rules

1. **Sync before every job.** Run `rsynctf` from the local project root before submitting any SLURM job, no exceptions.
2. **Preflight before submission.** After syncing and before submitting, run `bash scripts/slurm/preflight_check.sh` on the login node. Abort if any check fails.
3. **Verify the job launched.** After submission, immediately run `squeue -u $USER` on the login node and confirm the job ID appears. Report the job ID and its partition/state to the user.
4. **Use local tools for monitoring and debugging.** For job status use `jzstatus`; for diagnosing a failure use `jzdebug <jobid>`; for post-run analysis use `jzreport <jobid>`. Only fall back to raw SSH + `squeue`/`sacct` when these tools are unavailable.
5. **Ask before cancelling.** Never call `scancel` without explicit user confirmation.
6. **Pick the right partition.** Downloads that need internet → `setup=jz_prepost` (`prepost`). Preprocessing / eval without internet → `setup=jz_cpu` (default cpu, no `slurm_partition`). Archiving to `$STORE` is handled automatically on `prepost` by `submit_archive_job()` — `$STORE` is only mounted on prepost nodes.
7. **Never hardcode paths.** Reference cluster paths as `$WORK/tcfuse` or `$SCRATCH/tcfuse`, not as absolute paths. Note: `$HOME` is a separate linkhome symlink — code lives under `$WORK`, not `$HOME`.
8. **Report errors clearly.** If an SSH command fails or a preflight check fails, show the exact error and suggest a fix before proceeding.

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints, virtual env | Persistent, backed up, slower I/O. **`$HOME` is a separate linkhome dir — do not use it for project files.** |
| `$SCRATCH` | Raw data, preprocessed tensors, DataLoader cache | Fast NVMe — purged regularly (no fixed 30-day guarantee) |
| `$JOBSCRATCH` | Intra-job temp files, fast DataLoader cache during a run | Fastest NVMe — **deleted when the job ends**, not backed up |
| `$STORE` | Long-term archival of final weights, processed datasets | Cold storage, not for training I/O |

**Rule:** DataLoaders always read from `$SCRATCH`. After preprocessing, always copy a backup to `$STORE`.

## Job submission workflow

Follow these steps in order every time a job is submitted:

### Step 1 — Rsync
Run `rsynctf` from the local project root.

### Step 2 — Preflight check
```bash
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && bash scripts/slurm/preflight_check.sh"
```
Abort if any check fails. Fix the issue (quota, missing env, etc.) before continuing.

### Step 3 — Submit the job
```bash
# Training (V100 default)
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_gpu_v100 experiment=<name>"

# Training (A100)
ssh jz "cd \$WORK/tcfuse && module load arch/a100 && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_gpu_a100 experiment=<name>"

# Training (H100)
ssh jz "cd \$WORK/tcfuse && module load arch/h100 && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_gpu_h100 experiment=<name>"

# Preprocessing / eval (default CPU partition — no internet)
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/preprocess/<source>.py paths=jz setup=jz_cpu"

# Data download (prepost partition — internet access)
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/preprocess/<source>/download_<source>.py paths=jz setup=jz_prepost"
```

### Step 4 — Verify launch
```bash
ssh jz "squeue -u \$USER --format='%.10i %.15j %.8T %.10M %.10l %R'"
```
Confirm the new job ID appears and report it to the user. If it does not appear within ~10 seconds, check for immediate failures:
```bash
ssh jz "ls -lt \$WORK/tcfuse/submitit/ | head -5"
```

## Job monitoring — raw SSH fallbacks

Use the local CLI tools (see above) for day-to-day monitoring. These raw commands are fallbacks or when a specific format is needed.

### Queue snapshot
```bash
ssh jz "squeue -u \$USER --format='%.10i %.20j %.10T %.12M %.12l %.6D %R' --sort=-T"
```

Format as a table — states to highlight: `RUNNING` (good), `PENDING` (show the reason), `FAILED`/`CANCELLED` (alert the user).

### Recently completed jobs
```bash
ssh jz "sacct -u \$USER \
  --format=JobID,JobName,State,Start,End,Elapsed,AllocGRES,ExitCode \
  --starttime=\$(date -d '3 days ago' +%Y-%m-%d) -n"
```

### Live log tail (when `jzlog` is unavailable)
```bash
ssh jz "tail -f \$WORK/tcfuse/submitit/<jobid>_0_log.out"
```
Log files: `$WORK/tcfuse/submitit/<jobid>_<task>_log.out` / `…_log.err`.

### Interactive session on a compute node
```bash
# GPU node (replace <acct> with e.g. ute68qj@v100 or ute68qj@a100)
ssh jz "srun --pty -n1 --gres=gpu:1 --time=00:30:00 --account=<acct> bash"
# CPU node
ssh jz "srun --pty -n1 --time=00:30:00 --account=<acct>@cpu bash"
```
Requires the correct `--account` and a matching QoS with available quota.

### Quota check
```bash
ssh jz "idr_quota_user -m"          # personal $WORK and $SCRATCH usage
ssh jz "idr_quota_user -s"          # personal $STORE usage
ssh jz "idr_quota_project -w"       # project-level $WORK quota
```

### Hour consumption
```bash
ssh jz "idracct"        # CPU and GPU hour consumption for the current project
ssh jz "idr_compuse"    # project consumption state (% of allocation used)
```
Check these when jobs are slow to schedule or allocation warnings appear.

## Environment setup (login node only)

```bash
module purge
module load pytorch-gpu/py3/2.8.0
```

**A100 and H100 require loading the arch module first:**
```bash
module load arch/a100 && module load pytorch-gpu/py3/2.8.0   # A100
module load arch/h100 && module load pytorch-gpu/py3/2.8.0   # H100
```

**Internet access on compute nodes:** only the **`prepost`** partition has outbound internet. Regular CPU and GPU compute nodes have none. The login node has internet for pip installs, W&B auth, and ad-hoc transfers.

**Partition routing (CPU jobs):**

| Job type | Setup config | Partition | Notes |
|---|---|---|---|
| Data downloads (S3, HTTP, …) | `jz_prepost` | `prepost` | Only partition with internet on compute nodes; queues are tighter — reserve it for jobs that truly need network I/O |
| Preprocessing, eval, other CPU work (no internet) | `jz_cpu` | *(default cpu — do not set `slurm_partition`)* | Preferred for heavy CPU work; easier to schedule than `prepost` |
| Archiving SCRATCH → STORE tarballs | *(automatic)* | `archive` | Submitted by `submit_archive_job()` in `src/tcfuse/utils/archive.py`; not a `setup=` override |

Use the generated `requirements-jz.txt` overlay for pip installs on the login node, and regenerate it locally with `pixi run export-jz-requirements` after changing `pixi.toml`.

## W&B offline mode

Jobs must run with W&B offline. The `slurm_setup` commands in `conf/setup/jz_<hw>.yaml` set `WANDB_MODE=offline` automatically.

After a job completes, sync from the login node:
```bash
ssh jz "wandb sync \$WORK/tcfuse/wandb/offline-run-*/"
```

## One-time setup

Run once on the JZ login node after cloning or a fresh rsync:
```bash
ssh jz "cd \$WORK/tcfuse && bash scripts/setup_jz.sh"
```

Verify afterwards:
```bash
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && bash scripts/slurm/preflight_check.sh"
```

## SLURM hardware configs

| Config | Partition | Hardware | CPUs | Max walltime | Dev QoS (smoke test) |
|---|---|---|---|---|---|
| `jz_gpu_v100` | `gpu_p13` | 4× V100 32 GB | 40 (Intel) | 100 h (`qos_gpu-t4`) | `qos_gpu-dev` — 2 h, 32 GPU/user |
| `jz_gpu_a100` | `gpu_p5` | 8× A100 80 GB | 64 (AMD Milan) | **20 h** (no t4 QoS) | `qos_gpu_a100-dev` — 2 h, 32 GPU/user |
| `jz_gpu_h100` | `gpu_p6` | 4× H100 80 GB | 96 (Intel) | 100 h (`qos_gpu_h100-t4`) | `qos_gpu_h100-dev` — 2 h, 32 GPU/user |
| `jz_cpu` | *(default cpu)* | Regular CPU nodes — preprocessing, eval (no internet) | 40 (Intel) | 20 h (`qos_cpu-t3`) | `qos_cpu-dev` — 2 h, 128 nodes/user |
| `jz_prepost` | `prepost` | Pre/post CPU nodes — data downloads (**internet access**) | 4 (Intel) | 20 h | — |

**Dev QoS** (`qos_*-dev`) gets shorter queues and is the right choice for smoke tests, environment checks, or quick debugging runs. Override on the CLI: `setup.slurm_qos=qos_gpu-dev setup.timeout_min=60`.

**Do not** point preprocessing or eval jobs at `prepost` unless they need internet. `jz_cpu` intentionally omits `slurm_partition` so SLURM uses the default cpu partition. Archive jobs (SCRATCH→STORE tarballs submitted by `submit_archive_job()`) use `prepost` — `$STORE` is only mounted on prepost nodes. `--cpu-bind=none` is passed to `srun` inside those jobs to avoid a CPU affinity conflict on single-CPU allocations.

Override individual SLURM params on the CLI:
```bash
python scripts/train.py paths=jz setup=jz_gpu_v100 setup.timeout_min=6000 experiment=<name>
```

Local debug without submitit:
```bash
python scripts/train.py experiment=<name> submitit=false
```

## `conf/setup/` convention

Keys are passed verbatim to `submitit.AutoExecutor.update_parameters()`.

Template for a new hardware config:
```yaml
slurm_partition: ???
slurm_nodes: 1
slurm_ntasks_per_node: 1
slurm_gpus_per_node: ???
slurm_cpus_per_task: 10
slurm_mem_gb: ???
timeout_min: ???
name: tc_fusion
slurm_account: ???
slurm_setup:
  - module purge
  - module load pytorch-gpu/py3/2.8.0
  - export WANDB_MODE=offline
```

## Checkpoint and resume

`TrainJob` implements `submitit.helpers.Checkpointable`. On SIGUSR1 (~60 s before timeout), submitit calls `checkpoint()` and requeues the job with `resume_run_id` set. Resume is automatic.

To manually resume a run:
```bash
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_<hw> experiment=<name> resume_run_id=<run_id>"
```

## Maintenance

When changing cluster-related conventions (storage layout, SLURM configs, environment setup, rsync filters), update this skill in the same PR. If triggers or behavior rules change, also update `.claude/commands/jz.md` and `.cursor/rules/tcfuse-core.mdc` (Jean-Zay quick reference table).
