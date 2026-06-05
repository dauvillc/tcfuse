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

## Agent behavior rules

1. **Sync before every job.** Run the rsync step (see below) before submitting any SLURM job, no exceptions.
2. **Preflight before submission.** After syncing and before submitting, run `bash scripts/slurm/preflight_check.sh` on the login node. Abort if any check fails.
3. **Verify the job launched.** After submission, immediately run `squeue -u $USER` on the login node and confirm the job ID appears. Report the job ID and its partition/state to the user.
4. **Format monitoring output.** Never dump raw `squeue`/`sacct` output. Parse and format it into a readable table before presenting it to the user (see Monitoring section).
5. **Ask before cancelling.** Never call `scancel` without explicit user confirmation.
6. **Pick the right CPU partition.** Downloads that need internet → `setup=jz_prepost` (`prepost`). Preprocessing / eval without internet → `setup=jz_cpu` (default cpu, no `slurm_partition`). Archiving to `$STORE` is handled automatically on the `archive` partition by `submit_archive_job()` — never reuse `jz_prepost` for those jobs.
7. **Never hardcode paths.** Reference cluster paths as `$WORK/tcfuse` or `$SCRATCH/tcfuse`, not as absolute paths. Note: `$HOME` is a separate linkhome symlink — code lives under `$WORK`, not `$HOME`.
8. **Report errors clearly.** If an SSH command fails or a preflight check fails, show the exact error and suggest a fix before proceeding.

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints, virtual env | Persistent, backed up, slower I/O. **`$HOME` is a separate linkhome dir — do not use it for project files.** |
| `$SCRATCH` | Raw data, preprocessed tensors, DataLoader cache | Fast NVMe — **purged after 30 days** |
| `$STORE` | Long-term archival of final weights, processed datasets | Cold storage, not for training I/O |

**Rule:** DataLoaders always read from `$SCRATCH`. After preprocessing, always copy a backup to `$STORE`.

## Rsync — sync local → JZ

Run this before every job submission. Execute it from the local project root `/home/cdauvill/inria/tcfuse/`:

```bash
# $WORK on JZ differs from $HOME — expand it first
JZ_WORK=$(ssh jz 'bash -l -c "echo \$WORK"')
rsync -avz --filter=':- .gitignore' \
  --exclude='.git/' \
  --exclude='.pixi/' \
  --exclude='outputs/' \
  --exclude='wandb/' \
  --exclude='lightning_logs/' \
  --exclude='*.ckpt' \
  /home/cdauvill/inria/tcfuse/ "jz:${JZ_WORK}/tcfuse/"
```

**What this does:**
- Respects `.gitignore` (skips `__pycache__/`, `*.pyc`, `*.egg-info`, etc.)
- Explicitly excludes `.git/` and `.pixi/` (JZ uses a different environment)
- Excludes training outputs (`outputs/`, `wandb/`, `*.ckpt`) so they don't overwrite JZ-side results
- Syncs everything else: `src/`, `conf/`, `scripts/`, `notebooks/`, `pixi.toml`, `requirements-jz.txt`, etc.

After rsync, confirm the transfer completed without errors before proceeding.

## Job submission workflow

Follow these steps in order every time a job is submitted:

### Step 1 — Rsync
Run the rsync command above from the local machine.

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

## Job monitoring

### Queue status (running / pending jobs)
```bash
ssh jz "squeue -u \$USER --format='%.10i %.20j %.10T %.12M %.12l %.6D %R' --sort=-T"
```

Format the output as a table with headers:

| JOBID | NAME | STATE | ELAPSED | TIMELIMIT | NODES | REASON/NODELIST |
|---|---|---|---|---|---|---|
| 12345678 | tc_fusion | RUNNING | 1:23:45 | 100:00:00 | 1 | gpu123 |

States to highlight: `RUNNING` (good), `PENDING` (waiting — show the reason), `FAILED`/`CANCELLED` (alert the user).

### Recently completed jobs
```bash
ssh jz "sacct -u \$USER --format=JobID,JobName,State,Start,End,Elapsed,ExitCode \
  --starttime=\$(date -d '3 days ago' +%Y-%m-%d) -n"
```

Format as:

| JOBID | NAME | STATE | START | END | ELAPSED | EXIT |
|---|---|---|---|---|---|---|
| 12345678 | tc_fusion | COMPLETED | 2025-09-01T08:00 | 2025-09-01T10:30 | 02:30:00 | 0:0 |

### Live job output
```bash
ssh jz "tail -n 50 \$WORK/tcfuse/submitit/<jobid>_0_log.out"
# or follow live:
ssh jz "tail -f \$WORK/tcfuse/submitit/<jobid>_0_log.out"
```

submitit log files are at `$WORK/tcfuse/submitit/<jobid>_<task>_log.out` and `…_log.err`.

### Quota check
```bash
ssh jz "idrquota -m"        # $WORK and $SCRATCH
ssh jz "idrquota -s -m"     # $STORE
```

## Cancel a job
Always ask the user for confirmation first, then:
```bash
ssh jz "scancel <jobid>"
```

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

| Config | Partition | Hardware | CPUs | Max walltime |
|---|---|---|---|---|
| `jz_gpu_v100` | `gpu_p13` | 4× V100 32 GB | 40 (Intel) | 100 h (qos_gpu-t4) |
| `jz_gpu_a100` | `gpu_p5` | 8× A100 80 GB | 64 (AMD Milan) | **20 h** (no t4 QoS) |
| `jz_gpu_h100` | `gpu_p6` | 4× H100 80 GB | 96 (Intel) | 100 h (qos_gpu_h100-t4) |
| `jz_cpu` | *(default cpu)* | Regular CPU nodes — preprocessing, eval (no internet) | 40 (Intel) | per SLURM default |
| `jz_prepost` | `prepost` | Pre/post CPU nodes — data downloads (**internet access**) | 4 (Intel) | 20 h |
| `jz_archive` | `archive` | Archive nodes — SCRATCH→STORE tarballs (reference only; applied by `submit_archive_job()`) | 1 | 4 h |

**Do not** point preprocessing or eval jobs at `prepost` unless they need internet. `jz_cpu` intentionally omits `slurm_partition` so SLURM uses the default cpu partition.

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
