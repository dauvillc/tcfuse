# Jean-Zay Cluster Operations

You are now in Jean-Zay (JZ) mode. You drive the cluster through `ssh jz` from the local machine. Apply every rule below for all cluster-related tasks in this session.

---

## Agent behavior rules

1. **Sync before every job.** Run the rsync step (see below) before submitting any SLURM job, no exceptions.
2. **Preflight before submission.** After syncing and before submitting, run `bash scripts/slurm/preflight_check.sh` on the login node. Abort if any check fails.
3. **Verify the job launched.** After submission, immediately run `squeue -u $USER` on the login node and confirm the job ID appears. Report the job ID and its partition/state to the user.
4. **Format monitoring output.** Never dump raw `squeue`/`sacct` output. Parse and format it into a readable table before presenting it to the user (see Monitoring section).
5. **Ask before cancelling.** Never call `scancel` without explicit user confirmation.
6. **Never hardcode paths.** Reference cluster paths as `$WORK/tcfuse` or `$SCRATCH/tcfuse`, not as absolute paths. Note: `$HOME` is a separate linkhome symlink — code lives under `$WORK`, not `$HOME`.
7. **Report errors clearly.** If an SSH command fails or a preflight check fails, show the exact error and suggest a fix before proceeding.

---

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints, virtual env | Persistent, backed up, slower I/O. **`$HOME` is a separate linkhome dir — do not use it for project files.** |
| `$SCRATCH` | Raw data, preprocessed tensors, DataLoader cache | Fast NVMe — **purged after 30 days** |
| `$STORE` | Long-term archival of final weights, processed datasets | Cold storage, not for training I/O |

**Rule:** DataLoaders always read from `$SCRATCH`. After preprocessing, always copy a backup to `$STORE`.

---

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

---

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

# Preprocessing (CPU node)
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/preprocess/<source>.py paths=jz setup=jz_cpu"
```

### Step 4 — Verify launch
```bash
ssh jz "squeue -u \$USER --format='%.10i %.15j %.8T %.10M %.10l %R'"
```
Confirm the new job ID appears and report it to the user. If it does not appear within ~10 seconds, check for immediate failures:
```bash
ssh jz "ls -lt \$WORK/tcfuse/submitit/ | head -5"
```

---

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

---

## Cancel a job
Always ask the user for confirmation first, then:
```bash
ssh jz "scancel <jobid>"
```

---

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

Compute nodes have **no internet access**. All pip installs, W&B auth, and data downloads must happen on the login node. Use the generated `requirements-jz.txt` overlay for pip installs on Jean-Zay, and regenerate it locally with `pixi run export-jz-requirements` after changing `pixi.toml`.

---

## W&B offline mode

Jobs must run with W&B offline. The `slurm_setup` commands in `conf/setup/jz_<hw>.yaml` set `WANDB_MODE=offline` automatically.

After a job completes, sync from the login node:
```bash
ssh jz "wandb sync \$WORK/tcfuse/wandb/offline-run-*/"
```

---

## One-time setup

Run once on the JZ login node after cloning or a fresh rsync:
```bash
ssh jz "cd \$WORK/tcfuse && bash scripts/setup_jz.sh"
```

Verify afterwards:
```bash
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && bash scripts/slurm/preflight_check.sh"
```

---

## SLURM hardware configs

| Config | Partition | Hardware | CPUs | Max walltime |
|---|---|---|---|---|
| `jz_gpu_v100` | `gpu_p13` | 4× V100 32 GB | 40 (Intel) | 100 h (qos_gpu-t4) |
| `jz_gpu_a100` | `gpu_p5` | 8× A100 80 GB | 64 (AMD Milan) | **20 h** (no t4 QoS) |
| `jz_gpu_h100` | `gpu_p6` | 4× H100 80 GB | 96 (Intel) | 100 h (qos_gpu_h100-t4) |
| `jz_cpu` | `prepost` | Pre/post CPU nodes — heavy preprocessing | 40 (Intel) | 20 h |
| `jz_prepost` | `prepost` | Pre/post CPU nodes — data downloads (internet access) | 4 (Intel) | 20 h |

Override individual SLURM params on the CLI:
```bash
python scripts/train.py paths=jz setup=jz_gpu_v100 setup.timeout_min=6000 experiment=<name>
```

Local debug without submitit:
```bash
python scripts/train.py experiment=<name> submitit=false
```

---

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

---

## Checkpoint and resume

`TrainJob` implements `submitit.helpers.Checkpointable`. On SIGUSR1 (~60 s before timeout), submitit calls `checkpoint()` and requeues the job with `resume_run_id` set. Resume is automatic.

To manually resume a run:
```bash
ssh jz "cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_<hw> experiment=<name> resume_run_id=<run_id>"
```
