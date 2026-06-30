# TC-Fuse Jean-Zay cluster operations

Claude Code: invoke `/jz` (reads this skill).

**Coding style:** follow [`.agents/context.md`](context.md) § Human-readable code (priority).

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
7. **Never hardcode paths — but always use a login shell for SSH.** Reference cluster paths as `$WORK/tcfuse` or `$SCRATCH/tcfuse`, not as absolute paths. Note: `$HOME` is a separate linkhome symlink — code lives under `$WORK`, not `$HOME`.

   `$WORK`, `$SCRATCH`, and `$STORE` are **not set in non-interactive SSH sessions**. Every `ssh jz "..."` command must wrap its payload in a login shell so IDRIS profile scripts are sourced:
   ```bash
   ssh jz "bash -l -c '<command>'"
   ```
   See the Resolved absolute paths table for hardcoded fallback values when needed. Do **not** use bare `ssh jz "$WORK/..."` — `$WORK` will be empty.
8. **Report errors clearly.** If an SSH command fails or a preflight check fails, show the exact error and suggest a fix before proceeding.

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints, virtual env | Persistent, backed up, slower I/O. **`$HOME` is a separate linkhome dir — do not use it for project files.** |
| `$SCRATCH` | Raw data, preprocessed tensors, DataLoader cache | Fast NVMe — purged regularly (no fixed 30-day guarantee) |
| `$JOBSCRATCH` | Intra-job temp files, fast DataLoader cache during a run | Fastest NVMe — **deleted when the job ends**, not backed up |
| `$STORE` | Long-term archival of final weights, processed datasets | Cold storage, not for training I/O |

**Rule:** DataLoaders always read from `$SCRATCH`. After preprocessing, always copy a backup to `$STORE`.

### Resolved absolute paths

| Variable | Absolute path |
|---|---|
| `$WORK` | `/lustre/fswork/projects/rech/xyw/ute68qj` |
| `$SCRATCH` | `/lustre/fsn1/projects/rech/xyw/ute68qj` |
| `$STORE` | `/lustre/fsstor/projects/rech/xyw/ute68qj` |
| `$HOME` | `/linkhome/rech/genini01/ute68qj` |

In SSH commands, prefer `bash -l -c '...'` (login shell resolves them automatically) over substituting these directly. Use the table only when you truly need a hardcoded path.

## Job submission workflow

Follow these steps in order every time a job is submitted:

### Step 1 — Rsync
Run `rsynctf` from the local project root.

### Step 2 — Preflight check
```bash
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && bash scripts/slurm/preflight_check.sh'"
```
Abort if any check fails. Fix the issue (quota, missing env, etc.) before continuing.

### Step 3 — Submit the job
```bash
# Training — V100 single GPU (smoke test / debug)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_v100 experiment=<name>'"

# Training — V100 full node (4 GPUs)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_4xv100 experiment=<name>'"

# Training — H100 single GPU (smoke test / debug)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load arch/h100 && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_h100 experiment=<name>'"

# Training — H100 full node (4 GPUs)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load arch/h100 && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_4xh100 experiment=<name>'"

# Preprocessing / eval (default CPU partition — no internet)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/preprocess/<source>.py paths=jz setup=jz_cpu'"

# Data download (prepost partition — internet access)
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/preprocess/<source>/download_<source>.py paths=jz setup=jz_prepost'"
```

### Step 4 — Verify launch
```bash
ssh jz "bash -l -c 'squeue -u \$USER --format=\"%.10i %.15j %.8T %.10M %.10l %R\"'"
```
Confirm the new job ID appears and report it to the user. If it does not appear within ~10 seconds, check for immediate failures:
```bash
ssh jz "bash -l -c 'ls -lt \$WORK/tcfuse/submitit/ | head -5'"
```

## Job monitoring — raw SSH fallbacks

Use the local CLI tools (see above) for day-to-day monitoring. These raw commands are fallbacks or when a specific format is needed.

### Queue snapshot
```bash
ssh jz "bash -l -c 'squeue -u \$USER --format=\"%.10i %.20j %.10T %.12M %.12l %.6D %R\" --sort=-T'"
```

Format as a table — states to highlight: `RUNNING` (good), `PENDING` (show the reason), `FAILED`/`CANCELLED` (alert the user).

### Recently completed jobs
```bash
ssh jz "bash -l -c 'sacct -u \$USER \
  --format=JobID,JobName,State,Start,End,Elapsed,AllocGRES,ExitCode \
  --starttime=\$(date -d \"3 days ago\" +%Y-%m-%d) -n'"
```

### Live log tail (when `jzlog` is unavailable)
```bash
ssh jz "bash -l -c 'tail -f \$WORK/tcfuse/submitit/<jobid>_0_log.out'"
```
Log files: `$WORK/tcfuse/submitit/<jobid>_<task>_log.out` / `…_log.err`.

### Interactive session on a compute node

The compute account is keyed by the **project group `xyw`**, not the username — use `xyw@v100`, `xyw@h100`, or `xyw@cpu` (verify with `sacctmgr -n show assoc user=$USER format=Account,Partition,QOS`).

```bash
# GPU node (V100 dev QoS)
ssh jz "srun --pty -n1 --gres=gpu:1 --time=00:30:00 --qos=qos_gpu-dev --account=xyw@v100 bash"
# CPU node
ssh jz "srun --pty -n1 --time=00:30:00 --account=xyw@cpu bash"
```
Requires the correct `--account` and a matching QoS with available quota.

**`$WORK`/`$SCRATCH` are empty inside `srun`, even under `bash -l`.** The IDRIS profile that exports them is not sourced on compute nodes, so `srun bash -l -c 'cd $WORK/...'` lands in `/...`. Resolve the path in a login-node shell first and let `srun` inherit the working directory:
```bash
# Right: outer bash -l -c resolves $WORK on the login node; srun inherits CWD.
ssh jz "bash -l -c 'cd \$WORK/tcfuse && srun -n1 --gres=gpu:1 --time=00:30:00 \
  --qos=qos_gpu-dev --account=xyw@v100 \
  bash -c \"module load pytorch-gpu/py3/2.8.0; <command>\"'"
```

### Quota check
```bash
ssh jz "bash -l -c 'idr_quota_user -m'"          # personal $WORK and $SCRATCH usage
ssh jz "bash -l -c 'idr_quota_user -s'"          # personal $STORE usage
ssh jz "bash -l -c 'idr_quota_project -w'"       # project-level $WORK quota
```

### Hour consumption
```bash
ssh jz "bash -l -c 'idracct'"        # CPU and GPU hour consumption for the current project
ssh jz "bash -l -c 'idr_compuse'"    # project consumption state (% of allocation used)
```
Check these when jobs are slow to schedule or allocation warnings appear.

## Environment setup (login node only)

```bash
module purge
module load pytorch-gpu/py3/2.8.0
```

**H100 requires loading the arch module first:**
```bash
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

**Run model: grouped segments.** W&B cannot truly resume an offline run, so each process launch (initial run, SLURM requeue, or manual restart) logs as a **distinct W&B "segment" run** with a unique id `<run_id>-<launch-ts>`, all sharing `group=<run_id>`. The `run_id` is the logical-run key (set in `scripts/train/train.py`) and also names the checkpoint dir. Continuity comes from the checkpoint (`global_step`, model/optimizer state) plus the W&B group overlay — **not** from W&B resume or `wandb sync --append`.

Sync from the login node — safe to run repeatedly, mid-run and after, because unique ids make each offline folder a distinct run and wandb's `.synced` marker skips already-synced folders (no double-counting):
```bash
ssh jz "bash -l -c 'wandb sync \$WORK/tcfuse/wandb/offline-run-*/'"
```
Do **not** pass `--append` (fragile, not idempotent) or `--include-synced` (re-uploads already-synced folders).

**Manual resume.** To continue an interrupted run by hand, relaunch with the existing `run_id` so training picks up that run's `last.ckpt` and logs into the same group:
```bash
python scripts/train/train.py experiment=<exp> paths=jz setup=jz_<hw> run_id=<existing id>
```

**Viewing.** In the W&B workspace, group runs by `group` and plot metrics against `trainer/global_step` for a single continuous curve across segments.

## One-time setup

Run once on the JZ login node after cloning or a fresh rsync:
```bash
ssh jz "bash -l -c 'cd \$WORK/tcfuse && bash scripts/setup_jz.sh'"
```

Verify afterwards:
```bash
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && bash scripts/slurm/preflight_check.sh'"
```

## SLURM hardware configs

| Config | Partition | Hardware | CPUs | Max walltime | Dev QoS (smoke test) |
|---|---|---|---|---|---|
| `jz_v100` | `gpu_p13` | 1× V100 32 GB | 10 (Intel) | 100 h (`qos_gpu-t4`) | `qos_gpu-dev` — 2 h, 32 GPU/user |
| `jz_4xv100` | `gpu_p13` | 4× V100 32 GB | 40 (Intel) | 100 h (`qos_gpu-t4`) | `qos_gpu-dev` — 2 h, 32 GPU/user |
| `jz_h100` | `gpu_p6` | 1× H100 SXM5 80 GB | 24 (Intel) | 100 h (`qos_gpu_h100-t4`) | `qos_gpu_h100-dev` — 2 h, 32 GPU/user |
| `jz_4xh100` | `gpu_p6` | 4× H100 SXM5 80 GB | 96 (Intel) | 100 h (`qos_gpu_h100-t4`) | `qos_gpu_h100-dev` — 2 h, 32 GPU/user |
| `jz_cpu` | *(default cpu)* | Regular CPU nodes — preprocessing, eval (no internet) | 40 (Intel) | 20 h (`qos_cpu-t3`) | `qos_cpu-dev` — 2 h, 128 nodes/user |
| `jz_prepost` | `prepost` | Pre/post CPU nodes — data downloads (**internet access**) | 4 (Intel) | 20 h | — |

**Dev QoS** (`qos_*-dev`) gets shorter queues and is the right choice for smoke tests, environment checks, or quick debugging runs. Override on the CLI: `setup.slurm_qos=qos_gpu-dev setup.timeout_min=60`.

**Do not** point preprocessing or eval jobs at `prepost` unless they need internet. `jz_cpu` intentionally omits `slurm_partition` so SLURM uses the default cpu partition. Archive jobs (SCRATCH→STORE tarballs submitted by `submit_archive_job()`) use `prepost` — `$STORE` is only mounted on prepost nodes. `--cpu-bind=none` is passed to `srun` inside those jobs to avoid a CPU affinity conflict on single-CPU allocations.

Override individual SLURM params on the CLI:
```bash
python scripts/train.py paths=jz setup=jz_4xv100 setup.timeout_min=6000 experiment=<name>
```

Local debug without submitit:
```bash
python scripts/train.py experiment=<name> submitit=false
```

## `conf/setup/` convention

Keys are passed verbatim to `submitit.AutoExecutor.update_parameters()`, except **`name`**: `make_executor(cfg, job_name)` always overrides `setup.name` with the script-supplied `job_name`, so `squeue` shows a task-specific label rather than the generic `preproc_jz` / `download_jz` defaults in the yaml.

### Preprocessing SLURM job names

| Script | SLURM name(s) |
|---|---|
| `prepare_infrared.py` | `prep_ir` |
| `prepare_era5.py` | `prep_era5` |
| `prepare_sar.py` | `prep_sar` |
| `prepare_pmw.py` | `prep_pmw_<sensat>` (e.g. `prep_pmw_amsr2_gcomw1`) — one job per sensor |
| `prepare_radar.py` | `prep_radar_<sensat>` (e.g. `prep_radar_gmi_gpm`) — one job per sensor |
| `assemble.py` | `assemble` |
| `compute_normalization.py` | `norm_<source_name>` — one job per source |
| `download_tc_primed.py` | `download_tc_primed` |
| `download_sar_cyclobs.py` | `download_sar` |
| Archive (via `submit_archive_job`) | `archive_<source_name>` |

Submitit log folders mirror the SLURM name: `$WORK/tcfuse/submitit/{job_name}_{timestamp}/`.

Example queue snapshot:
```bash
squeue -u $USER --format='%.10i %.25j %.8T %.10M %R'
#   12345678          prep_pmw_amsr2_gcomw1  RUNNING       1:23:00 ...
#   12345679                    prep_era5  RUNNING         45:00 ...
#   12345680     archive_pmw_amsr2_gcomw1  PENDING         0:00  ...
```

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
ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
  python scripts/train.py paths=jz setup=jz_<hw> experiment=<name> resume_run_id=<run_id>'"
```

## Hyperparameter sweeps

Sweeps use Hydra's **Optuna sweeper** (`hydra/sweeper=...`) with Hydra's **submitit launcher**
(`hydra/launcher=jz_sweep`) — not the inner submitit path. Full workflow, search-space rules, and
the objective contract → [`/sweep`](sweep.md). Jean-Zay specifics:

1. **Install the plugin into the JZ env first.** `hydra-optuna-sweeper` is in `requirements-jz.txt`,
   but the loaded module env may not have it yet. After `rsynctf`, on the login node:
   ```bash
   ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
     pip install --user hydra-optuna-sweeper && python -c \"import optuna, hydra_plugins\"'"
   ```
2. **Launch from the login node, not a job.** Compute nodes cannot submit SLURM jobs, so the
   sweeper process (it submits one job per trial via the launcher) must run on the login node.
   Run it under `tmux`/`nohup` so it survives a disconnect:
   ```bash
   ssh jz "bash -l -c 'cd \$WORK/tcfuse && module load pytorch-gpu/py3/2.8.0 && \
     python scripts/train/train.py --multirun experiment=pmw_gmi_sweep submitit=false \
     hydra/launcher=jz_4xv100_32g_sweep hydra/sweeper=perceiver_capacity'"
   ```
   `submitit=false` disables the inner executor so Hydra's launcher owns SLURM submission. The
   experiment pairs `setup=jz_4xv100_32g` (`trainer.devices=4`) with the launcher's
   `tasks_per_node=4`, so each trial is a 4× V100 DDP job — **keep those two equal** or DDP hangs.
   `hydra.launcher.array_parallelism` caps concurrent trials (keep it equal to
   `hydra.sweeper.n_jobs`; at 4 trials × 4 GPUs that is 16 V100 in flight — lower both if quota is
   tight).
3. **Results.** Best config + objective land in `multirun/<date>/<time>/optimization_results.yaml`
   (authoritative, independent of W&B). Per-trial runs are offline `pmw-gmi-sweep-<run_id>` folders —
   `wandb sync` them as usual (see W&B offline mode above) to inspect curves.
4. **Launcher configs.** `conf/hydra/launcher/jz_4xv100_32g_sweep.yaml` (default — 4× V100 32 GB
   DDP) and the single-GPU `jz_sweep.yaml` (H100) / `jz_v100_sweep.yaml` (V100, cheap dry runs)
   mirror the matching `conf/setup/jz_*` SLURM spec.

## Maintenance

When changing cluster-related conventions (storage layout, SLURM configs, environment setup, rsync filters), update this skill in the same PR. If triggers or behavior rules change, also update `.claude/commands/jz.md` and [`.agents/context.md`](context.md) (Jean-Zay quick reference table).
