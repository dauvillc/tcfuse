# TC-Fuse CLEPS cluster operations

Claude Code: invoke `/cleps` (reads this skill).

**Coding style:** follow [`.agents/context.md`](context.md) § Human-readable code (priority).

CLEPS is the Inria Paris cluster. It is a second launch target alongside Jean-Zay
([`.agents/jz.md`](jz.md)). The two clusters differ in important ways — read the
§ Key differences from Jean-Zay table before reusing JZ habits here.

## When to use

- Syncing the local project to CLEPS before any cluster job.
- Submitting, monitoring, or cancelling SLURM jobs on CLEPS.
- Picking the right `setup=cleps_*` config for a given hardware target.
- Resuming a training run from a submitit checkpoint.
- Checking quotas, environment, or doing one-time setup on CLEPS.

## Key differences from Jean-Zay

| Aspect | Jean-Zay | CLEPS |
|---|---|---|
| Environment | `module load pytorch-gpu/...` | **pixi** — launch with `pixi run python …`, no modules |
| W&B | **offline** + login-node `wandb sync` | **online** — runs log directly, no sync step |
| Scratch lifetime | purged regularly → archive to `$STORE` | **persistent** (user backs up) — no `$STORE`, no archive jobs |
| Internet on compute nodes | only `prepost` partition | **all nodes** — downloads run on the normal CPU partition |
| GPU CPU binding | `cpus_per_task` | **`cpus_per_gpu`** (via `slurm_additional_parameters`) |
| Code location | `$WORK/tcfuse` | `$HOME/tcfuse` (the rsync target) |

## Local CLI tools

`rsynctf` (defined in `~/.bash_aliases`) **already syncs to both clusters** —
JZ (`jz:work`) and CLEPS (`cleps:`, i.e. `$HOME/tcfuse`). Run it from the local
project root before any CLEPS job.

There are currently **no `cleps*` shell helpers** (unlike the `jz*` aliases/
scripts). Monitoring and submission use raw `ssh cleps`. The SSH host `cleps`
is defined in `~/.ssh/config` (ProxyJump `bastion-paris`, ControlMaster on).

## Agent behavior rules

1. **Sync before every job.** Run `rsynctf` from the local project root before
   submitting any SLURM job, no exceptions.
2. **Preflight before submission.** After syncing, run
   `pixi run bash scripts/slurm/preflight_check_cleps.sh` on the login node.
   Abort if any check fails.
3. **Verify the job launched.** After submission, immediately run
   `squeue -u $USER` on the login node and confirm the job ID appears. Report the
   job ID and its partition/state to the user.
4. **Ask before cancelling.** Never call `scancel` without explicit user
   confirmation.
5. **Pick the right config.** Training (4 H200) → `setup=cleps_arches_x4`. Single
   H200 debug → `setup=cleps_arches`. Quick smoke test → `setup=cleps_rtx8000`
   (generic gpu, may be preempted; full RTX8000 node → `setup=cleps_rtx8000_x3`).
   Preprocessing / eval / downloads → `setup=cleps_cpu`.
6. **Use a login shell for SSH.** `$SCRATCH` (and other cluster env vars) are
   **not set in non-interactive SSH sessions**. Wrap every payload in a login
   shell that sources the profile:
   ```bash
   ssh cleps "bash -lc 'source /etc/profile; <command>'"
   ```
   Reference cluster paths as `$HOME/tcfuse` or `$SCRATCH/tcfuse`, never as
   absolute paths.
7. **Report errors clearly.** If an SSH or preflight check fails, show the exact
   error and suggest a fix before proceeding.

## Storage layout

| Location | Use | Notes |
|---|---|---|
| `$HOME` | Code, pixi env (`.pixi`), `figures` | Backed up, **100 GB quota** — keep it small |
| `$SCRATCH` | Raw data, preprocessed tensors, **checkpoints**, wandb dir, predictions | Lustre, **persistent on CLEPS**, 20 TB project quota. User makes their own backups |
| `/local` (`$TMP_DIR`) | Intra-job temp / fast local scratch | **Deleted when the job ends** |

**Unlike Jean-Zay, scratch is kept and there is no archive step** — every
`cleps_*` setup sets `archive: false`. Checkpoints live on `$SCRATCH` (home is
only 100 GB), which differs from JZ where they go to `$WORK`.

Path resolution is via [`conf/paths/cleps.yaml`](../conf/paths/cleps.yaml).
Select it at launch with `paths=cleps`. There is no real `$STORE`; the `store`
key aliases `$SCRATCH`.

## SLURM hardware configs

| Config | Partition | Account | Hardware | CPU binding | Max walltime |
|---|---|---|---|---|---|
| `cleps_cpu` | *(default `cpu_devel`)* | — | CPU nodes (12-18 cores), preprocessing / eval / **downloads** | `cpus_per_task=16` | 1 week |
| `cleps_arches` | `arches` | `arches` | 1× H200 — single-GPU debug | `cpus_per_gpu=32` | 2 days* |
| `cleps_arches_x4` | `arches` | `arches` | 4× H200 — full-node training default | `cpus_per_gpu=32` | 2 days* |
| `cleps_rtx8000` | `gpu` | — | 1× RTX8000 48 GB — smoke test | `cpus_per_gpu=16` | 2 days |
| `cleps_rtx8000_x3` | `gpu` | — | 3× RTX8000 48 GB — full RTX8000 node | `cpus_per_gpu=16` | 2 days |

\* The `arches` node spec (GPUs/node, CPUs/GPU, walltime cap) is **not in the
public CLEPS docs**. The config defaults to `gpu:h200:4` / `cpus_per_gpu=32` /
`trainer.devices=4`. Verify and adjust:
```bash
ssh cleps "bash -lc 'sinfo -p arches -N -o \"%n %G %c\" && scontrol show partition arches | grep -i maxtime'"
```

**Partition routing:**
- **Full training** → `cleps_arches_x4` (4× H200, priority partition, both
  `-p arches` and `-A arches` are set in the config).
- **Single-GPU debug on arches** → `cleps_arches` (1× H200, same priority partition).
- **Smoke test / quick debug** → `cleps_rtx8000` on the generic `gpu` partition. This
  may be **preempted** by jobs from the partitions that own the hardware. To avoid
  preemption, exclude proprietary nodes on the CLI:
  `setup.slurm_additional_parameters.exclude=gpu009,gpu01[2-3],gpu01[5-7]`.
- **Full RTX8000 node** → `cleps_rtx8000_x3` (3 GPUs/node, same `gpu` partition).
- **CPU work** (preprocessing, eval, downloads) → `cleps_cpu`. CLEPS compute
  nodes have internet, so downloads need no special partition.

Override individual SLURM params on the CLI, e.g.:
```bash
pixi run python scripts/train/train.py paths=cleps setup=cleps_arches_x4 \
  setup.timeout_min=120 experiment=<name>
```

The `conf/setup/` convention (keys passed verbatim to
`submitit.AutoExecutor.update_parameters()`, `name` overridden per script) is
shared with Jean-Zay — see [`.agents/jz.md`](jz.md) § `conf/setup/` convention
and the preprocessing SLURM job-name table.

## Job submission workflow

### Step 1 — Rsync
Run `rsynctf` from the local project root (syncs to both clusters).

### Step 2 — Preflight check
```bash
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run bash scripts/slurm/preflight_check_cleps.sh'"
```
Abort if any check fails.

### Step 3 — Submit the job
```bash
# Training — H200 full node (arches priority partition, 4 GPUs)
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run python scripts/train/train.py paths=cleps setup=cleps_arches_x4 experiment=<name>'"

# Training — H200 single GPU (arches priority partition, debug)
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run python scripts/train/train.py paths=cleps setup=cleps_arches experiment=<name>'"

# Debug / smoke test (generic gpu partition, RTX8000 single GPU)
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run python scripts/train/train.py paths=cleps setup=cleps_rtx8000 \
  setup.timeout_min=60 experiment=<name>'"

# Preprocessing / eval / downloads (default cpu_devel partition)
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run python scripts/preprocess/<source>.py paths=cleps setup=cleps_cpu'"
```

### Step 4 — Verify launch
```bash
ssh cleps "bash -lc 'squeue -u \$USER --format=\"%.10i %.20j %.8T %.10M %.10l %R\"'"
```
Confirm the new job ID appears and report it to the user.

## Job monitoring (raw SSH)

No `cleps*` helper scripts exist yet — use raw SSH.

### Queue snapshot
```bash
ssh cleps "bash -lc 'squeue -u \$USER --format=\"%.10i %.20j %.10T %.12M %.12l %.6D %R\" --sort=-T'"
```
Highlight states: `RUNNING` (good), `PENDING` (show the reason), `FAILED`/
`CANCELLED`/`PREEMPTED` (alert the user — preemption is possible on the generic
`gpu` partition).

### Recently completed jobs
```bash
ssh cleps "bash -lc 'sacct -u \$USER \
  --format=JobID,JobName,State,Start,End,Elapsed,AllocGRES,ExitCode \
  --starttime=\$(date -d \"3 days ago\" +%Y-%m-%d) -n'"
```

### Full job metadata / efficiency
```bash
ssh cleps "bash -lc 'scontrol show job <jobid>'"   # partition, timelimit, gres, …
ssh cleps "bash -lc 'seff <jobid>'"                # CPU/GPU efficiency (completed jobs)
```

### Live log tail
submitit log folders are at `~/tcfuse/submitit/{job_name}_{timestamp}/`:
```bash
ssh cleps "bash -lc 'tail -f ~/tcfuse/submitit/<job_name>_<ts>/<jobid>_0_log.out'"
```

### Cancel a job (ask first)
```bash
ssh cleps "bash -lc 'scancel <jobid>'"   # only with explicit user confirmation
```

### Interactive session
```bash
# GPU node (debug)
ssh cleps "salloc -p gpu --gres=gpu:rtx8000:1 --cpus-per-gpu=16 --time=00:30:00"
# CPU node (default cpu_devel)
ssh cleps "salloc -c 8 --time=00:30:00"
```

### Quota check
```bash
ssh cleps "bash -lc 'lfs quota -h \$SCRATCH'"   # scratch usage
ssh cleps "bash -lc 'du -sh ~'"                 # home usage (100 GB cap)
```

## Environment (pixi, no modules)

CLEPS uses pixi directly — there is no `module load`. Jobs run via
`pixi run python …`; submitit captures the pixi env interpreter
(`~/tcfuse/.pixi/envs/default/bin/python`) and re-runs it on the compute node,
which sees it over the shared home filesystem. `.pixi/` is excluded from rsync
(`.rsyncignore`), so the env is created on the cluster by `setup_cleps.sh`.

## W&B online mode

CLEPS compute nodes have internet, so **W&B runs online** — no `WANDB_MODE`,
no login-node `wandb sync`. The grouped-segment run model still applies across
SLURM requeues (each launch is a distinct segment sharing `group=<run_id>`; see
[`.agents/jz.md`](jz.md) § W&B offline mode for the run/group convention), but
on CLEPS continuity is fully handled online — no manual sync is needed.

## One-time setup

Run once on the CLEPS login node after the first rsync (creates the pixi env):
```bash
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && bash scripts/setup_cleps.sh'"
```
Then authenticate W&B once (online mode):
```bash
ssh cleps "bash -lc 'cd ~/tcfuse && pixi run wandb login'"
```
Verify:
```bash
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run bash scripts/slurm/preflight_check_cleps.sh'"
```

## Checkpoint and resume

Same submitit mechanism as Jean-Zay: `TrainingTask` implements
`submitit.helpers.Checkpointable`; on the SLURM signal the job is requeued and
resume is automatic. To manually resume a run, relaunch with the existing
`run_id`:
```bash
ssh cleps "bash -lc 'source /etc/profile; cd ~/tcfuse && \
  pixi run python scripts/train/train.py paths=cleps setup=cleps_arches_x4 \
  experiment=<name> run_id=<existing id>'"
```

## Maintenance

When changing CLEPS conventions (storage layout, SLURM configs, environment,
rsync filters), update this skill in the same PR. If triggers or behavior rules
change, also update `.claude/commands/cleps.md`, `.cursor/skills/tcfuse-cleps/SKILL.md`,
and [`.agents/context.md`](context.md) (CLEPS quick reference table).
