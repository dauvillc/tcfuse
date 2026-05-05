# Jean-Zay Cluster Operations

You are now in Jean-Zay mode. Apply the following context and rules for all cluster-related tasks in this session.

---

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints, virtual env | Persistent, backed up, slower I/O |
| `$SCRATCH` | Raw data, preprocessed tensors, DataLoader cache | Fast NVMe — **purged after 30 days** |
| `$STORE` | Long-term archival of final weights, processed datasets | Cold storage, not for training I/O |

**Rule:** DataLoaders always read from `$SCRATCH`. After preprocessing, always copy a backup to `$STORE`.

---

## Environment setup (login node only)

```bash
module purge
module load pytorch-gpu/py3/2.8.0
source $WORK/envs/tc_fusion/bin/activate
```

Run nodes have **no internet access**. All pip installs, W&B auth, and data downloads must happen on the login node before submitting jobs.

---

## W&B offline mode

Jobs must run with W&B offline. The `slurm_setup` commands in `conf/setup/jz_<hw>.yaml` set `WANDB_MODE=offline` automatically.

After a job completes, sync from the login node:
```bash
wandb sync $WORK/wandb/offline-run-*/
```

---

## Submitting jobs

Jobs are submitted from `scripts/train.py` (training) or `scripts/preprocess/<source>.py` (preprocessing) using `submitit.AutoExecutor`. SLURM parameters come from a `conf/setup/jz_<hw>.yaml` file.

**Training:**
```bash
python scripts/train.py paths=jz setup=jz_4xh100 experiment=<name>
```

**Preprocessing:**
```bash
python scripts/preprocess/tcprimed.py paths=jz setup=jz_4xh100
```

**Override individual SLURM params on the CLI:**
```bash
python scripts/train.py paths=jz setup=jz_4xh100 \
  setup.timeout_min=360 setup.slurm_gpus_per_node=4 experiment=<name>
```

**Local debug without submitit:**
```bash
python scripts/train.py experiment=<name> submitit=false
```

---

## `conf/setup/` convention

Each `jz_<hw>.yaml` file maps to a hardware target. Keys are passed verbatim to
`submitit.AutoExecutor.update_parameters()` and must be valid submitit `SlurmExecutor` kwargs.

Template for a new hardware config:
```yaml
slurm_partition: ???          # e.g. gpu_p13, gpu_p4, gpu_p5
slurm_nodes: 1
slurm_ntasks_per_node: 1
slurm_gpus_per_node: ???
slurm_cpus_per_task: 10
slurm_mem_gb: ???
timeout_min: ???
name: tc_fusion
slurm_account: ???            # idrproj account code
slurm_setup:
  - module purge
  - module load pytorch-gpu/py3/2.8.0
  - source ${oc.env:WORK}/envs/tc_fusion/bin/activate
  - export WANDB_MODE=offline
```

Common Jean-Zay partitions:

| Partition | Hardware | Max GPUs/node | Max time |
|---|---|---|---|
| `gpu_p13` | 4× V100 32 GB | 4 | 100 h |
| `gpu_p4` | 1× A100 80 GB | 1 | 100 h |
| `gpu_p5` | 8× H100 80 GB | 8 | 20 h |

---

## Preflight checklist

Before submitting any job, run:
```bash
bash scripts/slurm/preflight_check.sh
```

This verifies: venv exists, data paths resolve on `$SCRATCH`, W&B is in offline mode, `idrquota` is not exceeded.

Manual quota check:
```bash
idrquota -m          # $WORK and $SCRATCH usage
idrquota -s -m       # $STORE usage
```

---

## Checkpoint and resume

`TrainJob` implements `submitit.helpers.Checkpointable`. On SIGUSR1 (sent ~60 s before timeout), submitit calls `checkpoint()`, which requeues the job with `resume_run_id` set to the current run. Resume is automatic — no manual intervention needed.

To manually resume a run:
```bash
python scripts/train.py paths=jz setup=jz_<hw> experiment=<name> \
  resume_run_id=<run_id>
```
