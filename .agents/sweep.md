# TC-Fuse hyperparameter sweeps

Claude Code: invoke `/sweep` (reads this skill).

**Coding style:** follow [`.agents/context.md`](context.md) § Human-readable code (priority).

Read this before setting up or running a hyperparameter search, adding a new search space, or
changing how `scripts/train/train.py` reports its objective.

## When to use
- Search model or optimizer hyperparameters (e.g. the Perceiver's capacity knobs) over a budget of trials.
- Add a new search space (`conf/hydra/sweeper/*.yaml`) or a new sweep launcher.
- Change the metric a trial optimizes, or the short-budget sweep experiment.

## Approach — Hydra + Optuna over parallel SLURM trials

The project uses Hydra's **Optuna sweeper** with Hydra's **submitit launcher**. Rationale:

- **Optuna is offline-safe.** Its study is local (no server), so it works on Jean-Zay's offline
  compute nodes — unlike W&B Sweeps, whose server-side controller can't be reached there.
- **Parallelism comes from the launcher, not Optuna.** The sweeper proposes a batch of `n_jobs`
  trials; the **submitit launcher** submits them as a concurrent SLURM array
  (`array_parallelism`). The basic launcher would run a batch sequentially in-process.
- **`train.py` returns the objective.** With `submitit=false`, `scripts/train/train.py:main()`
  runs the trial in-process and returns the **best `val/loss`** (read from the val-loss-monitoring
  `ModelCheckpoint`'s `best_model_score`, with a fallback to the last logged `val/loss`); Optuna
  minimizes that. The inner submitit executor is skipped so Hydra's launcher owns SLURM submission.

## The pieces

| File | Role |
|---|---|
| `conf/hydra/sweeper/perceiver_capacity.yaml` | Optuna study: TPE sampler, `direction: minimize`, `n_trials`, `n_jobs`, and the `params` search space. |
| `conf/hydra/launcher/jz_4xv100_32g_sweep.yaml` | submitit-launcher SLURM spec for the default sweep — 4× V100 32 GB DDP per trial (`tasks_per_node: 4`), mirroring `conf/setup/jz_4xv100_32g.yaml`. |
| `conf/hydra/launcher/jz_sweep.yaml` / `jz_v100_sweep.yaml` | single-GPU launcher variants (H100 / V100), mirroring `conf/setup/jz_h100.yaml` / `jz_v100.yaml` — handy for cheap dry runs. |
| `conf/experiment/pmw_gmi_sweep.yaml` | Short-budget (`max_steps=6000`) variant of the training experiment; defaults to `setup=jz_4xv100_32g` (4 GPUs, DDP, `batch_size: 8`). |
| `scripts/train/train.py` | `TrainingTask.__call__` / `main()` return the best `val/loss` as the sweeper objective. |

**Match the launcher to the setup.** `trainer.devices` (from the `setup=` config) must equal the
launcher's `tasks_per_node` — the experiment pairs `setup=jz_4xv100_32g` (`devices=4`) with
`hydra/launcher=jz_4xv100_32g_sweep` (`tasks_per_node=4`). Lightning DDP then detects the 4 SLURM
tasks and runs one rank each; the launcher reads rank 0's returned objective. A mismatch makes DDP
hang waiting for ranks that never start.

## Running a sweep

From a node that can submit SLURM jobs (a login or `prepost` node — **not** a compute node),
inside `tmux`:

```bash
python scripts/train/train.py --multirun \
  experiment=pmw_gmi_sweep submitit=false \
  hydra/launcher=jz_4xv100_32g_sweep hydra/sweeper=perceiver_capacity
```

Cluster submission details (env, login-node rule, plugin install) → [`/jz`](jz.md)
§ Hyperparameter sweeps and [`/cleps`](cleps.md). On CLEPS, compute nodes have internet, so a W&B
Sweep is also viable — but Optuna keeps one workflow across both clusters.

**Dry run first.** Validate end-to-end cheaply on a single GPU before committing the full budget —
override the setup to a single-GPU one so `trainer.devices` matches the launcher:
```bash
... setup=jz_v100 hydra/launcher=jz_v100_sweep \
  hydra.sweeper.n_trials=2 hydra.launcher.array_parallelism=1 \
  trainer.max_steps=200 trainer.val_check_interval=100
```

## Defining the search space

Search-space entries live under `hydra.sweeper.params`, addressed at the same override paths used
on the CLI. The Perceiver is mounted at `lightning_module.model`, so:

```yaml
params:
  lightning_module.model.embed_dim: choice(128, 256, 384)
  lightning_module.model.latent_dim: choice(256, 512, 768)
  lightning_module.model.num_latents: choice(128, 256, 512)
  lightning_module.model.num_layers: range(4, 10)
```

**Respect the backbone's divisibility invariants** (`src/tcfuse/models/perceiver/backbone.py`):
`num_heads` must divide `latent_dim`; `cross_num_heads` must divide **both** `embed_dim` and
`latent_dim`. The default sweep keeps `num_heads = cross_num_heads = 8` fixed and restricts every
searched dim to a multiple of 8, so no trial can sample an invalid (crashing) config. If you sweep
the head counts too, constrain the space so divisibility always holds.

Keep `n_jobs` (sweeper) equal to `array_parallelism` (launcher) so each proposed batch maps to one
SLURM array wave.

## Reading results

- **Authoritative:** `multirun/<date>/<time>/optimization_results.yaml` — best params + best value,
  independent of W&B.
- **Per trial:** each trial logs as its own run named `<experiment.name>-<run_id>`; on JZ these are
  offline `wandb/offline-run-*` folders to `wandb sync`. Filter by name in the `tcfuse`/`arches`
  project to compare curves.

## Adding a new sweep
1. Copy `conf/hydra/sweeper/perceiver_capacity.yaml` to a new name; edit `params` (and `n_trials`/
   `direction`). Keep the search space valid w.r.t. any model invariants.
2. Reuse an existing `conf/hydra/launcher/jz_*_sweep.yaml`, or add one mirroring the target
   `conf/setup/jz_*` spec (translate the `slurm_*` keys to the submitit-launcher names; see the
   existing files for the mapping).
3. If the objective should be something other than the best `val/loss`, change what
   `TrainingTask.__call__` returns in `scripts/train/train.py`.

## Optional: pruning

Early-stopping unpromising trials needs an Optuna pruner plus a callback that reports intermediate
`val/loss` to `trial.report()` and checks `trial.should_prune()`. Not wired up — add only if the
extra complexity pays off for the budget.

## Maintenance

When the sweep configs, the objective contract in `scripts/train/train.py`, or the
launcher/sweeper conventions change, update this skill. If triggers or behavior change, also update
`.cursor/skills/tcfuse-sweep/SKILL.md`, `.claude/commands/sweep.md`, and the skills table in
[`.agents/context.md`](context.md), `CLAUDE.md`, and `AGENTS.md`.
