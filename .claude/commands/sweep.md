# /sweep — TC-Fuse Hyperparameter Search Agent

Source of truth: [`.agents/sweep.md`](../../.agents/sweep.md).

This command activates the TC-Fuse hyperparameter-search skill. **Before running a sweep, adding a
`conf/hydra/sweeper/` search space or a `conf/hydra/launcher/` config, or changing the objective
returned by `scripts/train/train.py`**, read the skill file. The Hydra + Optuna approach, the
submitit-launcher parallelism, the search-space / divisibility rules, and the results layout are
defined there.

Cluster submission: [`/jz`](jz.md) (login-node rule, plugin install), [`/cleps`](cleps.md). Model
hyperparameter meaning: [`/architecture`](architecture.md).

Keep docs in sync: when the sweep configs, the `train.py` objective contract, or the
launcher/sweeper conventions change, update `.agents/sweep.md` and this file together; update the
skills table in `.agents/context.md`, `CLAUDE.md`, and `AGENTS.md`.

---

## Quick pointer

| Need | Start here (in sweep.md) |
|---|---|
| Run a sweep (command, dry run) | "Running a sweep" |
| Why Optuna + submitit launcher | "Approach" |
| Define / constrain the search space | "Defining the search space" |
| Where the best config lands | "Reading results" |
| Add a new search space or launcher | "Adding a new sweep" |
