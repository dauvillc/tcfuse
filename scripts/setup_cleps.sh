#!/usr/bin/env bash
# One-time setup on the CLEPS login node (has internet access).
# Unlike Jean-Zay, CLEPS uses pixi directly (no `module load`) and W&B runs
# online (no offline sync). The pixi env is created under $HOME/tcfuse/.pixi,
# which is on the shared home filesystem, so it is visible from compute nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Resolve cluster env vars (e.g. $SCRATCH) on a non-interactive shell.
source /etc/profile 2>/dev/null || true

# Create the pixi environment from pixi.toml / pixi.lock (installs tcfuse editable).
echo "[setup] Installing pixi environment..."
pixi install

# Ensure the scratch tree exists (data, checkpoints, wandb, predictions live here).
echo "[setup] Ensuring \$SCRATCH/tcfuse exists..."
mkdir -p "${SCRATCH}/tcfuse"

# Verify the package imports inside the pixi env.
echo "[setup] Verifying tcfuse import..."
pixi run python -c "import tcfuse; print('tcfuse OK')"

echo "[setup] Done."
echo "[setup] If you have not already, authenticate W&B once: pixi run wandb login"
echo "[setup] Then run 'bash scripts/slurm/preflight_check_cleps.sh' before submitting jobs."
