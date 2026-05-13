#!/usr/bin/env bash
# One-time setup on Jean-Zay login node (has internet access).
# Can also run on the prepost partition if the login node is busy:
#   srun --pty --partition=prepost --account=xyw@cpu --time=01:00:00 bash scripts/setup_jz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the JZ prebuilt PyTorch environment
module load pytorch-gpu/py3/2.8.0

# Install extra packages into ~/.local (available on compute nodes via $HOME/.local)
echo "[setup] Installing extra packages..."
pip install --user -r "$REPO_ROOT/requirements-jz.txt"

# Install tcfuse in editable mode so src/ changes are reflected immediately
echo "[setup] Installing tcfuse package..."
pip install --user -e "$REPO_ROOT"

# Ensure W&B runs offline on compute nodes (no internet)
if ! grep -q 'WANDB_MODE' "${HOME}/.bash_profile" 2>/dev/null; then
    echo 'export WANDB_MODE=offline' >> "${HOME}/.bash_profile"
    echo "[setup] Added WANDB_MODE=offline to ~/.bash_profile"
fi

echo "[setup] Done. Run 'bash scripts/slurm/preflight_check.sh' before submitting jobs."
