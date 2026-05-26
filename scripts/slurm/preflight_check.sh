#!/usr/bin/env bash
# Validate environment and data paths before submitting a SLURM job.
# Run from the project root on the JZ login node.
set -euo pipefail

ERRORS=0

# Check that the pytorch module is loaded (proxy for correct environment)
echo "[preflight] Checking pytorch-gpu module..."
if ! module list 2>&1 | grep -q "pytorch-gpu"; then
    echo "  WARNING: pytorch-gpu module not loaded. Run: module load pytorch-gpu/py3/2.8.0"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that tcfuse is importable (package installed via setup_jz.sh)
echo "[preflight] Checking tcfuse package..."
if ! python -c "import tcfuse" 2>/dev/null; then
    echo "  ERROR: 'import tcfuse' failed. Run: bash scripts/setup_jz.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that WANDB_MODE=offline is set (compute nodes have no internet)
echo "[preflight] Checking W&B offline mode..."
if [[ "${WANDB_MODE:-}" != "offline" ]]; then
    echo "  ERROR: WANDB_MODE must be 'offline'. Add to ~/.bash_profile: export WANDB_MODE=offline"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that $SCRATCH/tcfuse exists and is writeable
echo "[preflight] Checking \$SCRATCH/tcfuse..."
if [[ ! -d "${SCRATCH}/tcfuse" ]]; then
    echo "  WARNING: \$SCRATCH/tcfuse does not exist. Create it: mkdir -p \$SCRATCH/tcfuse"
    ERRORS=$((ERRORS + 1))
elif [[ ! -w "${SCRATCH}/tcfuse" ]]; then
    echo "  ERROR: \$SCRATCH/tcfuse is not writeable."
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that $WORK/tcfuse exists
echo "[preflight] Checking \$WORK/tcfuse..."
if [[ ! -d "${WORK}/tcfuse" ]]; then
    echo "  WARNING: \$WORK/tcfuse does not exist. Expected project root at \$WORK/tcfuse."
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check scratch quota — warn if >80% used
echo "[preflight] Checking scratch quota..."
QUOTA_OUT=$(idrquota 2>/dev/null || true)
if echo "$QUOTA_OUT" | grep -qi "scratch"; then
    # idrquota prints usage; look for any line flagged as over limit
    if echo "$QUOTA_OUT" | grep -qi "dépassé\|exceeded\|warning"; then
        echo "  WARNING: scratch quota may be exceeded. Run: idrquota -m"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK"
    fi
else
    echo "  (idrquota not available or no scratch entry — skipping)"
fi

# Summary
echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "[preflight] All checks passed. Ready to submit."
else
    echo "[preflight] $ERRORS check(s) failed. Fix issues above before submitting."
    exit 1
fi
