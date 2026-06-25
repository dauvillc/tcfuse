#!/usr/bin/env bash
# Validate environment and data paths before submitting a SLURM job on CLEPS.
# Run from the project root on the CLEPS login node.
# Differs from the Jean-Zay preflight: pixi env instead of modules, and W&B is
# online (no WANDB_MODE=offline check).
set -euo pipefail

# Resolve cluster env vars (e.g. $SCRATCH) on a non-interactive shell.
source /etc/profile 2>/dev/null || true

ERRORS=0

# Check that the pixi environment exists.
echo "[preflight] Checking pixi environment..."
if [[ ! -d ".pixi" ]]; then
    echo "  ERROR: .pixi env missing. Run: bash scripts/setup_cleps.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that tcfuse is importable inside the pixi env.
echo "[preflight] Checking tcfuse package..."
if ! pixi run python -c "import tcfuse" 2>/dev/null; then
    echo "  ERROR: 'import tcfuse' failed. Run: bash scripts/setup_cleps.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK"
fi

# Check that $SCRATCH/tcfuse exists and is writeable.
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

# Check scratch quota (best-effort — warn only if it looks exceeded).
echo "[preflight] Checking scratch quota..."
QUOTA_OUT=$(lfs quota -h "${SCRATCH}" 2>/dev/null || true)
if [[ -n "$QUOTA_OUT" ]] && echo "$QUOTA_OUT" | grep -qi "exceeded\|over"; then
    echo "  WARNING: scratch quota may be exceeded — check with: lfs quota -h \$SCRATCH"
    ERRORS=$((ERRORS + 1))
else
    echo "  (quota check best-effort — skipping or OK)"
fi

# Summary
echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "[preflight] All checks passed. Ready to submit."
else
    echo "[preflight] $ERRORS check(s) failed. Fix issues above before submitting."
    exit 1
fi
