#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

# ─── Defaults ─────────────────────────────────────────────────────────────────
# n=7 seeds: base 5 + 2 extra (needed for Wilcoxon significance at α=0.05)
BASE_SEEDS="7 13 42 99 1337"
EXT_SEEDS="2024 2025"           # FOMAML + CADRE-ctx only (not CADRE / PPO-FT)
ALL_SEEDS="7 13 42 99 1337 2024 2025"
PPO_SEEDS="7 13 42 99 1337"    # PPO-FT at n=5 (consistent with paper)
NUM_ITERS=1500
OUTDIR="results/paper"
ABLDIR="results/paper_ablations"
P3DIR="results/p3_multitask"
QUICK=0
EVAL_ONLY=0
RUN_P3=1

# ─── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=1
            BASE_SEEDS="42"; EXT_SEEDS=""; ALL_SEEDS="42"; PPO_SEEDS="42"
            NUM_ITERS=50
            ;;
        --seeds)      BASE_SEEDS="$2"; ALL_SEEDS="$2"; PPO_SEEDS="$2"; EXT_SEEDS=""; shift ;;
        --eval-only)  EVAL_ONLY=1 ;;
        --no-p3)      RUN_P3=0 ;;
        --outdir)     OUTDIR="$2"; shift ;;
        --abldir)     ABLDIR="$2"; shift ;;
        --p3dir)      P3DIR="$2"; shift ;;
        --iters)      NUM_ITERS="$2"; shift ;;
        -h|--help)
            sed -n '3,60p' "$0"; exit 0 ;;
        *)
            echo "Unknown flag: $1  (use --help for usage)"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUTDIR" "$ABLDIR" "$P3DIR"

echo "========================================================"
echo "  CADRE Full Reproduction"
echo "  Base seeds:   $BASE_SEEDS"
echo "  Ext  seeds:   ${EXT_SEEDS:-none}"
echo "  PPO  seeds:   $PPO_SEEDS"
echo "  Iters:        $NUM_ITERS"
echo "  Output:       $OUTDIR"
echo "  Ablations:    $ABLDIR"
echo "  P3 multitask: $P3DIR"
echo "  Quick:        $QUICK"
echo "  EvalOnly:     $EVAL_ONLY"
echo "  Run P3:       $RUN_P3"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Environment check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[0/8] Checking Python environment..."
python3 -W ignore -c "
import sys
missing = []
for pkg in ['torch','numpy','gymnasium','sklearn','matplotlib','scipy']:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f'MISSING: {missing}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
import torch, numpy, gymnasium, sklearn, matplotlib, scipy
print(f'  Python     {sys.version.split()[0]}')
print(f'  PyTorch    {torch.__version__}')
print(f'  NumPy      {numpy.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  gymnasium  {gymnasium.__version__}')
print(f'  scipy      {scipy.__version__}')
print('  All dependencies OK')
" || { echo "Environment check failed. Run: pip install -r requirements.txt"; exit 1; }
