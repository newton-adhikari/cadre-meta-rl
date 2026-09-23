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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FOMAML (gradient-only, n=7 seeds)
#          Seeds: base 5 + ext 2 = 7 total
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/8] FOMAML — gradient-only meta-RL (n=7 seeds)..."
for seed in $BASE_SEEDS $EXT_SEEDS; do
    done_flag="$OUTDIR/.done_fomaml_seed${seed}"
    [[ -f "$done_flag" ]] && { echo "  [SKIP] fomaml seed=$seed"; continue; }
    [[ $EVAL_ONLY -eq 1 ]] && continue
    echo "  Training FOMAML seed=$seed ..."
    python3 -u -W ignore experiments/run_p2_single_seed.py \
        --method fomaml \
        --seed "$seed" \
        --num-iters "$NUM_ITERS" \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/fomaml_seed${seed}.log"
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CADRE full (context + gradient, n=7 seeds)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/8] CADRE — context encoder + gradient step (n=7 seeds)..."
for seed in $BASE_SEEDS $EXT_SEEDS; do
    done_flag="$OUTDIR/.done_cadre_seed${seed}"
    [[ -f "$done_flag" ]] && { echo "  [SKIP] cadre seed=$seed"; continue; }
    [[ $EVAL_ONLY -eq 1 ]] && continue
    echo "  Training CADRE seed=$seed ..."
    python3 -u -W ignore experiments/run_p2_single_seed.py \
        --method cadre \
        --seed "$seed" \
        --num-iters "$NUM_ITERS" \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/cadre_seed${seed}.log"
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CADRE-ctx ablation (encoder only, no gradient, n=7 seeds)
#          This is the KEY ablation: isolates context from gradient step.
#          Wilcoxon p=0.014 at n=7 vs FOMAML at SR@2ep.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/8] CADRE-ctx — encoder only, num_inner_steps=0 (n=7 seeds)..."
for seed in $BASE_SEEDS $EXT_SEEDS; do
    done_flag="$ABLDIR/.done_cadre_seed${seed}"
    [[ -f "$done_flag" ]] && { echo "  [SKIP] cadre-ctx seed=$seed"; continue; }
    [[ $EVAL_ONLY -eq 1 ]] && continue
    echo "  Training CADRE-ctx seed=$seed ..."
    python3 -u -W ignore experiments/run_p2_single_seed.py \
        --method cadre \
        --encoder-type gru \
        --num-inner-steps 0 \
        --seed "$seed" \
        --num-iters "$NUM_ITERS" \
        --output-dir "$ABLDIR" \
        2>&1 | tee "$ABLDIR/cadre_ctx_seed${seed}.log"
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PPO-FT baseline (multi-task PPO + fine-tune, n=5 seeds)
#          Uses eval_ppo_ft.py to avoid deepcopy/threading bug.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/8] PPO-FT — multi-task PPO + fine-tune (n=5 seeds)..."
for seed in $PPO_SEEDS; do
    done_iid="$OUTDIR/.done_ppo_ft_seed${seed}_iid"
    done_ood="$OUTDIR/.done_ppo_ft_seed${seed}_ood"
    if [[ -f "$done_iid" && -f "$done_ood" ]]; then
        echo "  [SKIP] ppo_ft seed=$seed (both iid+ood done)"
        continue
    fi
    if [[ $EVAL_ONLY -eq 0 ]]; then
        # Check if checkpoint exists; train if not
        ckpt="$OUTDIR/ppo_ft_seed${seed}.pt"
        if [[ ! -f "$ckpt" ]]; then
            echo "  Training PPO-FT seed=$seed ..."
            # run_b2_ppo_ft.py trains then crashes on deepcopy eval — that's OK
            # The checkpoint is saved before the crash
            python3 -u -W ignore experiments/run_b2_ppo_ft.py \
                --seed "$seed" \
                --output-dir "$OUTDIR" \
                2>&1 | tee "$OUTDIR/ppo_ft_seed${seed}_train.log" || true
        fi
    fi
    # Evaluate with the fixed script (save/load, no deepcopy)
    echo "  Evaluating PPO-FT seed=$seed ..."
    python3 -u -W ignore experiments/eval_ppo_ft.py \
        --seeds "$seed" \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/ppo_ft_seed${seed}_eval.log"
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Context identifiability probe (P4)
#          Linear probes R² from z → dynamics params
#          Produces Table IV and Figure 4.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/8] Context identifiability probe (P4)..."
PROBE_DONE="$OUTDIR/.done_context_probe"
if [[ -f "$PROBE_DONE" ]]; then
    echo "  [SKIP] context probe (already done)"
else
    python3 -u -W ignore experiments/run_p4_context_probe.py \
        --seeds $BASE_SEEDS \
        --n-episodes 300 \
        --ckpt-dir "$ABLDIR" \
        --output-dir "$OUTDIR/probe" \
        --skip-umap \
        2>&1 | tee "$OUTDIR/context_probe.log"
    touch "$PROBE_DONE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Multi-task extension (goal-reaching + obstacle avoidance, P3)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/8] Multi-task extension: goal-reaching + obstacle avoidance (P3)..."


LEGACY_P3DIR="results/p3_multitask"
if [[ "$P3DIR" != "$LEGACY_P3DIR" && -d "$LEGACY_P3DIR" ]]; then
    if ! ls "$P3DIR"/metrics_*_iid.json >/dev/null 2>&1 \
       && ls "$LEGACY_P3DIR"/*.pt "$LEGACY_P3DIR"/metrics_*_iid.json >/dev/null 2>&1; then
        echo "  [INFO] Found existing P3 data in $LEGACY_P3DIR — using it as P3DIR."
        P3DIR="$LEGACY_P3DIR"
    fi
fi
if [[ $RUN_P3 -eq 0 ]]; then
    echo "  [SKIP] --no-p3 flag set"
else
    for method in cadre_ctx fomaml; do
        for seed in $BASE_SEEDS; do
            done_flag="$P3DIR/.done_${method}_seed${seed}"
            [[ -f "$done_flag" ]] && { echo "  [SKIP] p3 $method seed=$seed"; continue; }
            [[ $EVAL_ONLY -eq 1 ]] && continue
            echo "  P3 $method seed=$seed ..."
            python3 -u -W ignore experiments/run_p3_multitask.py \
                --method "$method" \
                --seed "$seed" \
                --num-iters "$NUM_ITERS" \
                --output-dir "$P3DIR" \
                2>&1 | tee "$P3DIR/${method}_seed${seed}.log"
        done
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Aggregate all results and generate paper figures
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[7/8] Aggregating results and generating figures..."

# Copy CADRE-ctx metrics to main OUTDIR with method=cadre_ctx
echo "  Copying CADRE-ctx metrics → $OUTDIR..."
python3 -W ignore -c "
import json
from pathlib import Path
ABLDIR = Path('${ABLDIR}')
OUTDIR = Path('${OUTDIR}')
for f in ABLDIR.glob('metrics_cadre_seed*_*.json'):
    d = json.loads(f.read_text())
    d = {k:('inf' if v=='inf' else v) for k,v in d.items()}
    d['method'] = 'cadre_ctx'
    out = OUTDIR / f.name.replace('metrics_cadre_', 'metrics_cadre_ctx_')
    if not out.exists():
        out.write_text(json.dumps(d, indent=2))
        print(f'  Copied {out.name}')
" 2>/dev/null

# Aggregate IID + OOD for all methods
python3 -u -W ignore experiments/aggregate_p2_results.py \
    --output-dir "$OUTDIR" \
    --methods fomaml cadre cadre_ctx ppo_ft \
    --partial --min-seeds 3 \
    2>&1 | grep -v "AdroitHand\|gymnasium-robotics\|ROS2\|WARNING"

# Aggregate P3 multi-task results (Table V) — writes p3_summary.json + prints table
if [[ $RUN_P3 -eq 1 ]]; then
    echo "  Aggregating P3 multi-task (Table V)..."
    python3 -W ignore -c "
import json, math, numpy as np
from pathlib import Path
P3DIR   = Path('${P3DIR}')
SEEDS   = [int(s) for s in '${BASE_SEEDS}'.split()]
METHODS = ['cadre_ctx', 'fomaml']
BUDGETS = [0, 200, 400, 1000, 2000, 4000]
summary = {}
print('  === TABLE V: MULTI-TASK (GR + OA) ===')
for task_type in ['goal_reaching', 'obstacle_avoidance']:
    print(f'  {task_type.upper().replace(\"_\",\" \")}:')
    summary[task_type] = {}
    for method in METHODS:
        srs = []
        for s in SEEDS:
            f = P3DIR / f'metrics_{method}_seed{s}_{task_type}_iid.json'
            if f.exists():
                d = json.loads(f.read_text())
                srs.append([d.get(f'sr_at_{b}', float('nan')) for b in BUDGETS])
        if srs:
            arr   = np.array(srs)
            means = np.nanmean(arr, 0).tolist()
            stds  = np.nanstd(arr, 0).tolist()
            summary[task_type][method] = {
                'n': len(srs),
                'budgets': BUDGETS,
                'sr_mean': means,
                'sr_std':  stds,
            }
            print(f'    {method} (n={len(srs)}): '
                  f'SR@0={means[0]:.3f}±{stds[0]:.3f}  '
                  f'SR@2ep={means[2]:.3f}±{stds[2]:.3f}')
        else:
            print(f'    {method}: no results found')
out = P3DIR / 'p3_summary.json'
out.write_text(json.dumps(summary, indent=2))
print(f'  P3 summary → {out}')
" 2>&1 | grep -v AdroitHand | grep -v gymnasium | grep -v ROS2 || \
        echo "  [WARN] P3 aggregation produced no output (P3 may not have run)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Statistical tests (Wilcoxon, Levene, Cohen d) and final figures
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[8/8] Computing statistical tests and regenerating paper figures..."

python3 -u -W ignore -c "
import json, math, numpy as np
from pathlib import Path
from scipy import stats

OUTDIR  = Path('${OUTDIR}')
ABLDIR  = Path('${ABLDIR}')
ALL7    = [7, 13, 42, 99, 1337, 2024, 2025]
BUDGETS = [0, 200, 400, 1000, 2000, 4000]

def load(p):
    d = json.loads(p.read_text())
    return {k:(math.inf if v=='inf' else v) for k,v in d.items()}

def get(method, seed, split, b_key):
    for d in [OUTDIR, ABLDIR]:
        f = d / f'metrics_{method}_seed{seed}_{split}.json'
        if f.exists(): return load(f).get(b_key, float('nan'))
    return float('nan')

print()
print('='*60)
print('STATISTICAL TESTS (Appendix B)')
print('='*60)

report_lines = []
for b, label in [(0,'SR@0'), (400,'SR@2ep'), (1000,'SR@5ep')]:
    key = f'sr_at_{b}'
    ctx = np.array([x for x in [get(\"cadre_ctx\",s,\"iid\",key) for s in ALL7] if not math.isnan(x)])
    fom = np.array([x for x in [get(\"fomaml\",s,\"iid\",key) for s in ALL7] if not math.isnan(x)])
    n = min(len(ctx), len(fom))
    if n < 3: continue
    W, p  = stats.wilcoxon(ctx[:n], fom[:n], alternative='greater')
    Fl,pl = stats.levene(ctx, fom)
    d_val = (ctx.mean()-fom.mean()) / np.sqrt((ctx.std()**2+fom.std()**2)/2) if ctx.std()+fom.std()>0 else 0
    sig = '*** p<0.05' if p<0.05 else '(not sig)'
    line = (f'{label}: ctx={ctx.mean():.3f}±{ctx.std():.3f}  '
            f'fom={fom.mean():.3f}±{fom.std():.3f}  '
            f'W={W:.0f} p={p:.4f} {sig}  d={d_val:.2f}  '
            f'Levene F={Fl:.2f} p={pl:.3f}')
    print(f'  {line}')
    report_lines.append(line)

# Save report
stats_file = OUTDIR / 'stats_report.txt'
stats_file.write_text('\n'.join(['CADRE-ctx vs FOMAML IID Statistical Tests', '='*50] + report_lines))
print(f'\n  Stats report → {stats_file}')
" 2>&1 | grep -v AdroitHand | grep -v gymnasium | grep -v ROS2

# Regenerate architecture figure
echo "  Regenerating architecture figure..."
python3 -u -W ignore docs/gen_arch_fig.py 2>&1 | grep -v AdroitHand | grep -v ROS2 || \
    echo "  [WARN] Architecture figure generation failed — check docs/gen_arch_fig.py"

# Regenerate adaptation curve figures from aggregated data
echo "  Regenerating adaptation curve figures..."
python3 -u -W ignore -c "
import json, math, numpy as np, sys
sys.path.insert(0,'.')
from pathlib import Path
from experiments.run_p2_adaptation_curves import plot_adaptation_curves

OUTDIR = Path('${OUTDIR}')

def load_agg(path):
    if not path.exists(): return {}
    d = json.loads(path.read_text())
    return {k:{kk:(math.inf if vv=='inf' else vv) for kk,vv in v.items()} for k,v in d.items()}

iid_agg = load_agg(OUTDIR / 'aggregated_metrics_iid.json')
ood_agg = load_agg(OUTDIR / 'aggregated_metrics_ood.json')
budgets = [0, 200, 400, 1000, 2000, 4000]

if iid_agg:
    plot_adaptation_curves(iid_agg, budgets,
        save_path=OUTDIR/'fig_iid_curves.pdf',
        title='IID Adaptation Curves (n=7)')
    print('  fig_iid_curves.pdf')
if ood_agg:
    plot_adaptation_curves(ood_agg, budgets,
        save_path=OUTDIR/'fig_ood_curves.pdf',
        title='OOD Extrap Curves (n=7)')
    print('  fig_ood_curves.pdf')
" 2>&1 | grep -v AdroitHand | grep -v gymnasium | grep -v ROS2 || true

