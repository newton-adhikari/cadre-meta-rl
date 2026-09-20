#!/usr/bin/env python3
"""Run one seed of the P2 experiment.

Each invocation trains one method for one seed and runs the full
evaluation protocol.

"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Re-use everything from the main orchestrator
from experiments.run_p2_adaptation_curves import (
    BUDGET_SCHEDULE,
    N_EVAL_EPISODES_PER_TASK,
    N_EVAL_TASKS,
    MAX_EP_STEPS,
    DYNAMICS_SPLIT,
    DynGoalFactory,
    _make_cadre_agent,
    train_cadre_agent,
    train_ppo_ft_agent,
    evaluate_cadre_at_budget,
    evaluate_ppo_ft_at_budget,
    plot_adaptation_curves,
)
from meta_rl_tb3.tasks import (
    GoalReachingDistribution,
    GoalReachingDistributionConfig,
    DynamicsDistribution,
    get_fixed_test_tasks,
)
from meta_rl_tb3.evaluation.metrics import compute_curve_metrics, aggregate_seeds, format_summary_row


def parse_args():
    p = argparse.ArgumentParser(description="P2 single-seed run")
    p.add_argument("--method",    required=True, choices=["cadre", "fomaml", "ppo_ft"])
    p.add_argument("--seed",      type=int, required=True)
    p.add_argument("--num-iters", type=int, default=1500)
    p.add_argument("--output-dir", type=str, default="results/p2_full")
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--budget-schedule", type=int, nargs="+",
                   default=BUDGET_SCHEDULE,
                   help="Override budget schedule (default: %(default)s)")
    p.add_argument("--num-inner-steps", type=int, default=None,
                   help="Override num_inner_steps (default: from config). "
                        "Set to 0 for context-only ablation.")
    p.add_argument("--encoder-type", type=str, default=None,
                   choices=["gru", "mlp_avg", "none"],
                   help="Override encoder type. Set 'gru' + --num-inner-steps 0 for context-only.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing result — skip if already done
    raw_path = out_dir / f"raw_{args.method}_seed{args.seed}.jsonl"
    done_path = out_dir / f".done_{args.method}_seed{args.seed}"
    if done_path.exists():
        print(f"Already complete: {args.method} seed={args.seed}. Delete {done_path} to rerun.")
        return

    print(f"{'='*60}")
    print(f"P2 Single-seed: {args.method.upper()}  seed={args.seed}")
    print(f"  iters={args.num_iters}  budgets={args.budget_schedule}")
    print(f"  output → {out_dir}")
    print(f"{'='*60}")

    # Fixed held-out test tasks (same across all seeds/methods)
    task_dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS),
        seed=0,
    )
    dyn_dist  = DynamicsDistribution(split=DYNAMICS_SPLIT, seed=0)
    test_tasks = get_fixed_test_tasks(task_dist, dyn_dist, n=N_EVAL_TASKS, seed=0)
    print(f"  Fixed test tasks: {len(test_tasks)}")

    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────
    if args.method in ("cadre", "fomaml"):
        encoder_type = "gru" if args.method == "cadre" else "none"
        # Allow explicit encoder/inner-step overrides (for ablations)
        if args.encoder_type is not None:
            encoder_type = args.encoder_type
        agent = _make_cadre_agent(encoder_type, args.seed, args.device)
        if args.num_inner_steps is not None:
            agent.config.num_inner_steps = args.num_inner_steps
            print(f"  [override] num_inner_steps={args.num_inner_steps}")
        best_ckpt = out_dir / f"{args.method}_seed{args.seed}_best.pt"
        train_cadre_agent(
            agent, args.seed, args.num_iters,
            print_interval=max(1, args.num_iters // 15),
            label=args.method.upper(),
            save_best_to=str(best_ckpt),
        )
        # Save the final checkpoint too (for analysis / resuming)
        ckpt = out_dir / f"{args.method}_seed{args.seed}.pt"
        agent.save(str(ckpt))
        print(f"  Final checkpoint → {ckpt}")
        print(f"  Best checkpoint  → {best_ckpt}")

        # Evaluate from best checkpoint, not the final (possibly collapsed) one
        if best_ckpt.exists():
            agent.load(str(best_ckpt))
            print(f"  Loaded best checkpoint for evaluation")

    elif args.method == "ppo_ft":
        agent = train_ppo_ft_agent(
            args.seed, args.num_iters,
            print_interval=50,
            device=args.device,
        )
        ckpt = out_dir / f"ppo_ft_seed{args.seed}.pt"
        agent.save(str(ckpt))
        print(f"  Checkpoint saved → {ckpt}")

    train_time = time.time() - t0
    print(f"  Training time: {train_time/60:.1f} min")

    # ── Evaluate on IID (train-dynamics) and OOD test sets ────────────
    print(f"\n  Evaluating {len(args.budget_schedule)} budgets × "
          f"{N_EVAL_TASKS} tasks × {N_EVAL_EPISODES_PER_TASK} eps ...")

    for dyn_split, split_label in [("train", "iid"), ("ood_dyn_extrap", "ood")]:
        dyn_dist_eval = DynamicsDistribution(split=dyn_split, seed=0)
        eval_tasks    = get_fixed_test_tasks(task_dist, dyn_dist_eval,
                                              n=N_EVAL_TASKS, seed=0)
        episodes_by_budget = {}
        print(f"\n  [{split_label.upper()}] split={dyn_split}")

        for budget in args.budget_schedule:
            t_eval = time.time()
            if args.method in ("cadre", "fomaml"):
                eps = evaluate_cadre_at_budget(
                    agent, eval_tasks, budget,
                    n_eval_per_task=N_EVAL_EPISODES_PER_TASK,
                    method_label=args.method,
                    seed=args.seed,
                    dyn_split=dyn_split,
                )
            else:
                eps = evaluate_ppo_ft_at_budget(
                    agent, eval_tasks, budget,
                    n_eval_per_task=N_EVAL_EPISODES_PER_TASK,
                    seed=args.seed,
                    dyn_split=dyn_split,
                )
            episodes_by_budget[budget] = eps
            sr = float(np.mean([e.success for e in eps]))
            cr = float(np.mean([e.collision for e in eps]))
            n_adapt = budget // max(1, MAX_EP_STEPS)
            print(f"    budget={budget:5d} ({n_adapt:2d} adapt eps) | "
                  f"SR={sr:.3f}  CR={cr:.3f}  ({len(eps)} eval eps, "
                  f"{time.time()-t_eval:.0f}s)")

        # Save raw data per split
        raw_path = out_dir / f"raw_{args.method}_seed{args.seed}_{split_label}.jsonl"
        with open(raw_path, "w") as f:
            for budget, eps_list in episodes_by_budget.items():
                for ep in eps_list:
                    f.write(json.dumps({
                        "method": args.method, "seed": args.seed,
                        "budget": budget, "split": split_label,
                        "success": ep.success, "collision": ep.collision,
                        "n_steps": ep.n_steps, "final_dist": ep.final_dist,
                        "total_reward": ep.total_reward, "task_id": ep.task_id,
                    }) + "\n")
        print(f"  Raw data ({split_label}) → {raw_path}")

        # Per-split metrics
        seed_metrics = compute_curve_metrics(episodes_by_budget)
        seed_metrics["method"] = args.method
        seed_metrics["seed"]   = args.seed
        seed_metrics["split"]  = split_label
        summary_path = out_dir / f"metrics_{args.method}_seed{args.seed}_{split_label}.json"
        with open(summary_path, "w") as f:
            json.dump({k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
                       for k, v in seed_metrics.items()}, f, indent=2)
        print(f"  Metrics ({split_label}) → {summary_path}")

        show = args.budget_schedule[::max(1, len(args.budget_schedule)//4)]
        agg  = aggregate_seeds([seed_metrics])
        print("  " + format_summary_row(f"{args.method}[{split_label}]", agg, show))

    # Mark as done
    done_path.touch()
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  DONE: {args.method} seed={args.seed}")


if __name__ == "__main__":
    main()
