#!/usr/bin/env python3
"""Eval-only script for PPO-FT baselines from existing checkpoints.

Using save/load instead of deepcopy to avoid the threading.Lock pickling error.
Using num_steps=MAX_EP_STEPS (200) for fine-grained budget control.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_rl_tb3.algos.ppo import PPO, PPOConfig
from meta_rl_tb3.algos.networks import NetworkConfig
from meta_rl_tb3.envs.physics import sample_dynamics
from meta_rl_tb3.tasks import (
    GoalReachingDistribution,
    GoalReachingDistributionConfig,
    DynamicsDistribution,
    get_fixed_test_tasks,
)
from meta_rl_tb3.evaluation.metrics import (
    EpisodeResult,
    compute_curve_metrics,
    aggregate_seeds,
    format_summary_row,
)

BUDGET_SCHEDULE = [0, 200, 400, 1000, 2000, 4000]
N_EVAL_EPISODES = 20
N_EVAL_TASKS    = 10
MAX_EP_STEPS    = 200
OUTDIR          = Path("results/p2_full")


def parse_args():
    pass


def collect_ep(env, pol_fn, max_steps=MAX_EP_STEPS):
    # mock data
    return EpisodeResult(
        success=False,
        collision=0, n_steps=0,
        final_dist=0,
        total_reward=0,
    )


def eval_seed(seed: int, splits: list, out_dir: Path):
    # Prefer the final checkpoint over the best (which was saved too early)
    final_ckpt = out_dir / f"ppo_ft_seed{seed}.pt"
    best_ckpt  = out_dir / f"ppo_ft_seed{seed}_best.pt"

    # Choose correct checkpoint: final if it has more steps
    ckpt = None
    for candidate in [final_ckpt, best_ckpt]:
        if candidate.exists():
            try:
                c = torch.load(candidate, weights_only=False)
                steps = c.get("total_steps", 0)
                if ckpt is None or steps > torch.load(ckpt, weights_only=False).get("total_steps", 0):
                    ckpt = candidate
                    ckpt_steps = steps
            except Exception:
                pass

    if ckpt is None:
        print(f"  [SKIP] No checkpoint found for ppo_ft seed={seed}")
        return

    print(f"\n  PPO-FT seed={seed}  checkpoint={ckpt.name} ({ckpt_steps:,} steps)")

    task_dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS), seed=0
    )

    # Snapshot to temp file once per seed
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy(str(ckpt), tmp_path)

    try:
        for dyn_split, label in [
            (s, "iid" if s == "train" else s.replace("ood_dyn_", "ood_"))
            for s in splits
        ]:
            done_flag = out_dir / f".done_ppo_ft_seed{seed}_{label}"
            if done_flag.exists():
                print(f"    [{label}] already done — skipping")
                continue

            dyn_dist   = DynamicsDistribution(split=dyn_split, seed=0)
            eval_tasks = get_fixed_test_tasks(task_dist, dyn_dist,
                                              n=N_EVAL_TASKS, seed=0)
            eps_by_budget = {}
            print(f"    [{label.upper()}]")

            for budget in BUDGET_SCHEDULE:
                t0 = time.time()
                eps = []
                for task in eval_tasks:
                    task.config.dynamics = sample_dynamics(dyn_split)
                    env = task._default_env_fn()
                    try:
                        # Load fresh agent per task — no deepcopy needed
                        ft = PPO(env, PPOConfig(
                            lr=3e-4,
                            num_steps=MAX_EP_STEPS,   # one episode per rollout
                            batch_size=min(64, MAX_EP_STEPS),
                            num_epochs=5,
                            network_config=NetworkConfig(
                                hidden_sizes=[256, 256], activation="tanh"
                            ),
                            device="cpu",
                        ))
                        ft.load(tmp_path)

                        # Re-point internal buffer to new env
                        obs_init, _ = env.reset()
                        if not isinstance(obs_init, np.ndarray):
                            obs_init = np.concatenate([
                                v.flatten() for v in obs_init.values()
                            ])
                        ft._last_obs = obs_init
                        baseline = ft.total_steps

                        # Fine-tune for exactly budget steps
                        steps_done = 0
                        while steps_done < budget:
                            ft.collect_rollouts()
                            ft.update()
                            steps_done = ft.total_steps - baseline
                            if steps_done >= budget:
                                break

                        def pol(obs_np, _ft=ft):
                            with torch.no_grad():
                                return _ft.actor_critic.get_action(
                                    torch.from_numpy(obs_np).float().unsqueeze(0),
                                    deterministic=True,
                                ).cpu().numpy().squeeze(0)

                        for _ in range(N_EVAL_EPISODES):
                            ep = collect_ep(env, pol)
                            ep.method = "ppo_ft"
                            ep.seed   = seed
                            ep.budget = budget
                            ep.task_id = task.task_id
                            ep.dynamics_split = dyn_split
                            eps.append(ep)
                    finally:
                        env.close()

                eps_by_budget[budget] = eps
                sr = float(np.mean([e.success for e in eps]))
                cr = float(np.mean([e.collision for e in eps]))
                n_adapt = budget // MAX_EP_STEPS
                print(f"      budget={budget:5d} ({n_adapt:2d} eps) | "
                      f"SR={sr:.3f}  CR={cr:.3f}  ({time.time()-t0:.0f}s)")

            # Save raw data
            raw = out_dir / f"raw_ppo_ft_seed{seed}_{label}.jsonl"
            with open(raw, "w") as f:
                for b, el in eps_by_budget.items():
                    for ep in el:
                        f.write(json.dumps({
                            "method": "ppo_ft", "seed": seed, "budget": b,
                            "split": label, "success": ep.success,
                            "collision": ep.collision, "n_steps": ep.n_steps,
                            "final_dist": ep.final_dist,
                            "total_reward": ep.total_reward,
                            "task_id": ep.task_id,
                        }) + "\n")

            sm = compute_curve_metrics(eps_by_budget)
            sm.update({"method": "ppo_ft", "seed": seed, "split": label})
            sp = out_dir / f"metrics_ppo_ft_seed{seed}_{label}.json"
            with open(sp, "w") as f:
                json.dump(
                    {k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
                     for k, v in sm.items()},
                    f, indent=2,
                )
            agg  = aggregate_seeds([sm])
            show = BUDGET_SCHEDULE[::max(1, len(BUDGET_SCHEDULE)//4)]
            print("    " + format_summary_row(f"ppo_ft[{label}]", agg, show))
            done_flag.touch()
    finally:
        os.unlink(tmp_path)


def main():
    pass


if __name__ == "__main__":
    main()
