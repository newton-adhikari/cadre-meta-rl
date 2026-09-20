#!/usr/bin/env python3
"""PPO multi-task pretraining + fine-tuning baseline.

Protocol:
  1. Train multi-task PPO for an equivalent compute budget to CADRE/FOMAML
     (1500 meta-iters × 5 tasks × 6 eps × 200 steps = 9,000,000 env steps).
  2. At evaluation, fine-tune the pretrained checkpoint on each test task
     using PPO updates consuming exactly the same number of env steps
     as the adaptation budgets used by CADRE/FOMAML.
  3. Evaluate with the fine-tuned policy.

"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_rl_tb3.algos.ppo import PPO, PPOConfig
from meta_rl_tb3.algos.networks import NetworkConfig
from meta_rl_tb3.envs.physics import sample_dynamics
from meta_rl_tb3.envs.wrappers import CANONICAL_OBS_DIM
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

# CADRE: 1500 iters × 5 tasks × (3 support + 3 query) × 200 steps = 9,000,000
TOTAL_STEPS    = 9_000_000
MAX_EP_STEPS   = 200
BUDGET_SCHEDULE = [0, 200, 400, 1000, 2000, 4000]
N_EVAL_TASKS   = 10
N_EVAL_EPISODES = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int,  required=True)
    p.add_argument("--output-dir",  type=str,  default="results/p2_full")
    p.add_argument("--device",      type=str,  default="auto")
    p.add_argument("--num-steps",   type=int,  default=2048,
                   help="PPO rollout length (steps per update)")
    return p.parse_args()


class MultiTaskGoalEnv:
    """Re-samples task+dynamics on each reset — simulates a multi-task env."""

    def __init__(self, dist, dyn_split: str = "train", seed: int = 0):
        self.dist      = dist
        self.dyn_split = dyn_split
        self._rng      = np.random.RandomState(seed)
        self._env      = None
        self._rebuild()
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def _rebuild(self):
        if self._env is not None:
            try: self._env.close()
            except Exception: pass
        task = self.dist.sample()
        task.config.dynamics = sample_dynamics("train")
        self._env = task._default_env_fn()

    def reset(self, **kw):
        self._rebuild()
        return self._env.reset(**kw)

    def step(self, a):
        return self._env.step(a)

    def close(self):
        if self._env: self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped


def _collect_eval_episode(env, policy_fn, max_steps: int = MAX_EP_STEPS) -> EpisodeResult:
    obs, info = env.reset()
    if not isinstance(obs, np.ndarray):
        obs = np.concatenate([v.flatten() for v in obs.values()])
    total_rew = 0.0; any_col = False; n_steps = 0
    for _ in range(max_steps):
        act = policy_fn(obs)
        obs, rew, term, trunc, info = env.step(act)
        if not isinstance(obs, np.ndarray):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        total_rew += rew; n_steps += 1
        if info.get("collision", False): any_col = True
        if term or trunc: break
    return EpisodeResult(
        success=bool(info.get("is_success", False)),
        collision=any_col, n_steps=n_steps,
        final_dist=float(info.get("distance_to_goal", float("nan"))),
        total_reward=total_rew,
    )


def evaluate_ppo_ft_at_budget(
    pretrained_agent: PPO,
    test_tasks: list,
    budget_steps: int,
    n_eval: int = N_EVAL_EPISODES,
    dyn_split: str = "train",
    seed: int = 0,
) -> list:
    """Fine-tune from pretrained init and evaluate.

    Uses save/load rather than deepcopy — deepcopy fails on threading.Lock
    objects held inside the MultiTaskGoalEnv wrapper.
    """
    import tempfile, os
    episodes = []

    # Snapshot pretrained weights to a temp file; reload per task
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    pretrained_agent.save(tmp_path)

    try:
        for task in test_tasks:
            task.config.dynamics = sample_dynamics(dyn_split)
            env = task._default_env_fn()
            try:
                # Fresh PPO agent with same config, loaded from pretrained weights
                # Use num_steps=MAX_EP_STEPS so one collect_rollouts = one episode,
                # giving fine-grained budget control.
                device_str = (pretrained_agent.device.type
                              if hasattr(pretrained_agent.device, 'type')
                              else str(pretrained_agent.device))
                ft_cfg = PPOConfig(
                    lr=3e-4,
                    num_steps=MAX_EP_STEPS,   # one episode per rollout = fine-grained budget
                    batch_size=min(64, MAX_EP_STEPS),
                    num_epochs=5,             # fewer epochs for fine-tuning stability
                    network_config=NetworkConfig(hidden_sizes=[256,256], activation="tanh"),
                    device=device_str,
                )
                ft = PPO(env, ft_cfg)
                ft.load(tmp_path)

                # Sync the agent's internal obs buffer to the new env
                obs_init, _ = env.reset()
                if not isinstance(obs_init, np.ndarray):
                    obs_init = np.concatenate([v.flatten() for v in obs_init.values()])
                ft._last_obs = obs_init
                baseline_steps = ft.total_steps

                steps_done = 0
                while steps_done < budget_steps:
                    ft.collect_rollouts()
                    ft.update()
                    steps_done = ft.total_steps - baseline_steps
                    if steps_done >= budget_steps:
                        break

                def policy_fn(obs_np, _ft=ft):
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(_ft.device)
                        return _ft.actor_critic.get_action(obs_t, deterministic=True).cpu().numpy().squeeze(0)

                for _ in range(n_eval):
                    ep = _collect_eval_episode(env, policy_fn)
                    ep.method = "ppo_ft"; ep.seed = seed
                    ep.budget = budget_steps; ep.task_id = task.task_id
                    ep.dynamics_split = dyn_split
                    episodes.append(ep)
            finally:
                env.close()
    finally:
        os.unlink(tmp_path)

    return episodes


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_iid = out_dir / f".done_ppo_ft_seed{args.seed}_iid"
    done_ood = out_dir / f".done_ppo_ft_seed{args.seed}_ood"
    if done_iid.exists() and done_ood.exists():
        print(f"Already complete: ppo_ft seed={args.seed}")
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS), seed=args.seed
    )

    print(f"{'='*60}")
    print(f"B2: PPO-FT  seed={args.seed}  budget={TOTAL_STEPS:,} steps")
    print(f"{'='*60}")

    env = MultiTaskGoalEnv(dist, seed=args.seed)
    cfg = PPOConfig(
        lr=3e-4, num_steps=args.num_steps, batch_size=64, num_epochs=10,
        network_config=NetworkConfig(hidden_sizes=[256,256], activation="tanh"),
        device=args.device,
    )
    agent = PPO(env, cfg)

    # Train for equivalent compute budget
    n_updates = TOTAL_STEPS // args.num_steps
    print(f"Training {n_updates} updates ({TOTAL_STEPS:,} steps)...")
    t0 = time.time()
    best_reward = float("-inf")
    best_ckpt   = out_dir / f"ppo_ft_seed{args.seed}_best.pt"

    for upd in range(1, n_updates + 1):
        m = agent.train_step()
        if m.get("mean_reward", float("-inf")) > best_reward:
            best_reward = m["mean_reward"]
            agent.save(str(best_ckpt))
        if upd % max(1, n_updates // 10) == 0:
            print(f"  update {upd:5d}/{n_updates} | reward={m['mean_reward']:+.2f} | best={best_reward:+.2f}")

    train_time = time.time() - t0
    agent.save(str(out_dir / f"ppo_ft_seed{args.seed}.pt"))
    print(f"Training time: {train_time/60:.1f} min")

    # Load best checkpoint for eval
    agent.load(str(best_ckpt))
    print("Loaded best checkpoint for evaluation.")
    env.close()

    # Fixed test tasks
    task_dist_eval = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS), seed=0
    )

    for dyn_split, label in [("train", "iid"), ("ood_dyn_extrap", "ood")]:
        done_flag = out_dir / f".done_ppo_ft_seed{args.seed}_{label}"
        if done_flag.exists():
            print(f"  [{label}] already done")
            continue

        dyn_dist_eval = DynamicsDistribution(split=dyn_split, seed=0)
        eval_tasks    = get_fixed_test_tasks(task_dist_eval, dyn_dist_eval,
                                              n=N_EVAL_TASKS, seed=0)
        eps_by_budget = {}
        print(f"\n  [{label.upper()}]")

        for budget in BUDGET_SCHEDULE:
            t_eval = time.time()
            eps = evaluate_ppo_ft_at_budget(
                agent, eval_tasks, budget,
                n_eval=N_EVAL_EPISODES, dyn_split=dyn_split, seed=args.seed,
            )
            eps_by_budget[budget] = eps
            sr = np.mean([e.success for e in eps])
            cr = np.mean([e.collision for e in eps])
            n_adapt = budget // max(1, MAX_EP_STEPS)
            print(f"    budget={budget:5d} ({n_adapt:2d} eps) | SR={sr:.3f}  CR={cr:.3f}  ({time.time()-t_eval:.0f}s)")

        # Save raw data
        raw_path = out_dir / f"raw_ppo_ft_seed{args.seed}_{label}.jsonl"
        with open(raw_path, "w") as f:
            for b, eps_list in eps_by_budget.items():
                for ep in eps_list:
                    f.write(json.dumps({
                        "method": "ppo_ft", "seed": args.seed, "budget": b,
                        "split": label, "success": ep.success, "collision": ep.collision,
                        "n_steps": ep.n_steps, "final_dist": ep.final_dist,
                        "total_reward": ep.total_reward, "task_id": ep.task_id,
                    }) + "\n")

        seed_metrics = compute_curve_metrics(eps_by_budget)
        seed_metrics.update({"method": "ppo_ft", "seed": args.seed, "split": label,
                              "train_time_min": round(train_time / 60, 1)})
        summary_path = out_dir / f"metrics_ppo_ft_seed{args.seed}_{label}.json"
        with open(summary_path, "w") as f:
            json.dump({k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
                       for k, v in seed_metrics.items()}, f, indent=2)

        agg  = aggregate_seeds([seed_metrics])
        show = BUDGET_SCHEDULE[::max(1, len(BUDGET_SCHEDULE)//4)]
        print("  " + format_summary_row(f"ppo_ft[{label}]", agg, show))
        done_flag.touch()

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"DONE: ppo_ft seed={args.seed}")


if __name__ == "__main__":
    main()
