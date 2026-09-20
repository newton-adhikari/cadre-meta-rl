#!/usr/bin/env python3
"""Adaptation-curve of experiment.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_rl_tb3.algos.cadre import CADRE, CADREConfig
from meta_rl_tb3.algos.ppo import PPO, PPOConfig
from meta_rl_tb3.algos.networks import ContextEncoderConfig, NetworkConfig
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

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

# Budgets in environment steps.
# With MAX_EP_STEPS=200: 0→0eps, 200→1ep, 400→2eps, 1000→5eps, 2000→10eps, 4000→20eps
# With MAX_EP_STEPS=100 (validation): 0→0eps, 100→1ep, 300→3eps, 500→5eps
BUDGET_SCHEDULE = [0, 200, 400, 1000, 2000, 4000]

N_EVAL_EPISODES_PER_TASK = 20
N_EVAL_TASKS              = 10        # held-out test tasks
MAX_EP_STEPS              = 200
# Reduced batch for tractable CPU wall time:
# 5 tasks × (3+3) eps × 200 steps = 3,600 steps/iter → ~12s/iter
# 1500 iters × 12s = 5h per method per seed; 5 seeds × 2 methods = 50h sequential.
# Run seeds in parallel (separate terminals/machines) to get 5-seed results in 10h.
META_BATCH_SIZE           = 5         # was 10
N_SUPPORT_EPISODES        = 3         # was 5
N_QUERY_EPISODES          = 3         # was 5
NUM_ITERS_DEFAULT         = 1500
SEEDS_DEFAULT             = [42, 7, 13, 99, 1337]
DYNAMICS_SPLIT            = "train"

# Goal-reaching only for the primary comparison (matches P1 smoke test task)
TASK_TYPES = ["goal_reaching"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P2 adaptation-curve experiment")
    p.add_argument("--methods", nargs="+", default=["cadre", "fomaml", "ppo_ft"],
                   choices=["cadre", "fomaml", "ppo_ft"])
    p.add_argument("--seeds",     type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--num-iters", type=int, default=NUM_ITERS_DEFAULT)
    p.add_argument("--validate",  action="store_true",
                   help="Short validation run: 300 iters, 2 seeds, budget=[0,100,500]")
    p.add_argument("--output-dir", type=str, default="results/p2")
    p.add_argument("--device",    type=str, default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment factory (perturbed dynamics on every call)
# ---------------------------------------------------------------------------

class DynGoalFactory:
    """Callable that builds a fresh perturbed goal-reaching env each time."""
    def __init__(self, task, dyn_split: str = DYNAMICS_SPLIT):
        self.task      = task
        self.dyn_split = dyn_split

    def __call__(self):
        self.task.config.dynamics = sample_dynamics(self.dyn_split)
        return self.task._default_env_fn()


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def _make_cadre_agent(encoder_type: str, seed: int, device: str) -> CADRE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS),
        seed=seed,
    )
    sample_task = dist.sample()
    cfg = CADREConfig(
        meta_lr=3e-4,
        inner_lr=0.01,    # reduced from 0.05 — 0.05 overshoots in one step on goal-reaching
        num_inner_steps=1,
        meta_batch_size=META_BATCH_SIZE,
        num_support_episodes=N_SUPPORT_EPISODES,
        num_query_episodes=N_QUERY_EPISODES,
        max_episode_steps=MAX_EP_STEPS,
        encoder_type=encoder_type,
        context_encoder_config=ContextEncoderConfig(
            obs_dim=CANONICAL_OBS_DIM,
            action_dim=2,
            context_dim=16,
            context_window=5,
            gru_hidden_dim=64,
            num_gru_layers=2,
        ),
        first_order=True,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        network_config=NetworkConfig(hidden_sizes=[256, 256], activation="tanh"),
        device=device,
        seed=seed,
    )
    return CADRE(DynGoalFactory(sample_task), cfg)


def train_cadre_agent(
    agent: CADRE,
    seed: int,
    num_iters: int,
    print_interval: int = 100,
    label: str = "CADRE",
    save_best_to: Optional[str] = None,
    eval_interval: int = 50,
    n_eval_tasks: int = 5,
) -> List[Dict]:

    
    dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS),
        seed=seed,
    )
    # Separate validation tasks (not the fixed OOD test set)
    val_tasks_raw = [dist.sample() for _ in range(n_eval_tasks)]
    val_tasks = [DynGoalFactory(t) for t in val_tasks_raw]

    training_metrics = []
    best_post_smooth = float("-inf")
    # Exponential moving average for stable best-checkpoint selection
    ema_post = None
    EMA_ALPHA = 0.05  # smoothing factor

    for iteration in range(1, num_iters + 1):
        tasks = [DynGoalFactory(dist.sample()) for _ in range(agent.config.meta_batch_size)]
        m = agent.meta_update(tasks)

        post = m["post_adaptation_reward"]
        ema_post = post if ema_post is None else EMA_ALPHA * post + (1 - EMA_ALPHA) * ema_post

        training_metrics.append({
            "iteration": iteration,
            "pre":       m["pre_adaptation_reward"],
            "post":      post,
            "gap":       m["adaptation_improvement"],
            "loss":      m["meta_loss"],
            "env_steps": m["total_env_steps"],
            "ema_post":  ema_post,
        })

        # Save best checkpoint by smoothed post-reward (val selection, not test)
        if save_best_to is not None and ema_post > best_post_smooth:
            best_post_smooth = ema_post
            agent.save(save_best_to)

        if iteration % print_interval == 0 or iteration == 1 or iteration == num_iters:
            print(
                f"  [{label} s={seed}] iter {iteration:4d}/{num_iters} | "
                f"pre={m['pre_adaptation_reward']:+.2f} "
                f"post={post:+.2f} "
                f"gap={m['adaptation_improvement']:+.2f} "
                f"ema_post={ema_post:+.2f}"
            )
    return training_metrics


def train_ppo_ft_agent(
    seed: int,
    num_iters: int,
    print_interval: int = 100,
    device: str = "auto",
) -> PPO:
    """Train multi-task PPO.  Returns the trained agent."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS),
        seed=seed,
    )

    # Simple env that re-samples task+dynamics on each reset
    from meta_rl_tb3.envs.goal_reaching import GoalReachingEnv, GoalReachingConfig, FlatGoalReachingEnv

    class MultiTaskEnv:
        """Wraps distribution to re-sample task+dynamics on each reset."""
        def __init__(self, dist, dyn_split):
            self.dist      = dist
            self.dyn_split = dyn_split
            self._env      = None
            self._build_env()
            self.observation_space = self._env.observation_space
            self.action_space      = self._env.action_space

        def _build_env(self):
            if self._env is not None:
                try: self._env.close()
                except Exception: pass
            task = self.dist.sample()
            task.config.dynamics = sample_dynamics(self.dyn_split)
            self._env = task._default_env_fn()

        def reset(self, **kw):
            self._build_env()
            return self._env.reset(**kw)

        def step(self, a):
            return self._env.step(a)

        def close(self):
            if self._env: self._env.close()

        @property
        def unwrapped(self):
            return self._env.unwrapped

    env = MultiTaskEnv(dist, DYNAMICS_SPLIT)
    cfg = PPOConfig(
        lr=3e-4,
        num_steps=2048,
        batch_size=64,
        num_epochs=10,
        network_config=NetworkConfig(hidden_sizes=[256, 256], activation="tanh"),
        device=device,
    )
    agent = PPO(env, cfg)

    total_updates = (num_iters * META_BATCH_SIZE * N_SUPPORT_EPISODES * MAX_EP_STEPS) // cfg.num_steps
    print(f"  [PPO-FT s={seed}] training {total_updates} updates "
          f"(≈ CADRE compute budget)...")

    for upd in range(1, total_updates + 1):
        m = agent.train_step()
        if upd % max(1, total_updates // 10) == 0:
            print(f"  [PPO-FT s={seed}] update {upd}/{total_updates} | "
                  f"reward={m['mean_reward']:+.2f}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _collect_adaptation_episode(
    env,
    policy_fn,
    max_steps: int = MAX_EP_STEPS,
) -> EpisodeResult:
    """Run one evaluation episode with the given policy callable.

    policy_fn(obs_np) → action_np
    """
    obs, info = env.reset()
    if not isinstance(obs, np.ndarray):
        obs = np.concatenate([v.flatten() for v in obs.values()])

    total_reward = 0.0
    any_collision = False
    n_steps = 0

    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, rew, term, trunc, info = env.step(action)
        if not isinstance(obs, np.ndarray):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        total_reward += rew
        n_steps += 1
        if info.get("collision", False):
            any_collision = True
        if term or trunc:
            break

    success     = bool(info.get("is_success", False))
    final_dist  = float(info.get("distance_to_goal", float("nan")))
    return EpisodeResult(
        success=success,
        collision=any_collision,
        n_steps=n_steps,
        final_dist=final_dist,
        total_reward=total_reward,
    )


def evaluate_cadre_at_budget(
    agent: CADRE,
    test_tasks: list,
    budget_steps: int,
    n_eval_per_task: int = N_EVAL_EPISODES_PER_TASK,
    method_label: str = "cadre",
    seed: int = 0,
    dyn_split: str = DYNAMICS_SPLIT,
) -> List[EpisodeResult]:
    """Evaluate CADRE/FOMAML at a fixed adaptation budget (environment steps)."""
    import torch
    from collections import OrderedDict, deque

    n_adapt = budget_steps // max(1, MAX_EP_STEPS)

    episodes: List[EpisodeResult] = []

    for task in test_tasks:
        task.config.dynamics = sample_dynamics(dyn_split)
        env = task._default_env_fn()
        try:
            # ──  clone meta-params ─────────────────────────────────
            params = OrderedDict(
                (name, param.clone())
                for name, param in agent.actor_critic.named_parameters()
            )
            enc_params = (dict(agent.encoder.named_parameters())
                          if agent.encoder is not None else {})

            # ──  adaptation (exactly n_adapt episodes) ─────────────
            K         = agent.config.context_encoder_config.context_window
            trans_buf = deque(maxlen=K)

            for _ in range(n_adapt):
                obs_raw, _ = env.reset()
                obs = agent._flatten_obs(obs_raw)
                rollout_obs, rollout_acts, rollout_rews = [], [], []
                rollout_dones, rollout_vals = [], []
                done = False; step = 0

                while not done and step < MAX_EP_STEPS:
                    with torch.no_grad():
                        z    = agent._encode_context(trans_buf, enc_params)
                        z_np = z.cpu().numpy().squeeze(0)
                        p_obs = (np.concatenate([obs, z_np])
                                 if agent.context_dim > 0 else obs)
                        obs_t  = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
                        act_t, _, _, val_t = agent._forward_policy(obs_t, params)
                    act_np = act_t.cpu().numpy().squeeze(0)
                    next_raw, rew, term, trunc, _ = env.step(act_np)
                    done    = bool(term or trunc)
                    next_obs = agent._flatten_obs(next_raw)
                    rollout_obs.append(obs.copy()); rollout_acts.append(act_np.copy())
                    rollout_rews.append(float(rew)); rollout_dones.append(1.0 if done else 0.0)
                    rollout_vals.append(val_t.cpu().item())
                    trans_buf.append(agent._build_transition(obs, act_np, rew, next_obs))
                    obs = next_obs; step += 1

                if agent.config.num_inner_steps > 0:
                    rollout = {
                        "obs_np":     np.array(rollout_obs,   dtype=np.float32),
                        "actions_np": np.array(rollout_acts,  dtype=np.float32),
                        "context_np": np.zeros((len(rollout_obs), agent.context_dim), dtype=np.float32),
                        "rewards":    np.array(rollout_rews,  dtype=np.float32),
                        "dones":      np.array(rollout_dones, dtype=np.float32),
                        "values":     np.array(rollout_vals,  dtype=np.float32),
                    }
                    params = agent._inner_update(params, rollout, create_graph=False)

            # ── evaluate n_eval_per_task fresh episodes ───────────
            def policy_fn(obs_np):
                with torch.no_grad():
                    z    = agent._encode_context(trans_buf, enc_params)
                    z_np = z.cpu().numpy().squeeze(0)
                    p_obs = (np.concatenate([obs_np, z_np])
                             if agent.context_dim > 0 else obs_np)
                    obs_t  = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
                    act_t, _, _, _ = agent._forward_policy(obs_t, params)
                return act_t.cpu().numpy().squeeze(0)

            for _ in range(n_eval_per_task):
                ep = _collect_adaptation_episode(env, policy_fn)
                ep.method         = method_label
                ep.seed           = seed
                ep.budget         = budget_steps
                ep.task_id        = task.task_id
                ep.dynamics_split = dyn_split
                episodes.append(ep)

        finally:
            env.close()

    return episodes


def evaluate_ppo_ft_at_budget(
    agent: PPO,
    test_tasks: list,
    budget_steps: int,
    n_eval_per_task: int = N_EVAL_EPISODES_PER_TASK,
    seed: int = 0,
    dyn_split: str = DYNAMICS_SPLIT,
) -> List[EpisodeResult]:
    """Evaluate PPO-FT at a fixed adaptation budget.

    Creates a copy of the pretrained weights, fine-tunes on budget_steps
    steps of experience from the test task, then evaluates.
    """
    import copy
    import torch

    episodes: List[EpisodeResult] = []

    for task in test_tasks:
        task.config.dynamics = sample_dynamics(dyn_split)
        env = task._default_env_fn()

        try:
            # Deep-copy so we always start from the pretrained init
            ft_agent = copy.deepcopy(agent)
            ft_agent.env = env
            # Reset the agent's internal obs to the new env's initial obs
            obs_init, _ = env.reset()
            if not isinstance(obs_init, np.ndarray):
                obs_init = np.concatenate([v.flatten() for v in obs_init.values()])
            ft_agent._last_obs = obs_init
            pre_train_steps = ft_agent.total_steps  # save baseline for delta tracking

            steps_done = 0
            while steps_done < budget_steps:
                old_steps = ft_agent.total_steps
                ft_agent.collect_rollouts()
                collected = ft_agent.total_steps - old_steps
                if collected == 0:
                    break
                ft_agent.update()
                steps_done += (ft_agent.total_steps - pre_train_steps) - steps_done
                if steps_done >= budget_steps:
                    break

            def policy_fn(obs_np):
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(ft_agent.device)
                    act   = ft_agent.actor_critic.get_action(obs_t, deterministic=True)
                return act.cpu().numpy().squeeze(0)

            for _ in range(n_eval_per_task):
                ep = _collect_adaptation_episode(env, policy_fn)
                ep.method         = "ppo_ft"
                ep.seed           = seed
                ep.budget         = budget_steps
                ep.task_id        = task.task_id
                ep.dynamics_split = dyn_split
                episodes.append(ep)

        finally:
            env.close()

    return episodes


# ---------------------------------------------------------------------------
# Plotting (Figure 4)
# ---------------------------------------------------------------------------

def plot_adaptation_curves(
    agg_by_method: Dict[str, Dict[str, Dict[str, float]]],
    budgets: List[int],
    save_path: Path,
    title: str = "Adaptation Curves: SR vs Interaction Budget",
) -> None:
    """Generate Figure 4: SR vs adaptation steps, one curve per method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping Figure 4")
        return

    COLORS = {"cadre": "#2196F3", "fomaml": "#FF9800", "ppo_ft": "#4CAF50"}
    LABELS = {"cadre": "CADRE (ours)", "fomaml": "FOMAML", "ppo_ft": "PPO-FT"}
    STYLES = {"cadre": "-", "fomaml": "--", "ppo_ft": "-."}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xscale("symlog", linthresh=10)

    for method, agg in agg_by_method.items():
        means, lo, hi = [], [], []
        for b in budgets:
            key = f"sr_at_{b}"
            if key in agg:
                means.append(agg[key]["mean"])
                lo.append(agg[key]["ci95_lo"])
                hi.append(agg[key]["ci95_hi"])
            else:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))

        x     = np.array(budgets, dtype=float)
        y     = np.array(means,   dtype=float)
        y_lo  = np.array(lo,      dtype=float)
        y_hi  = np.array(hi,      dtype=float)

        color  = COLORS.get(method, "black")
        label  = LABELS.get(method, method)
        style  = STYLES.get(method, "-")

        ax.plot(x, y, style, color=color, label=label, linewidth=2, marker="o",
                markersize=5)
        # Shade 95% CI (hide NaN gaps)
        valid = ~np.isnan(y_lo) & ~np.isnan(y_hi)
        if valid.any():
            ax.fill_between(x[valid], y_lo[valid], y_hi[valid],
                            color=color, alpha=0.15)

    ax.set_xlabel("Adaptation budget (environment steps)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate with number of seeds in one corner
    n_seeds = next(
        (v[f"sr_at_{budgets[0]}"]["n_seeds"]
         for v in agg_by_method.values()
         if f"sr_at_{budgets[0]}" in v), "?"
    )
    ax.text(0.02, 0.02, f"n_seeds={n_seeds}", transform=ax.transAxes,
            fontsize=9, color="gray", va="bottom")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Figure 4 saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.validate:
        args.num_iters = min(args.num_iters, 300)
        args.seeds     = args.seeds[:2]
        budget_schedule = [0, 100, 300, 500]   # with MAX_EP_STEPS=100: 0,1,3,5 eps
        # Reduce eval load for validation
        global N_EVAL_EPISODES_PER_TASK, N_EVAL_TASKS, META_BATCH_SIZE
        global N_SUPPORT_EPISODES, N_QUERY_EPISODES, MAX_EP_STEPS
        N_EVAL_EPISODES_PER_TASK = 5    # 5 per task instead of 20
        N_EVAL_TASKS              = 5    # 5 tasks instead of 10
        META_BATCH_SIZE           = 5    # 5 tasks per update instead of 10
        N_SUPPORT_EPISODES        = 3    # 3 episodes instead of 5
        N_QUERY_EPISODES          = 3
        MAX_EP_STEPS              = 100  # shorter episodes
        print("VALIDATION MODE: 2 seeds, 300 iters, MAX_EP_STEPS=100")
    else:
        budget_schedule = BUDGET_SCHEDULE

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P2: Adaptation-Curve Experiment")
    print(f"  Methods:  {args.methods}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Iters:    {args.num_iters}")
    print(f"  Budgets:  {budget_schedule}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    # Fixed held-out test tasks — same across all methods and seeds
    task_dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS),
        seed=0,
    )
    dyn_dist = DynamicsDistribution(split=DYNAMICS_SPLIT, seed=0)
    test_tasks = get_fixed_test_tasks(task_dist, dyn_dist,
                                       n=N_EVAL_TASKS, seed=0)
    print(f"  Fixed test tasks: {len(test_tasks)}")

    # per_method_seed_metrics[method][seed] = {metric_name: value}
    per_method_seed_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    # raw episodes for reproducibility
    raw_episodes: Dict[str, List[dict]] = defaultdict(list)

    # ── Train and evaluate each method × seed ────────────────────────────
    t_start = time.time()

    for seed in args.seeds:
        for method in args.methods:
            print(f"\n{'─'*70}")
            print(f"  Training {method.upper()}  seed={seed}  iters={args.num_iters}")
            print(f"{'─'*70}")

            episodes_by_budget: Dict[int, List[EpisodeResult]] = {}

            if method in ("cadre", "fomaml"):
                encoder_type = "gru" if method == "cadre" else "none"
                agent = _make_cadre_agent(encoder_type, seed, args.device)
                train_cadre_agent(agent, seed, args.num_iters,
                                  print_interval=max(1, args.num_iters // 15),
                                  label=method.upper())

                # Checkpoint
                ckpt_path = out_dir / f"{method}_seed{seed}.pt"
                agent.save(str(ckpt_path))

                print(f"\n  Evaluating {method.upper()} seed={seed} "
                      f"across {len(budget_schedule)} budgets ×"
                      f" {N_EVAL_TASKS} tasks × {N_EVAL_EPISODES_PER_TASK} eps ...")
                for budget in budget_schedule:
                    eps = evaluate_cadre_at_budget(
                        agent, test_tasks, budget,
                        n_eval_per_task=N_EVAL_EPISODES_PER_TASK,
                        method_label=method, seed=seed,
                    )
                    episodes_by_budget[budget] = eps
                    sr = float(np.mean([e.success for e in eps]))
                    cr = float(np.mean([e.collision for e in eps]))
                    print(f"    budget={budget:5d} | SR={sr:.3f} CR={cr:.3f} "
                          f"(n={len(eps)} eps)")

            elif method == "ppo_ft":
                ppo_agent = train_ppo_ft_agent(seed, args.num_iters,
                                                print_interval=50,
                                                device=args.device)
                ckpt_path = out_dir / f"ppo_ft_seed{seed}.pt"
                ppo_agent.save(str(ckpt_path))

                print(f"\n  Evaluating PPO-FT seed={seed} ...")
                for budget in budget_schedule:
                    eps = evaluate_ppo_ft_at_budget(
                        ppo_agent, test_tasks, budget,
                        n_eval_per_task=N_EVAL_EPISODES_PER_TASK,
                        seed=seed,
                    )
                    episodes_by_budget[budget] = eps
                    sr = float(np.mean([e.success for e in eps]))
                    cr = float(np.mean([e.collision for e in eps]))
                    print(f"    budget={budget:5d} | SR={sr:.3f} CR={cr:.3f}")

            # ── Store metrics for this seed ───────────────────────────────
            seed_metrics = compute_curve_metrics(episodes_by_budget)
            seed_metrics["seed"]   = float(seed)
            seed_metrics["method"] = 0.0   # string stored separately
            per_method_seed_metrics[method].append(seed_metrics)

            # Save per-seed raw episodes as JSON lines
            raw_path = out_dir / f"raw_{method}_seed{seed}.jsonl"
            with open(raw_path, "w") as f:
                for budget, eps_list in episodes_by_budget.items():
                    for ep in eps_list:
                        f.write(json.dumps({
                            "method": method, "seed": seed, "budget": budget,
                            "success": ep.success, "collision": ep.collision,
                            "n_steps": ep.n_steps, "final_dist": ep.final_dist,
                            "total_reward": ep.total_reward,
                            "task_id": ep.task_id,
                        }) + "\n")
            print(f"  Raw episodes saved → {raw_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal training+eval time: {elapsed/3600:.1f} h")

    # ── Aggregate across seeds and write summary ───────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    agg_by_method: Dict[str, Dict] = {}
    for method in args.methods:
        seed_metrics_list = per_method_seed_metrics[method]
        agg = aggregate_seeds(seed_metrics_list)
        agg_by_method[method] = agg
        show_budgets = budget_schedule[::max(1, len(budget_schedule)//4)]
        print(format_summary_row(method, agg, show_budgets))

    # Write summary CSV (one row per method, columns = metric stats)
    csv_path = out_dir / "summary.csv"
    all_metric_keys = sorted({
        k for agg in agg_by_method.values() for k in agg.keys()
    })
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"] + [f"{k}_mean" for k in all_metric_keys] + \
                              [f"{k}_std"  for k in all_metric_keys]
        writer.writerow(header)
        for method, agg in agg_by_method.items():
            row = [method]
            row += [agg.get(k, {}).get("mean", float("nan")) for k in all_metric_keys]
            row += [agg.get(k, {}).get("std",  float("nan")) for k in all_metric_keys]
            writer.writerow(row)
    print(f"\nSummary CSV → {csv_path}")

    # Save full aggregated stats as JSON
    agg_path = out_dir / "aggregated_metrics.json"
    # Convert any inf values to strings for JSON serialisation
    def _json_safe(obj):
        if isinstance(obj, float) and math.isinf(obj):
            return "inf"
        return obj

    import json as _json
    with open(agg_path, "w") as f:
        _json.dump(
            {m: {k: {sk: _json_safe(sv) for sk, sv in v.items()}
                 for k, v in agg.items()}
             for m, agg in agg_by_method.items()},
            f, indent=2
        )
    print(f"Aggregated metrics JSON → {agg_path}")

    # ── Figure 4 ───────────────────────────────────────────────────────────
    fig_path = out_dir / "fig4_adaptation_curves.pdf"
    label = "validation" if args.validate else "5-seed"
    plot_adaptation_curves(
        agg_by_method,
        budget_schedule,
        save_path=fig_path,
        title=f"P2 Adaptation Curves ({label}, goal_reaching+dynamics)",
    )

    # ── Kill criteria check ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Kill criteria check (Phase 2 design)")
    cadre_auc = agg_by_method.get("cadre", {}).get("auc_sr", {}).get("mean", float("nan"))
    fomaml_auc = agg_by_method.get("fomaml", {}).get("auc_sr", {}).get("mean", float("nan"))
    ppo_auc   = agg_by_method.get("ppo_ft", {}).get("auc_sr", {}).get("mean", float("nan"))

    print(f"  AUC-SR: CADRE={cadre_auc:.3f}  FOMAML={fomaml_auc:.3f}  PPO-FT={ppo_auc:.3f}")

    cadre_sr_5k = agg_by_method.get("cadre", {}).get(
        f"sr_at_{budget_schedule[-1]}", {}
    ).get("mean", float("nan"))
    fomaml_sr_5k = agg_by_method.get("fomaml", {}).get(
        f"sr_at_{budget_schedule[-1]}", {}
    ).get("mean", float("nan"))

    k1 = "K1" if not math.isnan(cadre_auc) and not math.isnan(fomaml_auc) and cadre_auc > fomaml_auc else "K1 not met"
    k2 = "K2" if not math.isnan(cadre_auc) and not math.isnan(ppo_auc)   and cadre_auc > ppo_auc   else "K2 not met"
    print(f"  [{k1}] CADRE AUC-SR > FOMAML AUC-SR")
    print(f"  [{k2}] CADRE AUC-SR > PPO-FT AUC-SR")
    print("─" * 70)

    print("\nP2 complete.")
    print(f"  Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
