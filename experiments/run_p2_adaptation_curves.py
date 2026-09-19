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

def main():
    pass

if __name__ == "__main__":
    main()
