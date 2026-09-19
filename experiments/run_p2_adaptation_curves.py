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

def main():
    pass

if __name__ == "__main__":
    main()
