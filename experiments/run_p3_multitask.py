#!/usr/bin/env python3
"""Multi-task experiment — goal-reaching + obstacle avoidance.

Extends the P2 single-task experiment to two task types

Both task types use the same 370-dim canonical observation and the
same dynamics perturbation distribution, so the single CADRE encoder
must infer both task structure AND dynamics conditions from transitions.

"""

from __future__ import annotations

import argparse
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

from meta_rl_tb3.algos.cadre import CADRE, CADREConfig
from meta_rl_tb3.algos.networks import ContextEncoderConfig, NetworkConfig
from meta_rl_tb3.envs.physics import sample_dynamics
from meta_rl_tb3.envs.wrappers import CANONICAL_OBS_DIM
from meta_rl_tb3.tasks import (
    GoalReachingDistribution, GoalReachingDistributionConfig,
    DynamicsDistribution, get_fixed_test_tasks,
    MixedTaskDistribution,
)
from meta_rl_tb3.tasks.distributions import ObstacleAvoidanceDistribution, ObstacleAvoidanceDistributionConfig
from meta_rl_tb3.evaluation.metrics import (
    EpisodeResult, compute_curve_metrics, aggregate_seeds, format_summary_row,
)

# ── Constants ─────────────────────────────────────────────────────────────
BUDGET_SCHEDULE  = [0, 200, 400, 1000, 2000, 4000]
N_EVAL_EPISODES  = 20
N_EVAL_TASKS     = 10         # per task type
MAX_EP_STEPS     = 200
DYNAMICS_SPLIT   = "train"
OUTPUT_DIR       = Path("results/p3_multitask")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",     required=True, choices=["cadre_ctx", "fomaml", "cadre"])
    p.add_argument("--seed",       type=int, required=True)
    p.add_argument("--num-iters",  type=int, default=1500)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


class MultiTaskDynFactory:
    """Samples from goal-reaching OR obstacle avoidance with fresh dynamics."""

    def __init__(self, task, dyn_split: str = DYNAMICS_SPLIT):
        self.task      = task
        self.dyn_split = dyn_split

    def __call__(self):
        self.task.config.dynamics = sample_dynamics(self.dyn_split)
        return self.task._default_env_fn()


def make_cadre_agent(encoder_type: str, num_inner_steps: int,
                     seed: int, device: str) -> CADRE:
    torch.manual_seed(seed); np.random.seed(seed)
    # Use goal-reaching for env_fn init (any 370-dim task works)
    dist = GoalReachingDistribution(
        GoalReachingDistributionConfig(max_episode_steps=MAX_EP_STEPS), seed=seed)
    sample = dist.sample()
    cfg = CADREConfig(
        meta_lr=3e-4, inner_lr=0.01,
        num_inner_steps=num_inner_steps,
        meta_batch_size=10,               # 5 GR + 5 OA per batch
        num_support_episodes=3,
        num_query_episodes=3,
        max_episode_steps=MAX_EP_STEPS,
        encoder_type=encoder_type,
        context_encoder_config=ContextEncoderConfig(
            obs_dim=CANONICAL_OBS_DIM, action_dim=2, context_dim=16,
            context_window=5, gru_hidden_dim=64, num_gru_layers=2,
        ),
        first_order=True,
        network_config=NetworkConfig(hidden_sizes=[256, 256], activation="tanh"),
        device=device, seed=seed,
    )
    return CADRE(MultiTaskDynFactory(sample), cfg)


def train_agent(agent: CADRE, mixed_dist: MixedTaskDistribution,
                seed: int, num_iters: int, label: str,
                save_best_to: str) -> list:
    """Train on mixed GR+OA distribution with best-checkpoint saving."""
    metrics = []
    best_ema = float("-inf")
    ema_post = None
    EMA = 0.05

    for i in range(1, num_iters + 1):
        # Balanced: 5 GR + 5 OA tasks per update
        tasks = [MultiTaskDynFactory(t) for t in
                 mixed_dist.sample_balanced_batch(5)]  # 5 per type = 10 total
        m = agent.meta_update(tasks)
        post = m["post_adaptation_reward"]
        ema_post = post if ema_post is None else EMA * post + (1 - EMA) * ema_post

        metrics.append({"iteration": i, "pre": m["pre_adaptation_reward"],
                        "post": post, "gap": m["adaptation_improvement"],
                        "ema_post": ema_post})

        if save_best_to and ema_post > best_ema:
            best_ema = ema_post
            agent.save(save_best_to)

        if i % max(1, num_iters // 15) == 0 or i == 1 or i == num_iters:
            print(f"  [{label} s={seed}] iter {i:4d}/{num_iters} | "
                  f"pre={m['pre_adaptation_reward']:+.2f} "
                  f"post={post:+.2f} gap={m['adaptation_improvement']:+.2f} "
                  f"ema={ema_post:+.2f}")
    return metrics


def evaluate_at_budget(agent: CADRE, eval_tasks: list,
                       budget: int, n_eval: int, label: str,
                       seed: int, dyn_split: str) -> list:
    """Evaluate at fixed adaptation budget (same protocol as P2)."""
    import torch
    from collections import OrderedDict, deque

    n_adapt = budget // max(1, MAX_EP_STEPS)
    episodes = []

    for task in eval_tasks:
        task.config.dynamics = sample_dynamics(dyn_split)
        env = task._default_env_fn()
        try:
            params = OrderedDict(
                (n, p.clone()) for n, p in agent.actor_critic.named_parameters()
            )
            enc_params = dict(agent.encoder.named_parameters()) \
                if agent.encoder is not None else {}

            K   = agent.config.context_encoder_config.context_window
            buf = deque(maxlen=K)

            for _ in range(n_adapt):
                obs_raw, _ = env.reset()
                obs = agent._flatten_obs(obs_raw)
                rollout_obs, rollout_acts, rollout_rews = [], [], []
                rollout_dones, rollout_vals = [], []
                done = False; step = 0
                while not done and step < MAX_EP_STEPS:
                    with torch.no_grad():
                        z    = agent._encode_context(buf, enc_params)
                        z_np = z.cpu().numpy().squeeze(0)
                        p_obs = np.concatenate([obs, z_np]) if agent.context_dim > 0 else obs
                        obs_t = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
                        act_t, _, _, val_t = agent._forward_policy(obs_t, params)
                    act_np = act_t.cpu().numpy().squeeze(0)
                    next_raw, rew, term, trunc, _ = env.step(act_np)
                    done = bool(term or trunc)
                    next_obs = agent._flatten_obs(next_raw)
                    rollout_obs.append(obs.copy())
                    rollout_acts.append(act_np.copy())
                    rollout_rews.append(float(rew))
                    rollout_dones.append(1.0 if done else 0.0)
                    rollout_vals.append(val_t.cpu().item())
                    buf.append(agent._build_transition(obs, act_np, rew, next_obs))
                    obs = next_obs; step += 1
                if agent.config.num_inner_steps > 0:
                    rollout = {
                        "obs_np": np.array(rollout_obs, dtype=np.float32),
                        "actions_np": np.array(rollout_acts, dtype=np.float32),
                        "context_np": np.zeros((len(rollout_obs), agent.context_dim), dtype=np.float32),
                        "rewards": np.array(rollout_rews, dtype=np.float32),
                        "dones":   np.array(rollout_dones, dtype=np.float32),
                        "values":  np.array(rollout_vals, dtype=np.float32),
                    }
                    params = agent._inner_update(params, rollout, create_graph=False)

            def pol(obs_np):
                with torch.no_grad():
                    z   = agent._encode_context(buf, enc_params)
                    z_np = z.cpu().numpy().squeeze(0)
                    p_obs = np.concatenate([obs_np, z_np]) if agent.context_dim > 0 else obs_np
                    obs_t = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
                    act_t, _, _, _ = agent._forward_policy(obs_t, params)
                return act_t.cpu().numpy().squeeze(0)

            for _ in range(n_eval):
                obs_raw, info = env.reset()
                obs = agent._flatten_obs(obs_raw)
                tr = 0.0; col = False; ns = 0
                done = False
                while not done and ns < MAX_EP_STEPS:
                    act = pol(obs)
                    obs_raw, rew, term, trunc, info = env.step(act)
                    obs = agent._flatten_obs(obs_raw)
                    tr += rew; ns += 1
                    if info.get("collision", False): col = True
                    if term or trunc: break
                ep = EpisodeResult(
                    success=bool(info.get("is_success", False)),
                    collision=col, n_steps=ns, final_dist=float(info.get("distance_to_goal", float("nan"))),
                    total_reward=tr, method=label, seed=seed,
                    budget=budget, task_id=task.task_id, dynamics_split=dyn_split,
                )
                episodes.append(ep)
        finally:
            env.close()
    return episodes

