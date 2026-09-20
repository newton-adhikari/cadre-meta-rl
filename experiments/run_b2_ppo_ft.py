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
    pass


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
    pass


if __name__ == "__main__":
    main()
