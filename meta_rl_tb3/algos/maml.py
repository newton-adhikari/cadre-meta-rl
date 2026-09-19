"""Model-Agnostic Meta-Learning (MAML) for reinforcement learning.

This module implements MAML (Finn et al., 2017) for meta-learning navigation
policies that can quickly adapt to new tasks with few gradient steps.

Reference:
    Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
    of Deep Networks" (2017).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import multiprocessing as mp

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.func import functional_call

from meta_rl_tb3.algos.networks import ActorCritic, NetworkConfig
from meta_rl_tb3.algos.ppo import RolloutBuffer  # kept for API compatibility


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MAMLConfig:
    """Configuration for MAML.

    Attributes
    ----------
    meta_lr:
        Outer (meta) learning rate for Adam.
    inner_lr:
        Per-task inner-loop learning rate (step size α).
    num_inner_steps:
        Number of inner-loop gradient steps per task.
    meta_batch_size:
        Number of tasks per meta-update.
    num_trajectories_per_task:
        Episodes collected per task for the inner-loop support set and the
        outer-loop query set (same count for both).
    max_episode_steps:
        Maximum steps per episode during rollout collection.
    gamma:
        Discount factor γ.
    gae_lambda:
        GAE λ for advantage estimation.
    clip_eps:
        If > 0, use PPO-style clipping in the outer-loop loss.
        Set to 0 (default) for vanilla policy gradient, which is compatible
        with both FOMAML and second-order MAML outer loops.
    entropy_coef:
        Entropy bonus coefficient in the policy loss.
    first_order:
        If True, use first-order MAML (FOMAML): inner-loop gradients are
        detached, removing second-order Hessian terms.  The adapted
        parameters remain a function of the meta-parameters through the
        first-order (unrolled SGD) path.
        If False, use full second-order MAML: ``create_graph=True`` is
        passed to ``torch.autograd.grad`` in the inner loop.
    normalize_advantages:
        Normalize advantages to zero mean, unit variance within each task.
    max_grad_norm:
        Gradient clipping norm for the meta-optimizer step.
    network_config:
        Actor-critic architecture configuration.
    device:
        ``"auto"`` selects CUDA if available, else CPU.
    seed:
        Random seed.
    num_workers:
        Kept for API compatibility.  The gradient path always runs
        sequentially on the main process.  Workers > 0 have no effect
        on ``meta_update`` in this implementation.
    """
    # Meta-learning
    meta_lr:                  float        = 3e-4
    inner_lr:                 float        = 0.05
    num_inner_steps:          int          = 1
    meta_batch_size:          int          = 10

    # Rollout
    num_trajectories_per_task: int         = 10
    max_episode_steps:        int          = 200

    # Policy gradient
    gamma:                    float        = 0.99
    gae_lambda:               float        = 0.95
    clip_eps:                 float        = 0.0
    entropy_coef:             float        = 0.01

    # Training
    first_order:              bool         = False
    normalize_advantages:     bool         = True
    max_grad_norm:            float        = 0.5

    # Network
    network_config:           NetworkConfig = field(default_factory=NetworkConfig)

    # Misc
    device:                   str          = "auto"
    seed:                     int          = 42
    num_workers:              int          = 0   # kept for API compat; unused in gradient path

