"""CADRE: Context-Adaptive Differential-drive Robot Environment meta-RL.

CADRE combines two adaptation mechanisms:

    1. Context adaptation
       z_t = f_φ(τ_{t-K:t-1})
       Online context inference from the last K completed transitions.
       The encoder f_φ (a GRU) is trained end-to-end with the policy.

    2. Gradient adaptation
       θ' = θ - α ∇_θ L(D_support; z)
       One (or more) inner-loop FOMAML gradient steps applied to the
       support data collected with the current context.

The combination is the primary comparison point against:
  - FOMAML alone (no context encoder): CADRE with encoder_type="none"
  - Context-only (no gradient step):   CADRE with num_inner_steps=0
  - PPO fine-tuning baseline:          train_ppo_mt.py

Causal context update protocol
---------------------------------
At time step t:
  1. Compute z_t = encoder(buffer[-K:])   ← uses transitions up to t-1
  2. Select action a_t using π_θ(· | s_t, z_t)
  3. Execute a_t, observe r_t, s_{t+1}
  4. Append (s_t, a_t, r_t, s_{t+1}) to buffer
  5. z_{t+1} = encoder(buffer[-(K):])     ← will use t-th transition

The encoder never sees the transition that resulted from the action it
conditioned — the update is strictly causal.

Meta-gradient correctness
--------------------------
Same in-process pattern as the corrected MAML:
  - ``param.clone()`` retains graph connection to meta-parameters.
  - Rollouts collected under ``torch.no_grad()``.
  - Loss re-evaluated with gradients enabled on the stored (obs, action) pairs.
  - FOMAML: inner gradient detached (``grad.detach()``), not the params.
  - The encoder parameters φ are part of the meta-update graph.

"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.func import functional_call

from meta_rl_tb3.algos.networks import (
    ActorCritic,
    ContextEncoder,
    ContextEncoderConfig,
    NetworkConfig,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CADREConfig:
    """Configuration for CADRE.

    Parameters
    ----------
    meta_lr:
        Outer (meta) learning rate for Adam (applied to both θ and φ).
    inner_lr:
        Inner-loop gradient step size α.
    num_inner_steps:
        Number of inner-loop gradient steps.  Set to 0 for context-only
        (no gradient adaptation) ablation.
    meta_batch_size:
        Number of tasks per meta-update.
    num_support_episodes:
        Episodes collected for the inner-loop support set per task.
    num_query_episodes:
        Episodes collected for the outer-loop query set per task.
    max_episode_steps:
        Maximum steps per episode during rollout collection.
    encoder_type:
        "gru"     — full CADRE with GRU context encoder (proposed).
        "mlp_avg" — stateless context: mean-pool encoded transitions
                    (ablation: loses temporal structure).
        "none"    — no encoder; policy is conditioned on obs only
                    (equivalent to corrected FOMAML, primary baseline).
    context_encoder_config:
        ContextEncoderConfig.  Used only when encoder_type != "none".
    first_order:
        True  → FOMAML (default, faster).
        False → full second-order MAML.
    gamma / gae_lambda:
        Discount and GAE parameters.
    entropy_coef:
        Entropy bonus coefficient.
    normalize_advantages:
        Normalize advantages to zero mean, unit variance per task.
    max_grad_norm:
        Gradient clipping norm for the meta-optimizer.
    network_config:
        Actor-critic hidden sizes and activation.
    device:
        "auto" → CUDA if available, else CPU.
    seed:
        Random seed.
    """
    # Meta-learning
    meta_lr:               float = 3e-4
    inner_lr:              float = 0.05
    num_inner_steps:       int   = 1
    meta_batch_size:       int   = 10

    # Rollout
    num_support_episodes:  int   = 5
    num_query_episodes:    int   = 5
    max_episode_steps:     int   = 200

    # Context encoder
    encoder_type:          str   = "gru"    # "gru" | "mlp_avg" | "none"
    context_encoder_config: ContextEncoderConfig = field(
        default_factory=ContextEncoderConfig
    )

    # Policy gradient
    gamma:                 float = 0.99
    gae_lambda:            float = 0.95
    entropy_coef:          float = 0.01

    # Training
    first_order:           bool  = True
    normalize_advantages:  bool  = True
    max_grad_norm:         float = 0.5

    # Network
    network_config:        NetworkConfig = field(default_factory=NetworkConfig)

    # Misc
    device:                str   = "auto"
    seed:                  int   = 42

