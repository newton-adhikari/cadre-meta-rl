"""Proximal Policy Optimization (PPO) implementation.

This module implements PPO with Generalized Advantage Estimation (GAE),
designed to work both as a standalone algorithm and as the inner-loop
optimizer for MAML.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from meta_rl_tb3.algos.networks import ActorCritic, NetworkConfig


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm.
    
    Attributes:
        # Learning rates
        lr: Learning rate for optimizer
        lr_schedule: Learning rate schedule ('constant', 'linear')
        
        # PPO hyperparameters
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_eps: PPO clipping epsilon
        clip_value: Whether to clip value function updates
        value_clip_eps: Clipping epsilon for value function
        
        # Loss weights
        value_loss_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        
        # Training parameters
        num_epochs: Number of optimization epochs per update
        batch_size: Mini-batch size
        normalize_advantages: Whether to normalize advantages
        
        # Rollout parameters
        num_steps: Number of steps per rollout
        num_envs: Number of parallel environments
        
        # Network configuration
        network_config: Configuration for actor-critic network
        
        # Misc
        target_kl: Target KL divergence for early stopping (None to disable)
        seed: Random seed
        device: Device to use ('cpu', 'cuda', 'auto')
    """
    # Learning rates
    lr: float = 3e-4
    lr_schedule: str = "constant"
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_value: bool = True
    value_clip_eps: float = 0.2
    
    # Loss weights
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 64
    normalize_advantages: bool = True
    
    # Rollout parameters
    num_steps: int = 2048
    num_envs: int = 1
    
    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Misc
    target_kl: Optional[float] = None
    seed: int = 42
    device: str = "auto"

