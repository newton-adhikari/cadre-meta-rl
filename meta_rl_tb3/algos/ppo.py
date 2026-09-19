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


class RolloutBuffer:
    """Buffer for storing rollout data.
    
    Stores transitions from environment interactions for PPO updates.
    Handles advantage computation using GAE.
    
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: torch.device,
        num_envs: int = 1,
    ):
        """Initialize rollout buffer."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.num_envs = num_envs
        
        # Storage
        self.observations = np.zeros((buffer_size, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        # Computed quantities
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute returns and advantages using GAE."""
        last_gae = 0
        buffer_len = self.pos if not self.full else self.buffer_size
        
        for t in reversed(range(buffer_len)):
            if t == buffer_len - 1:
                # At the last collected step, bootstrap from last_value only if
                # that step did NOT end the episode.
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value * next_non_terminal
            else:
                # For every other step, mask the next value by whether the
                # current step terminated the episode.
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1] * next_non_terminal
            
            # TD error
            delta = (
                self.rewards[t] 
                + gamma * next_value
                - self.values[t]
            )
            
            # GAE — mask the carry-over advantage by the same done flag
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for training."""
        buffer_len = self.pos if not self.full else self.buffer_size
        
        # Flatten data across environments
        obs = self.observations[:buffer_len].reshape(-1, self.obs_dim)
        actions = self.actions[:buffer_len].reshape(-1, self.action_dim)
        values = self.values[:buffer_len].reshape(-1)
        log_probs = self.log_probs[:buffer_len].reshape(-1)
        advantages = self.advantages[:buffer_len].reshape(-1)
        returns = self.returns[:buffer_len].reshape(-1)
        
        total_size = len(obs)
        indices = np.arange(total_size)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        batches = []
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            batch = {
                "observations": torch.from_numpy(obs[batch_indices]).to(self.device),
                "actions": torch.from_numpy(actions[batch_indices]).to(self.device),
                "old_values": torch.from_numpy(values[batch_indices]).to(self.device),
                "old_log_probs": torch.from_numpy(log_probs[batch_indices]).to(self.device),
                "advantages": torch.from_numpy(advantages[batch_indices]).to(self.device),
                "returns": torch.from_numpy(returns[batch_indices]).to(self.device),
            }
            batches.append(batch)
        
        return batches

