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


class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[PPOConfig] = None,
        actor_critic: Optional[ActorCritic] = None,
    ):
        """Initialize PPO."""
        self.env = env
        self.config = config or PPOConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Get dimensions
        self.obs_dim = self._get_obs_dim(env.observation_space)
        self.action_dim = env.action_space.shape[0]
        
        # Create or use provided actor-critic
        if actor_critic is not None:
            self.actor_critic = actor_critic.to(self.device)
        else:
            self.actor_critic = ActorCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.config.network_config.hidden_sizes,
                activation=self.config.network_config.activation,
            ).to(self.device)
        
        # Optimizer
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.config.lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            buffer_size=self.config.num_steps,
            device=self.device,
            num_envs=self.config.num_envs,
        )
        
        # Training state
        self.total_steps = 0
        self.num_updates = 0
        self._last_obs = None
        
        # Initialize environment
        self._reset_env()
    
    def _get_obs_dim(self, obs_space: gym.Space) -> int:
        """Get observation dimension from space."""
        if isinstance(obs_space, gym.spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, gym.spaces.Dict):
            total = 0
            for key, space in obs_space.spaces.items():
                total += int(np.prod(space.shape))
            return total
        else:
            raise ValueError(f"Unsupported observation space: {type(obs_space)}")
    
    def _flatten_obs(self, obs: Union[np.ndarray, Dict]) -> np.ndarray:
        """Flatten observation if necessary."""
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs.flatten()
    
    def _reset_env(self) -> None:
        """Reset environment and store initial observation."""
        obs, _ = self.env.reset()
        self._last_obs = self._flatten_obs(obs)
    
    def collect_rollouts(self) -> Dict[str, float]:
        """Collect rollouts from environment.
        
        Returns:
            Dictionary with rollout statistics
        """
        self.buffer.reset()
        self.actor_critic.eval()
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        with torch.no_grad():
            for step in range(self.config.num_steps):
                # Get action from policy
                obs_tensor = torch.from_numpy(self._last_obs).float().unsqueeze(0).to(self.device)
                action, log_prob, _, value = self.actor_critic(obs_tensor)
                
                action = action.cpu().numpy().squeeze(0)
                value = value.cpu().numpy().item()
                log_prob = log_prob.cpu().numpy().item()
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.buffer.add(
                    obs=self._last_obs.reshape(1, -1),
                    action=action.reshape(1, -1),
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                )
                
                # Update tracking
                current_episode_reward += reward
                current_episode_length += 1
                self.total_steps += 1
                
                # Handle episode end
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    current_episode_reward = 0
                    current_episode_length = 0
                    
                    obs, _ = self.env.reset()
                    self._last_obs = self._flatten_obs(obs)
                else:
                    self._last_obs = self._flatten_obs(next_obs)
            
            # Compute last value for GAE
            obs_tensor = torch.from_numpy(self._last_obs).float().unsqueeze(0).to(self.device)
            _, _, _, last_value = self.actor_critic(obs_tensor)
            last_value = last_value.cpu().numpy().reshape(1)
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update on collected rollouts."""
        self.actor_critic.train()
        
        # Get batches
        batches = self.buffer.get_batches(
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # Track metrics
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        all_clip_fractions = []
        
        for epoch in range(self.config.num_epochs):
            for batch in batches:
                # Normalize advantages
                advantages = batch["advantages"]
                if self.config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current policy outputs
                log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    batch["observations"], batch["actions"]
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_value:
                    value_pred_clipped = batch["old_values"] + torch.clamp(
                        values - batch["old_values"],
                        -self.config.value_clip_eps,
                        self.config.value_clip_eps,
                    )
                    value_loss1 = F.mse_loss(values, batch["returns"])
                    value_loss2 = F.mse_loss(value_pred_clipped, batch["returns"])
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, batch["returns"])
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_loss_coef * value_loss 
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    kl_div = (batch["old_log_probs"] - log_probs).mean().item()
                    clip_fraction = (torch.abs(ratio - 1) > self.config.clip_eps).float().mean().item()
                    
                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropy_losses.append(-entropy_loss.item())
                    all_kl_divs.append(kl_div)
                    all_clip_fractions.append(clip_fraction)
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                if np.mean(all_kl_divs[-len(batches):]) > self.config.target_kl:
                    break
        
        self.num_updates += 1
        
        return {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy": np.mean(all_entropy_losses),
            "kl_divergence": np.mean(all_kl_divs),
            "clip_fraction": np.mean(all_clip_fractions),
        }
    
    def train_step(self) -> Dict[str, float]:
        """Perform one complete training iteration."""
        rollout_metrics = self.collect_rollouts()
        update_metrics = self.update()
        
        return {**rollout_metrics, **update_metrics, "total_steps": self.total_steps}
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[callable] = None,
        log_interval: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the agent for specified timesteps."""
        history = {
            "mean_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        
        num_iterations = total_timesteps // self.config.num_steps
        
        for iteration in range(num_iterations):
            metrics = self.train_step()
            
            # Store history
            for key in history:
                if key in metrics:
                    history[key].append(metrics[key])
            
            # Callback
            if callback is not None:
                callback(metrics)
            
            # Logging
            if (iteration + 1) % log_interval == 0:
                print(
                    f"Iteration {iteration + 1}/{num_iterations} | "
                    f"Steps: {self.total_steps} | "
                    f"Reward: {metrics['mean_reward']:.2f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f}"
                )
        
        return history
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "num_updates": self.num_updates,
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.num_updates = checkpoint["num_updates"]
    
    def get_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """Get action for a single observation.
        
        Args:
            obs: Observation
            deterministic: Whether to use deterministic action
            
        Returns:
            Action array
        """
        self.actor_critic.eval()
        with torch.no_grad():
            obs = self._flatten_obs(obs)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actor_critic.get_action(obs_tensor, deterministic)
            return action.cpu().numpy().squeeze(0)
    
    def evaluate(
        self,
        env: Optional[gym.Env] = None,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the current policy."""
        env = env or self.env
        
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = self.get_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if "is_success" in info:
                successes.append(info["is_success"])
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }
        
        if successes:
            results["success_rate"] = np.mean(successes)
        
        return results
