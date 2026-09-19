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



class MAML:
    """Model-Agnostic Meta-Learning for RL.

    Learns an initial policy that can rapidly adapt to new tasks with a
    small number of gradient steps at test time.

    The outer (meta) update is computed correctly in-process: rollout data
    is collected under ``torch.no_grad()``, then actions are *re-evaluated*
    with the autograd graph intact so that gradients propagate back to
    ``self.actor_critic.parameters()``.

    """

    def __init__(
        self,
        env_fn:       Callable[[], gym.Env],
        config:       Optional[MAMLConfig] = None,
        actor_critic: Optional[ActorCritic] = None,
    ):
        self.env_fn = env_fn
        self.config = config or MAMLConfig()

        # Device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Infer dims from a sample environment
        _env = env_fn()
        self.obs_dim    = self._get_obs_dim(_env.observation_space)
        self.action_dim = _env.action_space.shape[0]
        _env.close()

        # Actor-critic
        if actor_critic is not None:
            self.actor_critic = actor_critic.to(self.device)
        else:
            self.actor_critic = ActorCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.config.network_config.hidden_sizes,
                activation=self.config.network_config.activation,
            ).to(self.device)

        # Meta-optimizer
        self.meta_optimizer = Adam(
            self.actor_critic.parameters(), lr=self.config.meta_lr
        )

        # State counters
        self.total_meta_iterations: int = 0
        self.total_env_steps:       int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_obs_dim(obs_space: gym.Space) -> int:
        """Flatten observation space to a single integer dimension."""
        if isinstance(obs_space, gym.spaces.Box):
            return int(np.prod(obs_space.shape))
        if isinstance(obs_space, gym.spaces.Dict):
            return sum(int(np.prod(s.shape)) for s in obs_space.spaces.values())
        raise ValueError(f"Unsupported observation space: {type(obs_space)}")

    @staticmethod
    def _flatten_obs(obs: Union[np.ndarray, dict]) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return np.asarray(obs, dtype=np.float32).flatten()

    # ------------------------------------------------------------------
    # In-process rollout collection  (no gradient)
    # ------------------------------------------------------------------

    def _collect_trajectories(
        self,
        env:              gym.Env,
        params:           Dict[str, torch.Tensor],
        num_trajectories: int,
    ) -> Dict[str, Any]:
        """Collect *num_trajectories* independent episodes.

        Actions are sampled with ``torch.no_grad()``.  The returned dict
        keeps observations and actions as numpy arrays so that the caller
        can re-evaluate them inside a gradient-enabled context for the
        differentiable loss computation.

        """
        max_steps = self.config.max_episode_steps

        all_obs:    List[np.ndarray] = []
        all_acts:   List[np.ndarray] = []
        all_rews:   List[float]      = []
        all_dones:  List[float]      = []
        all_values: List[float]      = []

        for _ in range(num_trajectories):
            obs_raw, _ = env.reset()
            obs = self._flatten_obs(obs_raw)

            done = False
            step = 0
            while not done and step < max_steps:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_t, _, _, value_t = self._forward_with_params(obs_t, params)

                action_np = action_t.cpu().numpy().squeeze(0)
                value_np  = value_t.cpu().item()

                next_raw, rew, term, trunc, _ = env.step(action_np)
                done = bool(term or trunc)

                all_obs.append(obs.copy())
                all_acts.append(action_np.copy())
                all_rews.append(float(rew))
                all_dones.append(1.0 if done else 0.0)
                all_values.append(value_np)

                obs = self._flatten_obs(next_raw)
                step += 1
                self.total_env_steps += 1

        return {
            "obs_np":     np.array(all_obs,    dtype=np.float32),
            "actions_np": np.array(all_acts,   dtype=np.float32),
            "rewards":    np.array(all_rews,   dtype=np.float32),
            "dones":      np.array(all_dones,  dtype=np.float32),
            "values":     np.array(all_values, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Differentiable forward / loss
    # ------------------------------------------------------------------

    def _forward_with_params(
        self,
        obs:    torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the actor-critic with *params*.

        Uses ``torch.func.functional_call`` so that arbitrary parameter
        dicts (including ones derived via inner-loop SGD) can be used
        without mutating the module's state_dict.

        """
        return functional_call(self.actor_critic, params, (obs,))

    def _evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
        params:  Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate (obs, action) pairs under *params* with gradients.

        This is called during the loss computation step (inner loop and
        outer/meta loop).  Unlike ``_forward_with_params``, this method
        is intended to be called inside a gradient-enabled context.

        """
        # Split params by sub-module prefix
        if self.actor_critic.share_features and self.actor_critic.feature_net is not None:
            feat_params = {
                k[len("feature_net."):]: v
                for k, v in params.items()
                if k.startswith("feature_net.")
            }
            features = functional_call(
                self.actor_critic.feature_net, feat_params, (obs,)
            )
        else:
            features = obs

        actor_params = {
            k[len("actor."):]: v
            for k, v in params.items()
            if k.startswith("actor.")
        }
        mean, log_std = functional_call(
            self.actor_critic.actor, actor_params, (features,)
        )

        if not (torch.isfinite(mean).all() and torch.isfinite(log_std).all()):
            raise RuntimeError(
                "NaN/Inf in actor output during evaluate_actions.  "
                "Reduce inner_lr or increase gradient clipping."
            )

        std  = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        # Invert tanh squashing to recover pre-tanh actions
        actions_clamped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions     = torch.atanh(actions_clamped)

        log_probs  = dist.log_prob(raw_actions).sum(dim=-1)
        log_probs -= torch.log(1.0 - actions_clamped.pow(2) + 1e-6).sum(dim=-1)
        entropy    = dist.entropy().sum(dim=-1)

        critic_params = {
            k[len("critic."):]: v
            for k, v in params.items()
            if k.startswith("critic.")
        }
        values = functional_call(
            self.actor_critic.critic, critic_params, (features,)
        )

        return log_probs, entropy, values

    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values:  np.ndarray,
        dones:   np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE advantage and returns, with episode-boundary masking.

        Returns (advantages, returns) as float32 tensors on self.device.
        """
        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value       = 0.0
                next_non_terminal = 0.0
            else:
                next_value        = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )
            advantages[t] = last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values

        if self.config.normalize_advantages and len(advantages) > 1:
            std = advantages.std()
            if np.isfinite(std) and std > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        return (
            torch.from_numpy(advantages.astype(np.float32)).to(self.device),
            torch.from_numpy(returns.astype(np.float32)).to(self.device),
        )

    def _pg_loss(
        self,
        params:        Dict[str, torch.Tensor],
        rollout:       Dict[str, Any],
        advantages:    torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Policy gradient loss (vanilla PG or PPO-clip) plus entropy bonus. """
        obs     = torch.from_numpy(rollout["obs_np"]).float().to(self.device)
        actions = torch.from_numpy(rollout["actions_np"]).float().to(self.device)

        log_probs, entropy, _ = self._evaluate_actions(obs, actions, params)

        if self.config.clip_eps > 0 and old_log_probs is not None:
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_eps,
                1.0 + self.config.clip_eps,
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            policy_loss = -(log_probs * advantages).mean()

        entropy_loss = -entropy.mean()
        return policy_loss + self.config.entropy_coef * entropy_loss

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    _INNER_GRAD_CLIP = 10.0  # clip inner-loop gradients to prevent explosion

    def _inner_loop_update(
        self,
        params:       Dict[str, torch.Tensor],
        rollout:      Dict[str, Any],
        create_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """One inner-loop gradient step on *rollout* data.

        ``grad.detach()`` drops the gradient *of the gradient* (i.e., the
        Hessian contribution) but ``adapted[name]`` is still a function
        of the original ``param`` tensor through the additive structure.
        This is for FOMAML first-order approximation.

        """
        advantages, _ = self._compute_advantages(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )

        # Differentiable loss: re-evaluate actions under current params
        obs     = torch.from_numpy(rollout["obs_np"]).float().to(self.device)
        actions = torch.from_numpy(rollout["actions_np"]).float().to(self.device)

        log_probs, entropy, _ = self._evaluate_actions(obs, actions, params)
        inner_loss = (
            -(log_probs * advantages).mean()
            - self.config.entropy_coef * entropy.mean()
        )

        grads = torch.autograd.grad(
            inner_loss,
            params.values(),
            create_graph=create_graph,
            allow_unused=True,
        )

        updated = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                updated[name] = param
            else:
                grad_clipped = torch.clamp(grad, -self._INNER_GRAD_CLIP, self._INNER_GRAD_CLIP)
                if not create_graph:
                    # FOMAML: detach only the gradient tensor, NOT the param.
                    # The update param - lr * grad.detach() is still
                    # differentiable w.r.t. param (through the "-" operation).
                    grad_clipped = grad_clipped.detach()
                updated[name] = param - self.config.inner_lr * grad_clipped

        return updated

    # ------------------------------------------------------------------
    # Public: adapt (test-time)
    # ------------------------------------------------------------------

    def adapt(
        self,
        env:       gym.Env,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Adapt the meta-policy to *env* using inner-loop gradient steps."""
        num_steps = num_steps if num_steps is not None else self.config.num_inner_steps

        params = OrderedDict(
            (name, param.clone())
            for name, param in self.actor_critic.named_parameters()
        )

        for _ in range(num_steps):
            rollout = self._collect_trajectories(
                env, params, self.config.num_trajectories_per_task
            )
            # No graph needed at test time
            params = self._inner_loop_update(params, rollout, create_graph=False)

        return params

    # ------------------------------------------------------------------
    # Public: meta_update  (training)
    # ------------------------------------------------------------------

    def meta_update(
        self,
    ) -> Dict[str, float]:
        pass
