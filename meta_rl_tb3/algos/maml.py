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
# Module-level worker for data-parallel collection.
# ---------------------------------------------------------------------------

def _collect_task_data_worker(args: tuple) -> dict:
    """Collect one task's rollout data in a subprocess (data-only, no gradients).

    This function is intentionally NOT called from ``MAML.meta_update``.
    It exists so external callers that need cheap parallelism for large-scale
    rollout collection can reuse the environment/policy machinery.

    Returns plain numpy dicts — no tensors, no gradient graph.
    """
    task_fn, param_dict_cpu, config = args

    device = torch.device("cpu")
    from meta_rl_tb3.algos.networks import ActorCritic

    # Infer dims from param names
    obs_dim = next(v.shape[0] for k, v in param_dict_cpu.items()
                   if "actor.mean_net.network.0.weight" in k)
    action_dim = 2  # default; override from log_std if present
    for k, v in param_dict_cpu.items():
        if "actor.log_std" in k:
            action_dim = v.shape[0]
            break

    actor_critic = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=config.network_config.hidden_sizes,
        activation=config.network_config.activation,
    ).to(device)

    params = OrderedDict(
        (k, torch.tensor(v, dtype=torch.float32, device=device))
        for k, v in param_dict_cpu.items()
    )

    def flatten_obs(obs):
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return np.asarray(obs).flatten()

    def collect_trajectories(env, p, n_traj):
        max_steps = config.max_episode_steps
        all_obs, all_acts, all_rews, all_dones, all_lp, all_vals = (
            [] for _ in range(6)
        )
        for _ in range(n_traj):
            obs_raw, _ = env.reset()
            obs = flatten_obs(obs_raw)
            done = False
            step = 0
            while not done and step < max_steps:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    a_t, lp_t, _, v_t = functional_call(actor_critic, p, (obs_t,))
                a = a_t.squeeze(0).numpy()
                next_raw, rew, term, trunc, _ = env.step(a)
                done = term or trunc
                all_obs.append(obs.copy())
                all_acts.append(a)
                all_rews.append(float(rew))
                all_dones.append(float(done))
                all_lp.append(float(lp_t.item()))
                all_vals.append(float(v_t.item()))
                obs = flatten_obs(next_raw)
                step += 1
        return {
            "observations":   np.array(all_obs,   dtype=np.float32),
            "actions":        np.array(all_acts,  dtype=np.float32),
            "rewards":        np.array(all_rews,  dtype=np.float32),
            "dones":          np.array(all_dones, dtype=np.float32),
            "old_log_probs":  np.array(all_lp,    dtype=np.float32),
            "values":         np.array(all_vals,  dtype=np.float32),
        }

    def compute_adv_returns(rewards, values, dones, gamma, lam):
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if t == len(rewards) - 1 else values[t + 1]
            nt = 0.0 if t == len(rewards) - 1 else 1.0 - dones[t]
            delta = rewards[t] + gamma * nv * nt - values[t]
            adv[t] = last_gae = delta + gamma * lam * nt * last_gae
        std = adv.std()
        if np.isfinite(std) and std > 1e-8:
            adv = (adv - adv.mean()) / (std + 1e-8)
        return adv

    def inner_update_numpy(p, data):
        adv = compute_adv_returns(
            data["rewards"], data["values"], data["dones"],
            config.gamma, config.gae_lambda,
        )
        adv_t  = torch.from_numpy(adv)
        obs_t  = torch.from_numpy(data["observations"])
        acts_t = torch.from_numpy(data["actions"])
        ap = {k[len("actor."):]: v for k, v in p.items() if k.startswith("actor.")}
        mean, log_std = functional_call(actor_critic.actor, ap, (obs_t,))
        std_t = torch.exp(log_std)
        dist  = torch.distributions.Normal(mean, std_t)
        acts_c = acts_t.clamp(-1 + 1e-6, 1 - 1e-6)
        raw    = torch.atanh(acts_c)
        lp     = dist.log_prob(raw).sum(-1) - torch.log(1 - acts_c.pow(2) + 1e-6).sum(-1)
        ent    = dist.entropy().sum(-1)
        loss   = -(lp * adv_t).mean() - config.entropy_coef * ent.mean()
        grads  = torch.autograd.grad(loss, p.values(), allow_unused=True)
        _CLIP  = 10.0
        new_p  = OrderedDict()
        for (name, param), grad in zip(p.items(), grads):
            if grad is None:
                new_p[name] = param
            else:
                new_p[name] = param - config.inner_lr * torch.clamp(grad, -_CLIP, _CLIP)
        return new_p

    env = task_fn()
    try:
        pre_data = collect_trajectories(env, params, config.num_trajectories_per_task)
        adapted  = OrderedDict((k, v.clone().detach()) for k, v in params.items())
        for _ in range(config.num_inner_steps):
            adapted = inner_update_numpy(adapted, pre_data)
        post_data = collect_trajectories(env, adapted, config.num_trajectories_per_task)
    finally:
        env.close()

    return {
        "pre_data":      pre_data,
        "post_data":     post_data,
        "adapted_params": {k: v.detach().numpy() for k, v in adapted.items()},
    }


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


# ---------------------------------------------------------------------------
# TaskBatch helper
# ---------------------------------------------------------------------------

class TaskBatch:
    """Wraps a list of task callables for meta-training."""

    def __init__(self, tasks: List[Any]):
        self.tasks = list(tasks)
        self.size  = len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# MAML
# ---------------------------------------------------------------------------

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
        task_batch: Union[TaskBatch, List],
    ) -> Dict[str, float]:
        """One meta-update across a batch of tasks.

        The full computation graph is maintained in-process:

        1. For each task, clone the meta-parameters into a dict of
           differentiable tensors rooted at ``self.actor_critic.parameters()``.
        2. Collect support rollout under ``torch.no_grad()``.
        3. Compute the inner-loop loss (with gradients enabled) and step
           the cloned parameters.  Depending on ``config.first_order``,
           inner gradients are detached (FOMAML) or retained (full MAML).
        4. Collect query rollout under ``torch.no_grad()`` using the
           adapted parameters.
        5. Compute the query (meta) loss.  Gradients flow back through
           the adapted parameters to the original meta-parameters.
        6. After accumulating across all tasks, call
           ``meta_loss.backward()`` and ``meta_optimizer.step()``.
           
        """
        self.actor_critic.train()

        if not isinstance(task_batch, TaskBatch):
            task_batch = TaskBatch(task_batch)

        # Whether to build the full second-order graph
        create_inner_graph = not self.config.first_order

        meta_loss        = torch.tensor(0.0, device=self.device)
        all_pre_rewards  = []
        all_post_rewards = []
        all_ep_lengths   = []
        all_entropies    = []
        per_task_improvements: List[float] = []

        for task_fn in task_batch:
            env = task_fn() if callable(task_fn) else task_fn
            try:
                # ── Step 1: clone meta-params keeping graph connection ─────
                # ``param.clone()`` creates a new tensor that is part of the
                # same autograd graph as ``param`` (it is differentiable
                # w.r.t. the original module weights).
                params = OrderedDict(
                    (name, param.clone())
                    for name, param in self.actor_critic.named_parameters()
                )

                # ── Step 2: support rollout (no gradient) ─────────────────
                support = self._collect_trajectories(
                    env, params, self.config.num_trajectories_per_task
                )

                # ── Step 3: inner-loop update ──────────────────────────────
                adapted = self._inner_loop_update(
                    params, support, create_graph=create_inner_graph
                )

                # ── Step 4: query rollout with adapted params (no gradient) ─
                query = self._collect_trajectories(
                    env, adapted, self.config.num_trajectories_per_task
                )

                # ── Step 5: query (meta) loss — gradient must reach params ──
                query_advantages, _ = self._compute_advantages(
                    query["rewards"], query["values"], query["dones"]
                )
                task_loss = self._pg_loss(adapted, query, query_advantages)
                meta_loss = meta_loss + task_loss / task_batch.size

                # ── Metrics (no gradient needed) ───────────────────────────
                pre_rew  = float(support["rewards"].sum()) / self.config.num_trajectories_per_task
                post_rew = float(query["rewards"].sum())  / self.config.num_trajectories_per_task
                all_pre_rewards.append(pre_rew)
                all_post_rewards.append(post_rew)
                per_task_improvements.append(post_rew - pre_rew)

                ep_len = (len(support["rewards"]) + len(query["rewards"])) / (
                    2.0 * self.config.num_trajectories_per_task
                )
                all_ep_lengths.append(ep_len)

                # Entropy estimate (detached, for logging only)
                with torch.no_grad():
                    obs_t = torch.from_numpy(query["obs_np"]).float().to(self.device)
                    act_t = torch.from_numpy(query["actions_np"]).float().to(self.device)
                    _, ent_t, _ = self._evaluate_actions(obs_t, act_t, adapted)
                    all_entropies.append(ent_t.mean().item())

            finally:
                env.close()

        # ── Step 6: meta-gradient update ──────────────────────────────────
        self.meta_optimizer.zero_grad()
        meta_loss.backward()

        # Verify gradient arrived (helps with debugging; removed in production)
        # assert any(p.grad is not None for p in self.actor_critic.parameters())

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.config.max_grad_norm,
            )

        self.meta_optimizer.step()
        self.total_meta_iterations += 1

        return {
            "meta_loss":                    meta_loss.item(),
            "pre_adaptation_reward":        float(np.mean(all_pre_rewards)),
            "post_adaptation_reward":       float(np.mean(all_post_rewards)),
            "pre_adaptation_reward_std":    float(np.std(all_pre_rewards)),
            "post_adaptation_reward_std":   float(np.std(all_post_rewards)),
            "adaptation_improvement":       float(np.mean(per_task_improvements)),
            "adaptation_improvement_median": float(np.median(per_task_improvements)),
            "fraction_tasks_improved":      float(np.mean([i > 0 for i in per_task_improvements])),
            "mean_episode_length":          float(np.mean(all_ep_lengths)),
            "mean_entropy":                 float(np.mean(all_entropies)),
            "total_env_steps":              self.total_env_steps,
            "meta_iteration":               self.total_meta_iterations,
            "per_task_improvements":        per_task_improvements,
        }

    # ------------------------------------------------------------------
    # Public: evaluate_adaptation
    # ------------------------------------------------------------------

    def evaluate_adaptation(
        self,
        env:                  gym.Env,
        num_adaptation_steps: Optional[List[int]] = None,
        num_eval_episodes:    int = 5,
    ) -> Dict[str, float]:
        """Evaluate adaptation quality at several inner-step counts."""
        if num_adaptation_steps is None:
            num_adaptation_steps = [0, 1, 3, 5, 10]

        results: Dict[str, float] = {}

        for k in sorted(num_adaptation_steps):
            # Fresh clone of meta-parameters for each k — independent measurement
            params = OrderedDict(
                (name, param.clone())
                for name, param in self.actor_critic.named_parameters()
            )

            for _ in range(k):
                rollout = self._collect_trajectories(
                    env, params, self.config.num_trajectories_per_task
                )
                params = self._inner_loop_update(params, rollout, create_graph=False)

            # Evaluate
            rewards = []
            for _ in range(num_eval_episodes):
                obs_raw, _ = env.reset()
                obs = self._flatten_obs(obs_raw)
                ep_reward = 0.0
                done = False
                step = 0

                while not done and step < self.config.max_episode_steps:
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_t, _, _, _ = self._forward_with_params(obs_t, params)
                    action_np = action_t.cpu().numpy().squeeze(0)
                    next_raw, rew, term, trunc, _ = env.step(action_np)
                    done = bool(term or trunc)
                    ep_reward += rew
                    obs = self._flatten_obs(next_raw)
                    step += 1

                rewards.append(ep_reward)

            results[f"reward_at_step_{k}"] = float(np.mean(rewards))

        return results

    # ------------------------------------------------------------------
    # Public: save / load / get_adapted_policy
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save a checkpoint to *path*."""
        torch.save(
            {
                "actor_critic_state_dict":    self.actor_critic.state_dict(),
                "meta_optimizer_state_dict":  self.meta_optimizer.state_dict(),
                "total_meta_iterations":      self.total_meta_iterations,
                "total_env_steps":            self.total_env_steps,
                "config":                     self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a checkpoint from *path*."""
        # weights_only=False required because the checkpoint includes the
        # MAMLConfig dataclass object (not just raw tensors).  The checkpoint
        # is produced exclusively by MAML.save() in this codebase.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        self.total_meta_iterations = checkpoint["total_meta_iterations"]
        self.total_env_steps       = checkpoint["total_env_steps"]

    def get_adapted_policy(
        self,
        env:       gym.Env,
        num_steps: Optional[int] = None,
    ) -> ActorCritic:
        """Return a deep-copied ActorCritic loaded with adapted parameters."""
        adapted_params  = self.adapt(env, num_steps)
        adapted_network = copy.deepcopy(self.actor_critic)

        state_dict = adapted_network.state_dict()
        for name, param in adapted_params.items():
            state_dict[name] = param.detach().cpu()
        adapted_network.load_state_dict(state_dict)
        return adapted_network
