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


# ---------------------------------------------------------------------------
# CADRE
# ---------------------------------------------------------------------------

class CADRE:
    """Context-Adaptive meta-RL combining context inference + gradient adaptation.

    The three ablation variants are all instantiated through this class:

    - CADRE (proposed):            encoder_type="gru",  num_inner_steps=1
    - CADRE-no-grad (context only): encoder_type="gru",  num_inner_steps=0
    - CADRE-no-context (= FOMAML): encoder_type="none", num_inner_steps=1

    The last variant uses exactly the same code path as the primary FOMAML
    baseline so comparisons are fair by construction.
    """

    _INNER_GRAD_CLIP = 10.0   # prevent inner-loop gradient explosion

    def __init__(
        self,
        env_fn:  Callable[[], gym.Env],
        config:  Optional[CADREConfig] = None,
    ):
        self.env_fn = env_fn
        self.config = config or CADREConfig()

        # Device
        # normally i use cpu for testing
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

        # Infer dims
        _env = env_fn()
        self.obs_dim    = self._get_obs_dim(_env.observation_space)
        self.action_dim = _env.action_space.shape[0]
        _env.close()

        # Sync encoder config dims with env
        self.config.context_encoder_config.obs_dim    = self.obs_dim
        self.config.context_encoder_config.action_dim = self.action_dim

        # Actor-critic
        self.actor_critic = ActorCritic(
            obs_dim=self.obs_dim + (
                self.config.context_encoder_config.context_dim
                if self.config.encoder_type != "none" else 0
            ),
            action_dim=self.action_dim,
            hidden_sizes=self.config.network_config.hidden_sizes,
            activation=self.config.network_config.activation,
        ).to(self.device)

        # Context encoder (None when encoder_type=="none")
        if self.config.encoder_type != "none":
            self.encoder = ContextEncoder(
                self.config.context_encoder_config
            ).to(self.device)
            self.context_dim = self.config.context_encoder_config.context_dim
        else:
            self.encoder     = None
            self.context_dim = 0

        # Meta-optimizer covers both actor-critic AND encoder (if present)
        params = list(self.actor_critic.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        self.meta_optimizer = Adam(params, lr=self.config.meta_lr)

        # Counters
        self.total_meta_iterations: int = 0
        self.total_env_steps:       int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_obs_dim(obs_space: gym.Space) -> int:
        if isinstance(obs_space, gym.spaces.Box):
            return int(np.prod(obs_space.shape))
        if isinstance(obs_space, gym.spaces.Dict):
            return sum(int(np.prod(s.shape)) for s in obs_space.spaces.values())
        raise ValueError(f"Unsupported obs space: {type(obs_space)}")

    @staticmethod
    def _flatten_obs(obs: Union[np.ndarray, dict]) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return np.asarray(obs, dtype=np.float32).flatten()

    def _policy_obs_dim(self) -> int:
        """Input dimension to the actor-critic (obs + context)."""
        return self.obs_dim + self.context_dim

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _build_transition(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
    ) -> np.ndarray:
        """Concatenate (s, a, r, s') into a flat transition vector."""
        return np.concatenate([
            obs.astype(np.float32),
            action.astype(np.float32),
            np.array([reward], dtype=np.float32),
            next_obs.astype(np.float32),
        ])

    def _encode_context(
        self,
        transition_buffer: deque,
        encoder_params:    Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode the current transition buffer into a context vector."""
        K = self.config.context_encoder_config.context_window

        if self.encoder is None or len(transition_buffer) == 0:
            return torch.zeros(1, self.context_dim, device=self.device)

        # Take the most recent K transitions (or fewer if buffer not full)
        buf = list(transition_buffer)[-K:]
        transitions = np.stack(buf, axis=0)          # (len, trans_dim)

        # Pad to K if we don't have enough yet (beginning of episode)
        if len(buf) < K:
            pad = np.zeros(
                (K - len(buf), transitions.shape[1]), dtype=np.float32
            )
            transitions = np.concatenate([pad, transitions], axis=0)

        t_tensor = torch.from_numpy(transitions).float().to(self.device)
        t_tensor = t_tensor.unsqueeze(0)             # (1, K, trans_dim)

        # Functional call so we can use adapted encoder params
        enc_params = {
            k[len(""):]: v
            for k, v in encoder_params.items()
        }
        z, _ = functional_call(self.encoder, enc_params, (t_tensor,))
        return z   # (1, context_dim)

    # ------------------------------------------------------------------
    # In-process rollout collection (no gradient)
    # ------------------------------------------------------------------

    def _collect_rollout(
        self,
        env:             gym.Env,
        policy_params:   Dict[str, torch.Tensor],
        encoder_params:  Dict[str, torch.Tensor],
        num_episodes:    int,
    ) -> Dict[str, Any]:
        """Collect rollouts under no_grad with causal context updates."""
        K = self.config.context_encoder_config.context_window
        trans_dim = self.obs_dim + self.action_dim + 1 + self.obs_dim

        all_obs     : List[np.ndarray] = []
        all_acts    : List[np.ndarray] = []
        all_ctxs    : List[np.ndarray] = []
        all_rews    : List[float]      = []
        all_dones   : List[float]      = []
        all_values  : List[float]      = []

        for _ in range(num_episodes):
            obs_raw, _ = env.reset()
            obs = self._flatten_obs(obs_raw)

            # Fresh transition buffer for each episode
            trans_buf: deque = deque(maxlen=K)

            done = False
            step = 0

            while not done and step < self.config.max_episode_steps:
                # ── 1. Encode context from transitions BEFORE this step ──
                with torch.no_grad():
                    z = self._encode_context(trans_buf, encoder_params)
                    z_np = z.cpu().numpy().squeeze(0)   # (context_dim,)

                    # Build policy input: [obs; z]
                    if self.context_dim > 0:
                        policy_obs = np.concatenate([obs, z_np])
                    else:
                        policy_obs = obs

                    obs_t = torch.from_numpy(policy_obs).float().unsqueeze(0).to(self.device)
                    action_t, _, _, value_t = self._forward_policy(obs_t, policy_params)

                action_np = action_t.cpu().numpy().squeeze(0)
                value_np  = value_t.cpu().item()

                # ── 2. Execute action ────────────────────────────────────
                next_raw, rew, term, trunc, _ = env.step(action_np)
                done = bool(term or trunc)
                next_obs = self._flatten_obs(next_raw)

                # ── 3. Record ────────────────────────────────────────────
                all_obs.append(obs.copy())
                all_acts.append(action_np.copy())
                all_ctxs.append(z_np.copy())
                all_rews.append(float(rew))
                # dones[t] = 1.0 means step t is the last step in the episode
                # (term or trunc observed after executing action at step t).
                all_dones.append(1.0 if done else 0.0)
                all_values.append(value_np)

                # ── 4. Update transition buffer AFTER action ─────────────
                trans = self._build_transition(obs, action_np, rew, next_obs)
                trans_buf.append(trans)

                obs = next_obs
                step += 1
                self.total_env_steps += 1

        return {
            "obs_np":     np.array(all_obs,    dtype=np.float32),
            "actions_np": np.array(all_acts,   dtype=np.float32),
            "context_np": np.array(all_ctxs,   dtype=np.float32),
            "rewards":    np.array(all_rews,   dtype=np.float32),
            "dones":      np.array(all_dones,  dtype=np.float32),
            "values":     np.array(all_values, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Differentiable forward / loss
    # ------------------------------------------------------------------

    def _forward_policy(
        self,
        obs:           torch.Tensor,
        policy_params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the actor-critic with custom params."""
        return functional_call(self.actor_critic, policy_params, (obs,))

    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values:  np.ndarray,
        dones:   np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE advantages and returns."""
        # Indexing convention (dones[t] semantics):
        #   dones[t] = 1.0 means transition t ended the episode, i.e.
        #   the episode was terminal/truncated AFTER step t was executed.
        #   Therefore s_{t+1} is a terminal state and we must NOT bootstrap
        #   from values[t+1]. The condition `1.0 - dones[t]` correctly
        #   zeroes out the bootstrap for the last step in each episode.
        #   This is the standard convention (Schulman et al. 2016 GAE;
        #   Stable-Baselines3 PPO) and is NOT an off-by-one error.
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nv  = 0.0 if t == len(rewards) - 1 else values[t + 1]
            nt  = 0.0 if t == len(rewards) - 1 else 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * nv * nt - values[t]
            adv[t] = last_gae = (
                delta + self.config.gamma * self.config.gae_lambda * nt * last_gae
            )
        returns = adv + values
        if self.config.normalize_advantages and len(adv) > 1:
            std = adv.std()
            if np.isfinite(std) and std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
        return (
            torch.from_numpy(adv.astype(np.float32)).to(self.device),
            torch.from_numpy(returns.astype(np.float32)).to(self.device),
        )

    def _pg_loss(
        self,
        policy_params: Dict[str, torch.Tensor],
        rollout:       Dict[str, Any],
        advantages:    torch.Tensor,
    ) -> torch.Tensor:
        """Policy gradient loss.

        Re-evaluates (obs, action) pairs with the CURRENT policy_params.
        If a context encoder is active, the stored context vectors are
        concatenated to the observations so the gradient flows through
        both the policy parameters AND (via the stored context) through
        the encoder parameters as well.

        Note: the context vectors stored in rollout["context_np"] were
        produced under no_grad during rollout collection.  They are used
        here as fixed inputs — the encoder gradient enters through the
        outer-loop query loss, not the inner-loop loss.
        """
        obs_np = rollout["obs_np"]
        ctx_np = rollout["context_np"]
        act_np = rollout["actions_np"]

        if self.context_dim > 0:
            combined_np = np.concatenate([obs_np, ctx_np], axis=-1)
        else:
            combined_np = obs_np

        obs_t  = torch.from_numpy(combined_np).float().to(self.device)
        acts_t = torch.from_numpy(act_np).float().to(self.device)

        # Split params: actor params only for the PG loss
        actor_params = {
            k[len("actor."):]: v
            for k, v in policy_params.items()
            if k.startswith("actor.")
        }
        feat_params = {
            k[len("feature_net."):]: v
            for k, v in policy_params.items()
            if k.startswith("feature_net.")
        }

        if self.actor_critic.share_features and feat_params:
            features = functional_call(
                self.actor_critic.feature_net, feat_params, (obs_t,)
            )
        else:
            features = obs_t

        mean, log_std = functional_call(
            self.actor_critic.actor, actor_params, (features,)
        )

        if not (torch.isfinite(mean).all() and torch.isfinite(log_std).all()):
            raise RuntimeError(
                "NaN/Inf in actor output — reduce inner_lr or check reward scale."
            )

        std  = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        acts_c  = acts_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw     = torch.atanh(acts_c)
        lprobs  = dist.log_prob(raw).sum(-1)
        lprobs -= torch.log(1.0 - acts_c.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)

        return -(lprobs * advantages).mean() - self.config.entropy_coef * entropy.mean()

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _inner_update(
        self,
        policy_params:  Dict[str, torch.Tensor],
        rollout:        Dict[str, Any],
        create_graph:   bool = False,
    ) -> Dict[str, torch.Tensor]:
        """One FOMAML/MAML inner-loop gradient step on policy_params.

        Only the policy parameters (actor-critic) are updated in the
        inner loop.  The encoder parameters φ are NOT updated — they
        receive their gradient through the outer meta-loss only.
        This is a deliberate design choice: the encoder adapts at the
        meta-level (between tasks), not within a task.
        """
        adv, _ = self._compute_advantages(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        loss = self._pg_loss(policy_params, rollout, adv)

        grads = torch.autograd.grad(
            loss,
            policy_params.values(),
            create_graph=create_graph,
            allow_unused=True,
        )

        updated = OrderedDict()
        for (name, param), grad in zip(policy_params.items(), grads):
            if grad is None:
                updated[name] = param
            else:
                g = torch.clamp(grad, -self._INNER_GRAD_CLIP, self._INNER_GRAD_CLIP)
                if not create_graph:
                    g = g.detach()   # FOMAML: stop second-order graph
                updated[name] = param - self.config.inner_lr * g

        return updated

    # ------------------------------------------------------------------
    # Public: meta_update (training)
    # ------------------------------------------------------------------

    def meta_update(
        self,
        task_batch: Union[list, Any],
    ) -> Dict[str, float]:
        """One meta-update across a batch of tasks."""
        self.actor_critic.train()
        if self.encoder is not None:
            self.encoder.train()

        # Resolve tasks to callables
        tasks = list(task_batch) if hasattr(task_batch, "__iter__") else [task_batch]

        create_inner_graph = not self.config.first_order

        meta_loss         = torch.tensor(0.0, device=self.device)
        all_pre_rewards:  List[float] = []
        all_post_rewards: List[float] = []
        per_task_improvements: List[float] = []
        all_ep_lengths:   List[float] = []
        all_entropies:    List[float] = []

        for task_fn in tasks:
            env = task_fn() if callable(task_fn) else task_fn

            try:
                # ── Clone policy params (keeps graph connection) ─────────
                policy_params = OrderedDict(
                    (name, param.clone())
                    for name, param in self.actor_critic.named_parameters()
                )
                # Encoder params are NOT inner-loop adapted
                encoder_params = dict(self.encoder.named_parameters()) \
                    if self.encoder is not None else {}

                # ── Support rollout ──────────────────────────────────────
                support = self._collect_rollout(
                    env, policy_params, encoder_params,
                    self.config.num_support_episodes,
                )

                # ── Inner loop ───────────────────────────────────────────
                adapted = policy_params
                for _ in range(self.config.num_inner_steps):
                    adapted = self._inner_update(
                        adapted, support, create_graph=create_inner_graph
                    )

                # ── Query rollout with adapted params ────────────────────
                query = self._collect_rollout(
                    env, adapted, encoder_params,
                    self.config.num_query_episodes,
                )

                # ── Query (meta) loss ────────────────────────────────────
                q_adv, _ = self._compute_advantages(
                    query["rewards"], query["values"], query["dones"]
                )

                # For the outer loss, get a differentiable context from the
                # support transitions so φ receives meta-gradients.
                # Use the FULL support sequence in one GRU call (O(N_support_steps))
                # rather than a per-timestep causal recompute (O(N² ) previously).
                # This is a valid approximation: the context used at eval time is
                # the support context, not a per-step causal context.
                if self.encoder is not None:
                    # Encode support transitions once; broadcast to query batch size
                    n_query   = len(query["obs_np"])
                    query_ctx = self._compute_episode_context_with_grad(
                        support, encoder_params, n_broadcast=n_query
                    )
                    task_loss = self._pg_loss_with_context_tensor(
                        adapted, query, q_adv, query_ctx
                    )
                else:
                    task_loss = self._pg_loss(adapted, query, q_adv)

                meta_loss = meta_loss + task_loss / len(tasks)

                # ── Metrics ──────────────────────────────────────────────
                pre  = float(support["rewards"].sum()) / self.config.num_support_episodes
                post = float(query["rewards"].sum())   / self.config.num_query_episodes
                all_pre_rewards.append(pre)
                all_post_rewards.append(post)
                per_task_improvements.append(post - pre)
                ep_len = (len(support["rewards"]) + len(query["rewards"])) / (
                    2.0 * max(self.config.num_support_episodes,
                              self.config.num_query_episodes)
                )
                all_ep_lengths.append(ep_len)

                with torch.no_grad():
                    obs_t = torch.from_numpy(
                        np.concatenate([query["obs_np"], query["context_np"]], axis=-1)
                        if self.context_dim > 0 else query["obs_np"]
                    ).float().to(self.device)
                    acts_t = torch.from_numpy(query["actions_np"]).float().to(self.device)
                    ap = {k[len("actor."):]: v for k, v in adapted.items()
                          if k.startswith("actor.")}
                    feats = obs_t
                    if self.actor_critic.share_features and self.actor_critic.feature_net:
                        fp = {k[len("feature_net."):]: v for k, v in adapted.items()
                              if k.startswith("feature_net.")}
                        feats = functional_call(self.actor_critic.feature_net, fp, (obs_t,))
                    mean, log_std = functional_call(
                        self.actor_critic.actor, ap, (feats,)
                    )
                    ent = torch.distributions.Normal(
                        mean, torch.exp(log_std)
                    ).entropy().sum(-1).mean()
                    all_entropies.append(ent.item())

            finally:
                env.close()

        # ── Meta-gradient update ─────────────────────────────────────────
        self.meta_optimizer.zero_grad()
        meta_loss.backward()

        all_params = list(self.actor_critic.parameters())
        if self.encoder is not None:
            all_params += list(self.encoder.parameters())
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

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
    # Re-encode context with gradient (for meta-loss)
    # ------------------------------------------------------------------

    def _compute_episode_context_with_grad(
        self,
        rollout:        Dict[str, Any],
        encoder_params: Dict[str, torch.Tensor],
        n_broadcast:    int = None,
    ) -> torch.Tensor:
        """Encode ALL support transitions in one GRU call and broadcast."""
        trans_dim = self.encoder.transition_dim

        obs_np  = rollout["obs_np"]
        acts_np = rollout["actions_np"]
        rews_np = rollout["rewards"]

        next_obs_np          = np.empty_like(obs_np)
        next_obs_np[:-1]     = obs_np[1:]
        next_obs_np[-1]      = obs_np[-1]

        transitions_np = np.concatenate([
            obs_np, acts_np, rews_np[:, None], next_obs_np,
        ], axis=-1).astype(np.float32)

        # Single GRU call over the full support sequence
        t_full = torch.from_numpy(transitions_np).float().to(self.device).unsqueeze(0)

        enc_t_params = {k[len("transition_encoder."):]: v
                        for k, v in encoder_params.items()
                        if k.startswith("transition_encoder.")}
        encoded = functional_call(self.encoder.transition_encoder, enc_t_params, (t_full,))

        gru_params = {k[len("gru."):]: v
                      for k, v in encoder_params.items()
                      if k.startswith("gru.")}
        gru_out, _ = functional_call(self.encoder.gru, gru_params, (encoded,))

        head_params = {k[len("task_head."):]: v
                       for k, v in encoder_params.items()
                       if k.startswith("task_head.")}
        final      = gru_out[:, -1, :]             # (1, gru_hidden)
        z          = functional_call(self.encoder.task_head, head_params, (final,))
        # (1, context_dim)

        # Broadcast to the requested batch size
        n = n_broadcast if n_broadcast is not None else len(rollout["obs_np"])
        return z.expand(n, -1)  # (n, context_dim)

    def _recompute_context_with_grad(
        self,
        rollout:        Dict[str, Any],
        encoder_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Re-encode stored transitions through the encoder WITH gradients."""
        N         = len(rollout["obs_np"])
        trans_dim = self.encoder.transition_dim

        obs_np  = rollout["obs_np"]
        acts_np = rollout["actions_np"]
        rews_np = rollout["rewards"]

        # Reconstruct next_obs (shift by 1; last row repeated)
        next_obs_np          = np.empty_like(obs_np)
        next_obs_np[:-1]     = obs_np[1:]
        next_obs_np[-1]      = obs_np[-1]

        # Build full transition sequence: (N, trans_dim)
        transitions_np = np.concatenate([
            obs_np, acts_np, rews_np[:, None], next_obs_np,
        ], axis=-1).astype(np.float32)

        # Run encoder over entire sequence in one batched call (batch=1, seq=N)
        t_full = torch.from_numpy(transitions_np).float().to(self.device).unsqueeze(0)
        # encoder forward: (1, N, context_dim)
        # We need the GRU's per-step outputs — so we call transition_encoder + GRU
        # manually to get all outputs rather than just the final one.

        # -- Run transition MLP over all N steps --
        enc_t_params = {
            k[len("transition_encoder."):]: v
            for k, v in encoder_params.items()
            if k.startswith("transition_encoder.")
        }
        encoded = functional_call(
            self.encoder.transition_encoder, enc_t_params, (t_full,)
        )  # (1, N, gru_hidden)

        gru_params = {
            k[len("gru."):]: v
            for k, v in encoder_params.items()
            if k.startswith("gru.")
        }
        gru_out, _ = functional_call(
            self.encoder.gru, gru_params, (encoded,)
        )  # (1, N, gru_hidden)

        head_params = {
            k[len("task_head."):]: v
            for k, v in encoder_params.items()
            if k.startswith("task_head.")
        }
        # task_head maps each timestep's GRU output → context
        # Shape: (1, N, gru_hidden) → (1*N, gru_hidden) → head → (N, context_dim)
        gru_flat  = gru_out.squeeze(0)          # (N, gru_hidden)
        z_all     = functional_call(
            self.encoder.task_head, head_params, (gru_flat,)
        )  # (N, context_dim)

        # Causal shift: context at step t = output after seeing transitions 0..t-1.
        # GRU output at position t-1 has seen transitions 0..t-1.
        # For t=0 (no history) we use zeros.
        zero_ctx  = torch.zeros(1, self.context_dim, device=self.device)
        if N == 1:
            return zero_ctx

        # z_all[i] = context after transition i; use z_all[i-1] for step i.
        # Step 0 gets zero context; steps 1..N-1 get z_all[0..N-2].
        causal_ctx = torch.cat([zero_ctx, z_all[:-1]], dim=0)  # (N, context_dim)
        return causal_ctx

    def _pg_loss_with_context_tensor(
        self,
        policy_params: Dict[str, torch.Tensor],
        rollout:       Dict[str, Any],
        advantages:    torch.Tensor,
        context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """PG loss using a differentiable context tensor (for meta-gradient)."""
        obs_t  = torch.from_numpy(rollout["obs_np"]).float().to(self.device)
        acts_t = torch.from_numpy(rollout["actions_np"]).float().to(self.device)

        # Concatenate obs with differentiable context
        combined = torch.cat([obs_t, context_tensor], dim=-1)

        actor_params = {
            k[len("actor."):]: v
            for k, v in policy_params.items()
            if k.startswith("actor.")
        }
        feat_params = {
            k[len("feature_net."):]: v
            for k, v in policy_params.items()
            if k.startswith("feature_net.")
        }

        if self.actor_critic.share_features and feat_params:
            features = functional_call(
                self.actor_critic.feature_net, feat_params, (combined,)
            )
        else:
            features = combined

        mean, log_std = functional_call(
            self.actor_critic.actor, actor_params, (features,)
        )

        std  = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        acts_c  = acts_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw     = torch.atanh(acts_c)
        lprobs  = dist.log_prob(raw).sum(-1)
        lprobs -= torch.log(1.0 - acts_c.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)

        return -(lprobs * advantages).mean() - self.config.entropy_coef * entropy.mean()

    # ------------------------------------------------------------------
    # Public: adapt (test time)
    # ------------------------------------------------------------------

    def adapt(
        self,
        env:       gym.Env,
        num_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], deque]:
        """Adapt the meta-policy to a specific task.

        Returns
        -------
        adapted_params: Dict of adapted policy parameters.
        trans_buffer:   Deque of all transitions collected during adaptation
                        (useful for context computation after adaptation).
        """
        num_steps = num_steps if num_steps is not None else self.config.num_inner_steps
        policy_params = OrderedDict(
            (name, param.clone())
            for name, param in self.actor_critic.named_parameters()
        )
        encoder_params = dict(self.encoder.named_parameters()) \
            if self.encoder is not None else {}

        all_trans: deque = deque(
            maxlen=self.config.context_encoder_config.context_window
        )

        for _ in range(num_steps):
            rollout = self._collect_rollout(
                env, policy_params, encoder_params,
                self.config.num_support_episodes,
            )
            policy_params = self._inner_update(
                policy_params, rollout, create_graph=False
            )
            # Accumulate transitions for context
            trans = np.concatenate([
                rollout["obs_np"], rollout["actions_np"],
                rollout["rewards"][:, None],
                np.roll(rollout["obs_np"], -1, axis=0),
            ], axis=-1)
            for t in trans:
                all_trans.append(t)

        return policy_params, all_trans

    