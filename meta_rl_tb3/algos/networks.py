"""This is the core Neural network architectures for policy and value functions.

This module provides modular neural network components that can be used
with various RL algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class NetworkConfig:
    """Configuration for neural networks.
    
    Attributes:
        hidden_sizes: List of hidden layer sizes
        activation: Activation function name ('relu', 'tanh', 'elu')
        output_activation: Output activation (None for linear)
        init_std: Initial standard deviation for output layers
        use_layer_norm: Whether to use layer normalization
        use_orthogonal_init: Use orthogonal weight initialization
        log_std_bounds: Bounds for log standard deviation (actor)
    """
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    output_activation: Optional[str] = None
    init_std: float = 0.5
    use_layer_norm: bool = False
    use_orthogonal_init: bool = True
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)


def get_activation(name: str) -> nn.Module:
    # Get activation function by name
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
    }
    return activations.get(name.lower(), nn.ReLU())


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    # Apply orthogonal initialization to a layer.

    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.
    
    A flexible MLP implementation that supports:
    - Variable number of hidden layers
    - Different activation functions
    - Optional layer normalization
    - Orthogonal initialization
    
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        self.input_size = input_size
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        if use_orthogonal_init:
            self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                # Use smaller gain for output layer
                if i == len(self.network) - 1 or (
                    i == len(self.network) - 2 and 
                    not isinstance(self.network[-1], nn.Linear)
                ):
                    orthogonal_init(layer, gain=0.01)
                else:
                    orthogonal_init(layer, gain=math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        """
        return self.network(x)


class ActorNetwork(nn.Module):
    """Gaussian policy network for continuous action spaces.
    
    Outputs mean and log standard deviation of a Gaussian distribution
    over actions. The standard deviation can be state-dependent or
    independent (learned parameter).
    
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        log_std_init: float = 0.0,
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
        state_dependent_std: bool = False,
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        """Initialize actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            log_std_init: Initial log standard deviation
            log_std_bounds: Bounds for log std
            state_dependent_std: Whether std depends on state
            use_layer_norm: Whether to use layer normalization
            use_orthogonal_init: Use orthogonal initialization
        """
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        self.action_dim = action_dim
        self.log_std_bounds = log_std_bounds
        self.state_dependent_std = state_dependent_std
        
        # Mean network
        self.mean_net = MLP(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
        
        # Standard deviation
        if state_dependent_std:
            self.log_std_net = MLP(
                input_size=obs_dim,
                output_size=action_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                use_layer_norm=use_layer_norm,
                use_orthogonal_init=use_orthogonal_init,
            )
        else:
            # Learnable parameter
            self.log_std = nn.Parameter(
                torch.full((action_dim,), log_std_init)
            )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.

        """
        mean = self.mean_net(obs)
        
        if self.state_dependent_std:
            log_std = self.log_std_net(obs)
        else:
            log_std = self.log_std.expand_as(mean)
        
        # Clamp log_std to bounds
        log_std = torch.clamp(log_std, *self.log_std_bounds)
        
        return mean, log_std
    
    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Get action distribution.
        
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        return Normal(mean, std)
    
    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.
        
        Actions are squashed through tanh to [-1, 1].  The log-probability
        is corrected for the tanh change-of-variables so that it matches what
        ``evaluate`` returns for the same (post-tanh) action.
        
        """
        dist = self.get_distribution(obs)
        
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()  # Reparameterized sample
        
        # Squash to [-1, 1]
        actions = torch.tanh(raw)
        
        # Log-prob with tanh change-of-variables correction:
        #   log p(a) = log p(raw) - sum log(1 - tanh(raw)^2 + eps)
        log_probs = dist.log_prob(raw).sum(dim=-1)
        log_probs -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        
        # Entropy of the underlying Gaussian (approximation; exact entropy for
        # the squashed distribution is intractable but this is the standard
        # proxy used in SAC / PPO implementations)
        entropy = dist.entropy().sum(dim=-1)
        
        return actions, log_probs, entropy
    
    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given (post-tanh) actions.
        
        Applies the inverse tanh (atanh) to recover the raw pre-squash value,
        then computes the log-prob with the tanh change-of-variables correction
        so that the result is consistent with ``sample``.
        """
        dist = self.get_distribution(obs)
        
        # Recover pre-tanh values; clamp to avoid atanh of ±1
        actions_clamped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(actions_clamped)
        
        # Log-prob with tanh change-of-variables correction
        log_probs = dist.log_prob(raw).sum(dim=-1)
        log_probs -= torch.log(1.0 - actions_clamped.pow(2) + 1e-6).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Value function network.
    
    Can be used as state value function V(s) or state-action value Q(s, a).

    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        input_dim = obs_dim + action_dim
        
        self.network = MLP(
            input_size=input_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
        
        self.action_dim = action_dim
    
    def forward(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute value estimate.
        
        """
        if self.action_dim > 0 and action is not None:
            x = torch.cat([obs, action], dim=-1)
        else:
            x = obs
        
        return self.network(x).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic network.
    
    Provides a unified interface for actor and critic networks,
    with optional shared feature extraction.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        share_features: bool = False,
        feature_dim: int = 256,
        log_std_init: float = 0.0,
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        self.share_features = share_features
        
        if share_features:
            # Shared feature extractor
            self.feature_net = MLP(
                input_size=obs_dim,
                output_size=feature_dim,
                hidden_sizes=[hidden_sizes[0]],
                activation=activation,
                output_activation=activation,
                use_layer_norm=use_layer_norm,
                use_orthogonal_init=use_orthogonal_init,
            )
            actor_input = feature_dim
            critic_input = feature_dim
        else:
            self.feature_net = None
            actor_input = obs_dim
            critic_input = obs_dim
        
        # Actor network
        self.actor = ActorNetwork(
            obs_dim=actor_input,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes if not share_features else hidden_sizes[1:],
            activation=activation,
            log_std_init=log_std_init,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
        
        # Critic network
        self.critic = CriticNetwork(
            obs_dim=critic_input,
            hidden_sizes=hidden_sizes if not share_features else hidden_sizes[1:],
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
    
    def _get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        if self.share_features and self.feature_net is not None:
            return self.feature_net(obs)
        return obs
    
    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through actor and critic.
        
        """
        features = self._get_features(obs)
        
        actions, log_probs, entropy = self.actor.sample(features, deterministic)
        values = self.critic(features)
        
        return actions, log_probs, entropy, values
    
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Get action for a single observation."""
        features = self._get_features(obs)
        actions, _, _ = self.actor.sample(features, deterministic)
        return actions
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self._get_features(obs)
        return self.critic(features)
    
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given observations.
        """
        features = self._get_features(obs)
        
        log_probs, entropy = self.actor.evaluate(features, actions)
        values = self.critic(features)
        
        return log_probs, entropy, values


def create_actor_critic(
    obs_dim: int,
    action_dim: int,
    config: Optional[NetworkConfig] = None,
) -> ActorCritic:
    """Factory function to create actor-critic network.
    
    """
    config = config or NetworkConfig()
    
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        log_std_init=math.log(config.init_std),
        use_layer_norm=config.use_layer_norm,
        use_orthogonal_init=config.use_orthogonal_init,
    )


# =============================================================================
# HELM: Heterogeneous-task Encoder for Latent Meta-RL
# =============================================================================

@dataclass
class TaskEncoderConfig:
    """Configuration for HELM task encoder.
    
    The task encoder reads recent transitions and outputs a task embedding
    that captures the task structure without requiring task-type labels.
    
    """
    transition_hidden_dim: int = 64
    gru_hidden_dim: int = 64
    task_embedding_dim: int = 16
    num_gru_layers: int = 2
    context_length: int = 5
    use_layer_norm: bool = True


class TaskEncoder(nn.Module):
    """HELM Task Encoder - infers task embedding from recent transitions.
    
    This is the core novel component of HELM. Instead of zero-padding
    observations from heterogeneous tasks to a fixed dimension, we learn
    a task embedding from recent experience that captures task structure.
    
    Architecture:
        1. Transition encoder: MLP that encodes (obs, action, reward, next_obs)
        2. Temporal aggregator: GRU that processes sequence of transitions
        3. Task head: Linear projection to task embedding space
    
    The encoder is trained end-to-end with the policy via meta-learning.
    No task-type labels are required - the encoder learns to infer task
    structure from the reward and dynamics patterns.
    """
    
    def __init__(
        self,
        max_obs_dim: int,
        action_dim: int,
        config: Optional[TaskEncoderConfig] = None,
    ):
        """Initialize task encoder.
        
        Args:
            max_obs_dim: Maximum observation dimension across all task types
            action_dim: Action dimension
            config: Encoder configuration
        """
        super().__init__()
        
        self.config = config or TaskEncoderConfig()
        self.max_obs_dim = max_obs_dim
        self.action_dim = action_dim
        
        # Transition input: obs + action + reward + next_obs
        transition_dim = max_obs_dim + action_dim + 1 + max_obs_dim
        self.transition_dim = transition_dim
        
        # Transition encoder MLP
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, self.config.transition_hidden_dim),
            nn.LayerNorm(self.config.transition_hidden_dim) if self.config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.config.transition_hidden_dim, self.config.gru_hidden_dim),
            nn.LayerNorm(self.config.gru_hidden_dim) if self.config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        
        # Temporal aggregator GRU
        self.gru = nn.GRU(
            input_size=self.config.gru_hidden_dim,
            hidden_size=self.config.gru_hidden_dim,
            num_layers=self.config.num_gru_layers,
            batch_first=True,
        )
        
        # Task embedding projection
        self.task_head = nn.Sequential(
            nn.Linear(self.config.gru_hidden_dim, self.config.task_embedding_dim),
            nn.Tanh(),  # Bound embedding to [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        transitions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode transitions into task embedding."""
        
        batch_size, seq_len, _ = transitions.shape
        
        # Encode each transition
        # (batch, seq_len, transition_dim) -> (batch, seq_len, gru_hidden_dim)
        encoded = self.transition_encoder(transitions)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.config.num_gru_layers,
                batch_size,
                self.config.gru_hidden_dim,
                device=transitions.device,
                dtype=transitions.dtype,
            )
        
        # Process through GRU
        # output: (batch, seq_len, gru_hidden_dim)
        # hidden: (num_layers, batch, gru_hidden_dim)
        output, hidden = self.gru(encoded, hidden)
        
        # Use final output for task embedding
        final_output = output[:, -1, :]  # (batch, gru_hidden_dim)
        
        # Project to task embedding
        task_embedding = self.task_head(final_output)
        
        return task_embedding, hidden
    
    def get_initial_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        
        return torch.zeros(
            batch_size,
            self.config.task_embedding_dim,
            device=device,
        )


class HELMActorCritic(nn.Module):
    """HELM Actor-Critic that conditions on task embeddings.
    
    This network takes both observations and task embeddings as input.
    The task embedding is concatenated with the observation before being
    processed by the actor and critic networks.
    
    Key difference from standard ActorCritic:
    - Input is (obs, task_embedding) instead of just obs
    - Works with raw (variable-dimension) observations - no padding needed
    - Task encoder is a separate module that can be trained jointly
    
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        task_embedding_dim: int = 16,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        log_std_init: float = 0.0,
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        """Initialize HELM actor-critic."""
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        self.obs_dim = obs_dim
        self.task_embedding_dim = task_embedding_dim
        
        # Combined input: obs + task_embedding
        combined_dim = obs_dim + task_embedding_dim
        
        # Actor network
        self.actor = ActorNetwork(
            obs_dim=combined_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            log_std_init=log_std_init,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
        
        # Critic network
        self.critic = CriticNetwork(
            obs_dim=combined_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
        )
    
    def _combine_inputs(
        self, obs: torch.Tensor, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate observation and task embedding."""
        return torch.cat([obs, task_embedding], dim=-1)
    
    def forward(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        combined = self._combine_inputs(obs, task_embedding)
        
        actions, log_probs, entropy = self.actor.sample(combined, deterministic)
        values = self.critic(combined)
        
        return actions, log_probs, entropy, values
    
    def get_action(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        
        combined = self._combine_inputs(obs, task_embedding)
        actions, _, _ = self.actor.sample(combined, deterministic)
        return actions
    
    def get_value(
        self, obs: torch.Tensor, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        
        combined = self._combine_inputs(obs, task_embedding)
        return self.critic(combined)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        combined = self._combine_inputs(obs, task_embedding)
        
        log_probs, entropy = self.actor.evaluate(combined, actions)
        values = self.critic(combined)
        
        return log_probs, entropy, values


class HELM(nn.Module):
    """Complete HELM system: Task Encoder + Actor-Critic.
    
    This is the full HELM architecture that combines:
    1. TaskEncoder: Infers task embedding from recent transitions
    2. HELMActorCritic: Policy conditioned on task embedding
    
    The system processes heterogeneous tasks without task-type labels:
    - Each task type can have different observation dimensions
    - The task encoder learns to infer task structure from experience
    - The policy adapts its behavior based on the inferred task embedding
    """
    
    def __init__(
        self,
        max_obs_dim: int,
        action_dim: int,
        task_encoder_config: Optional[TaskEncoderConfig] = None,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        log_std_init: float = 0.0,
        use_layer_norm: bool = False,
    ):
        """Initialize HELM system.
        
        """
        super().__init__()
        
        self.max_obs_dim = max_obs_dim
        self.action_dim = action_dim
        
        # Task encoder
        self.task_encoder_config = task_encoder_config or TaskEncoderConfig()
        self.task_encoder = TaskEncoder(
            max_obs_dim=max_obs_dim,
            action_dim=action_dim,
            config=self.task_encoder_config,
        )
        
        # Actor-critic (takes max_obs_dim for padded inputs)
        self.actor_critic = HELMActorCritic(
            obs_dim=max_obs_dim,
            action_dim=action_dim,
            task_embedding_dim=self.task_encoder_config.task_embedding_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            log_std_init=log_std_init,
            use_layer_norm=use_layer_norm,
        )
    
    def get_initial_task_embedding(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Get initial task embedding for episode start."""
        return self.task_encoder.get_initial_embedding(batch_size, device)
    
    def encode_transitions(
        self,
        transitions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode transitions into task embedding.   """
        return self.task_encoder(transitions, hidden)
    
    def forward(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass - sample action given obs and task embedding."""
        return self.actor_critic(obs, task_embedding, deterministic)
    
    def get_action(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for observation and task embedding."""
        return self.actor_critic.get_action(obs, task_embedding, deterministic)
    
    def get_value(
        self, obs: torch.Tensor, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Get value estimate."""
        return self.actor_critic.get_value(obs, task_embedding)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions."""
        return self.actor_critic.evaluate_actions(obs, task_embedding, actions)


def create_helm(
    max_obs_dim: int,
    action_dim: int,
    task_encoder_config: Optional[TaskEncoderConfig] = None,
    network_config: Optional[NetworkConfig] = None,
) -> HELM:
    """Factory function to create HELM system."""
    network_config = network_config or NetworkConfig()
    
    return HELM(
        max_obs_dim=max_obs_dim,
        action_dim=action_dim,
        task_encoder_config=task_encoder_config,
        hidden_sizes=network_config.hidden_sizes,
        activation=network_config.activation,
        log_std_init=math.log(network_config.init_std),
        use_layer_norm=network_config.use_layer_norm,
    )


# =============================================================================
# CADRE ContextEncoder
# ========================Cha=====================================================

@dataclass
class ContextEncoderConfig:
    """Configuration for the CADRE causal context encoder.

    The encoder infers a latent context vector ``z`` from the last
    ``context_window`` completed transitions ``(s, a, r, s')``.
    The context vector is used to condition the CADRE policy and critic.
    """
    obs_dim:               int   = 370
    action_dim:            int   = 2
    context_dim:           int   = 16
    context_window:        int   = 5
    gru_hidden_dim:        int   = 64
    num_gru_layers:        int   = 2
    transition_hidden_dim: int   = 64
    use_layer_norm:        bool  = True


class ContextEncoder(nn.Module):
    """CADRE causal context encoder.

    Encodes the last ``context_window`` completed transitions into a
    ``context_dim``-dimensional context vector ``z_t`` that conditions
    the navigation policy and critic.

    Causality
    ---------
    The encoder is **causal**: at step ``t`` it only sees transitions
    ``(s_{t-K}, a_{t-K}, r_{t-K}, s_{t-K+1}), ..., (s_{t-1}, a_{t-1}, r_{t-1}, s_t)``.
    Specifically, the action ``a_t`` chosen at step ``t`` is selected
    BEFORE the encoder sees the transition that results from ``a_t``.
    This matches the deployment protocol exactly and prevents information
    leakage.

    Architecture
    ------------
    1. Per-transition MLP:
         (obs + action + reward + next_obs) → hidden_dim
         with optional LayerNorm + ReLU
    2. Temporal GRU:
         sequence of hidden_dim vectors → GRU output
    3. Task head:
         GRU final output → Linear(context_dim) + Tanh → z ∈ [-1, 1]^{d_z}

    Transition input dimension
    --------------------------
        obs_dim + action_dim + 1 + obs_dim = 370 + 2 + 1 + 370 = 743

    """

    def __init__(self, config: Optional[ContextEncoderConfig] = None):
        super().__init__()
        self.config = config or ContextEncoderConfig()
        cfg = self.config

        # Transition input: obs + action + reward + next_obs
        self.transition_dim = cfg.obs_dim + cfg.action_dim + 1 + cfg.obs_dim

        # Per-transition encoder
        layers: List[nn.Module] = [
            nn.Linear(self.transition_dim, cfg.transition_hidden_dim),
        ]
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.transition_hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.transition_hidden_dim, cfg.gru_hidden_dim))
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.gru_hidden_dim))
        layers.append(nn.ReLU())
        self.transition_encoder = nn.Sequential(*layers)

        # Temporal aggregator
        self.gru = nn.GRU(
            input_size=cfg.gru_hidden_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=cfg.num_gru_layers,
            batch_first=True,
        )

        # Task embedding head
        self.task_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.context_dim),
            nn.Tanh(),   # bound z to [-1, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(
        self,
        transitions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of transition sequences.

        """
        batch, seq_len, _ = transitions.shape

        # Encode each transition independently
        encoded = self.transition_encoder(transitions)  # (B, K, gru_hidden)

        if hidden is None:
            hidden = torch.zeros(
                self.config.num_gru_layers,
                batch,
                self.config.gru_hidden_dim,
                device=transitions.device,
                dtype=transitions.dtype,
            )

        gru_out, hidden = self.gru(encoded, hidden)   # gru_out: (B, K, H)
        final = gru_out[:, -1, :]                     # (B, H) — last step

        z = self.task_head(final)                     # (B, context_dim)
        return z, hidden

    def get_zero_context(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return a zero context vector for the start of an episode."""
        return torch.zeros(
            batch_size,
            self.config.context_dim,
            device=device,
            dtype=torch.float32,
        )
