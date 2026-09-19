"""Reinforcement learning algorithms for Meta-RL-TB3."""

from meta_rl_tb3.algos.networks import (
    MLP,
    ActorNetwork,
    CriticNetwork,
    ActorCritic,
    # CADRE context encoder
    ContextEncoder,
    ContextEncoderConfig,
    # HELM components (retained for backward compat)
    TaskEncoder,
    TaskEncoderConfig,
    HELMActorCritic,
    HELM,
    create_helm,
)
from meta_rl_tb3.algos.ppo import PPO, PPOConfig, RolloutBuffer
from meta_rl_tb3.algos.maml import MAML, MAMLConfig
from meta_rl_tb3.algos.cadre import CADRE, CADREConfig

__all__ = [
    "MLP",
    "ActorNetwork",
    "CriticNetwork",
    "ActorCritic",
    # CADRE
    "ContextEncoder",
    "ContextEncoderConfig",
    "CADRE",
    "CADREConfig",
    # HELM
    "TaskEncoder",
    "TaskEncoderConfig",
    "HELMActorCritic",
    "HELM",
    "create_helm",
    # PPO
    "PPO",
    "PPOConfig",
    "RolloutBuffer",
    # MAML
    "MAML",
    "MAMLConfig",
]
