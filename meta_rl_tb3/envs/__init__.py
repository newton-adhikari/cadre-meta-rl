"""Gymnasium environments for TurtleBot3 navigation tasks."""

from meta_rl_tb3.envs.physics import DynamicsConfig, sample_dynamics, DYNAMICS_RANGES
from meta_rl_tb3.envs.turtlebot3_env import TurtleBot3Env
from meta_rl_tb3.envs.goal_reaching import GoalReachingEnv
from meta_rl_tb3.envs.obstacle_avoidance import ObstacleAvoidanceEnv
from meta_rl_tb3.envs.path_following import PathFollowingEnv
from meta_rl_tb3.envs.wrappers import (
    CanonicalObsWrapper,
    PaddedObsWrapper,          # backward-compat alias
    CANONICAL_OBS_DIM,
    MIXED_OBS_DIM,             # backward-compat alias
)

__all__ = [
    "DynamicsConfig",
    "sample_dynamics",
    "DYNAMICS_RANGES",
    "TurtleBot3Env",
    "GoalReachingEnv",
    "ObstacleAvoidanceEnv",
    "PathFollowingEnv",
    "CanonicalObsWrapper",
    "PaddedObsWrapper",
    "CANONICAL_OBS_DIM",
    "MIXED_OBS_DIM",
]
