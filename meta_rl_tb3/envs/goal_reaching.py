"""Goal-reaching navigation task for TurtleBot3.

This module implements a goal-reaching environment where the robot
must navigate to a target position while avoiding collisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from meta_rl_tb3.envs.turtlebot3_env import TurtleBot3Env, TurtleBot3EnvConfig


@dataclass
class GoalReachingConfig(TurtleBot3EnvConfig):
    """Configuration for goal-reaching task.
    
    Attributes:
        goal_threshold: Distance threshold to consider goal reached
        goal_reward: Bonus reward for reaching the goal
        collision_penalty: Penalty for collisions
        distance_reward_scale: Scale factor for distance-based reward
        time_penalty: Small penalty per step to encourage efficiency
        min_goal_distance: Minimum initial distance to goal
        max_goal_distance: Maximum initial distance to goal
        goal_sampling_area: Area bounds for goal sampling [(x_min, x_max), (y_min, y_max)]
        fixed_goal: If set, use this fixed goal position (x, y)
        fixed_start: If set, use this fixed start position (x, y, theta)
    """
    goal_threshold: float = 0.25
    goal_reward: float = 10.0  # Reduced from 100 for consistent scale across tasks
    collision_penalty: float = -10.0  # Reduced from -50
    distance_reward_scale: float = 1.0  # Reduced from 10.0
    time_penalty: float = -0.01  # Reduced from -0.1
    min_goal_distance: float = 1.0
    max_goal_distance: float = 3.0
    goal_sampling_area: Optional[List[Tuple[float, float]]] = None
    fixed_goal: Optional[Tuple[float, float]] = None
    fixed_start: Optional[Tuple[float, float, float]] = None
