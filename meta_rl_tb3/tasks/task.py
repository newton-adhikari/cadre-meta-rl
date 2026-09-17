"""Task dataclass and configuration for meta-learning.

This module defines the Task abstraction used for meta-learning,
including task configurations and utilities for task management.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np


class TaskType(Enum):
    """Types of navigation tasks."""
    GOAL_REACHING = "goal_reaching"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    PATH_FOLLOWING = "path_following"
    COMBINED = "combined"


@dataclass
class TaskConfig:
    """Configuration for a specific task instance.
    
    Attributes:
        task_type: Type of navigation task
        task_id: Unique identifier for this task
        
        # Environment parameters
        arena_size: Size of the arena
        max_episode_steps: Maximum steps per episode
        
        # Goal-reaching parameters
        goal_position: Fixed goal position (x, y)
        goal_threshold: Distance to consider goal reached
        goal_reward: Reward for reaching goal
        
        # Start position
        start_position: Fixed start position (x, y)
        start_orientation: Fixed start orientation (theta)
        
        # Obstacle parameters
        obstacles: List of obstacle configurations
        num_random_obstacles: Number of random obstacles to add
        
        # Path following parameters
        path_waypoints: List of waypoint positions
        path_type: Type of path ('straight', 'circular', etc.)
        
        # Reward weights (for customization)
        reward_weights: Dictionary of reward component weights
        
        # Difficulty estimation
        difficulty: Estimated task difficulty (0-1)
        
        # Metadata
        metadata: Additional task metadata
    """
    task_type: TaskType = TaskType.GOAL_REACHING
    task_id: Optional[str] = None
    
    # Environment parameters
    arena_size: float = 4.0
    max_episode_steps: int = 200  # Reduced from 500 to match MAML training config
    
    # Goal-reaching parameters
    goal_position: Optional[Tuple[float, float]] = None
    goal_threshold: float = 0.25
    goal_reward: float = 10.0   # Fixed: was 100.0, inconsistent with GoalReachingConfig default
    
    # Start position
    start_position: Optional[Tuple[float, float]] = None
    start_orientation: Optional[float] = None
    
    # Obstacle parameters
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    num_random_obstacles: int = 0
    
    # Path following parameters
    path_waypoints: Optional[List[Tuple[float, float]]] = None
    path_type: str = "random"
    
    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)
    
    # Difficulty
    difficulty: float = 0.5
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamics perturbation (None = nominal)
    dynamics: Optional[Any] = None  # DynamicsConfig | None
    
    def __post_init__(self):
        """Generate task ID if not provided."""
        if self.task_id is None:
            self.task_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on task configuration."""
        # Create a hash from key parameters
        key_params = (
            self.task_type.value,
            self.goal_position,
            self.start_position,
            len(self.obstacles),
            self.difficulty,
        )
        hash_input = str(key_params).encode()
        return hashlib.md5(hash_input).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['task_type'] = self.task_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """Create from dictionary.
        
        Args:
            data: Dictionary produced by ``to_dict`` or loaded from JSON.
            
        Raises:
            KeyError: If a required field is missing.
            ValueError: If a field value has an invalid type or value.
            
        Returns:
            TaskConfig instance.
        """
        data = data.copy()
        
        # Convert task_type string to enum
        if 'task_type' in data:
            raw = data['task_type']
            try:
                data['task_type'] = TaskType(raw)
            except ValueError:
                valid = [e.value for e in TaskType]
                raise ValueError(
                    f"Invalid task_type '{raw}'. Must be one of: {valid}"
                )
        
        # Validate numeric fields are in range
        if 'difficulty' in data:
            d = data['difficulty']
            if not isinstance(d, (int, float)):
                raise ValueError(f"difficulty must be numeric, got {type(d).__name__}")
            if not (0.0 <= float(d) <= 1.0):
                raise ValueError(f"difficulty must be in [0, 1], got {d}")
        
        if 'arena_size' in data:
            a = data['arena_size']
            if not isinstance(a, (int, float)) or float(a) <= 0:
                raise ValueError(f"arena_size must be a positive number, got {a!r}")
        
        if 'max_episode_steps' in data:
            s = data['max_episode_steps']
            if not isinstance(s, int) or s <= 0:
                raise ValueError(
                    f"max_episode_steps must be a positive integer, got {s!r}"
                )
        
        # Convert tuple fields that may have been serialised as lists
        for field_name in ('goal_position', 'start_position'):
            if field_name in data and data[field_name] is not None:
                val = data[field_name]
                if isinstance(val, (list, tuple)):
                    if len(val) != 2:
                        raise ValueError(
                            f"{field_name} must have exactly 2 elements, got {len(val)}"
                        )
                    data[field_name] = tuple(float(v) for v in val)
                else:
                    raise ValueError(
                        f"{field_name} must be a 2-element list/tuple, got {type(val).__name__}"
                    )
        
        if 'path_waypoints' in data and data['path_waypoints'] is not None:
            wps = data['path_waypoints']
            if not isinstance(wps, (list, tuple)):
                raise ValueError(
                    f"path_waypoints must be a list of (x, y) pairs, got {type(wps).__name__}"
                )
            data['path_waypoints'] = [tuple(float(v) for v in wp) for wp in wps]
        
        # Drop unknown keys so the dataclass constructor doesn't choke
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - valid_fields
        if unknown:
            import warnings
            warnings.warn(
                f"TaskConfig.from_dict: ignoring unknown fields: {sorted(unknown)}",
                stacklevel=2,
            )
            data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save task configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TaskConfig':
        """Load task configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Task:
    """Represents a task for meta-learning.
    
    A task combines a configuration with environment setup functions
    and provides utilities for task management.
    
    Example:
        >>> config = TaskConfig(task_type=TaskType.GOAL_REACHING, goal_position=(2.0, 2.0))
        >>> task = Task(config)
        >>> env = task.create_env()
    """
    
    config: TaskConfig
    env_fn: Optional[Callable] = None
    
    def __post_init__(self):
        """Set up default environment function if not provided."""
        if self.env_fn is None:
            self.env_fn = self._default_env_fn
    
    def _default_env_fn(self):
        """Default environment creation function.

        Returns a flat Gymnasium environment whose observation space is
        exactly ``CANONICAL_OBS_DIM`` (370) dims.  DynamicsConfig is
        applied if set on this task's config.
        """
        from meta_rl_tb3.envs import GoalReachingEnv, ObstacleAvoidanceEnv, PathFollowingEnv
        from meta_rl_tb3.envs.goal_reaching import GoalReachingConfig, FlatGoalReachingEnv
        from meta_rl_tb3.envs.obstacle_avoidance import (
            ObstacleAvoidanceConfig, FlatObstacleAvoidanceEnv,
        )
        from meta_rl_tb3.envs.path_following import PathFollowingConfig, FlatPathFollowingEnv
        from meta_rl_tb3.envs.wrappers import CanonicalObsWrapper, CANONICAL_OBS_DIM

        dyn = self.config.dynamics   # DynamicsConfig or None

        if self.config.task_type == TaskType.GOAL_REACHING:
            env_config = GoalReachingConfig(
                arena_size=self.config.arena_size,
                max_episode_steps=self.config.max_episode_steps,
                goal_threshold=self.config.goal_threshold,
                goal_reward=self.config.goal_reward,
                fixed_goal=self.config.goal_position,
                fixed_start=(
                    (*self.config.start_position, self.config.start_orientation or 0.0)
                    if self.config.start_position else None
                ),
                dynamics=dyn,
            )
            env = FlatGoalReachingEnv(GoalReachingEnv(config=env_config, use_ros2=False))

        elif self.config.task_type == TaskType.OBSTACLE_AVOIDANCE:
            env_config = ObstacleAvoidanceConfig(
                arena_size=self.config.arena_size,
                max_episode_steps=self.config.max_episode_steps,
                fixed_obstacles=self.config.obstacles if self.config.obstacles else None,
                num_obstacles=self.config.num_random_obstacles,
                include_goal=self.config.goal_position is not None,
                goal_position=self.config.goal_position,
                dynamics=dyn,
            )
            env = FlatObstacleAvoidanceEnv(
                ObstacleAvoidanceEnv(config=env_config, use_ros2=False)
            )

        elif self.config.task_type == TaskType.PATH_FOLLOWING:
            env_config = PathFollowingConfig(
                arena_size=self.config.arena_size,
                max_episode_steps=self.config.max_episode_steps,
                custom_waypoints=self.config.path_waypoints,
                dynamics=dyn,
            )
            env = FlatPathFollowingEnv(
                PathFollowingEnv(config=env_config, use_ros2=False)
            )

        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}")

        # Enforce canonical dimension (no-op for well-formed envs, safety net otherwise)
        if env.observation_space.shape[0] != CANONICAL_OBS_DIM:
            env = CanonicalObsWrapper(env, target_dim=CANONICAL_OBS_DIM)
        return env
    
    def create_env(self):
        """Create an environment instance for this task.
        
        Returns:
            Gymnasium environment configured for this task
        """
        return self.env_fn()
    
    @property
    def task_id(self) -> str:
        """Get task ID."""
        return self.config.task_id
    
    @property
    def task_type(self) -> TaskType:
        """Get task type."""
        return self.config.task_type
    
    @property
    def difficulty(self) -> float:
        """Get estimated task difficulty."""
        return self.config.difficulty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return self.config.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        config = TaskConfig.from_dict(data)
        return cls(config=config)
    
    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, type={self.task_type.value}, difficulty={self.difficulty:.2f})"


def estimate_task_difficulty(config: TaskConfig) -> float:
    """Estimate the difficulty of a task.
    
    Difficulty is based on:
    - Distance from start to goal
    - Number of obstacles
    - Path complexity
    - Arena size
    
    Args:
        config: Task configuration
        
    Returns:
        Difficulty score in [0, 1]
    """
    difficulty = 0.5  # Base difficulty
    
    # Goal distance factor
    if config.goal_position is not None and config.start_position is not None:
        start = np.array(config.start_position)
        goal = np.array(config.goal_position)
        distance = np.linalg.norm(goal - start)
        # Normalize by arena size
        normalized_distance = distance / (config.arena_size * 2)
        difficulty += 0.2 * normalized_distance
    
    # Obstacle factor
    num_obstacles = len(config.obstacles) + config.num_random_obstacles
    obstacle_factor = min(1.0, num_obstacles / 10)
    difficulty += 0.2 * obstacle_factor
    
    # Path complexity factor (for path following)
    if config.path_waypoints is not None:
        num_waypoints = len(config.path_waypoints)
        path_factor = min(1.0, num_waypoints / 20)
        difficulty += 0.1 * path_factor
    
    # Clip to [0, 1]
    return float(np.clip(difficulty, 0.0, 1.0))


def create_task_batch(
    tasks: List[Task],
) -> List[Callable]:
    """Create a batch of task environment functions.
    
    Args:
        tasks: List of Task objects
        
    Returns:
        List of environment creation functions
    """
    return [task.create_env for task in tasks]
