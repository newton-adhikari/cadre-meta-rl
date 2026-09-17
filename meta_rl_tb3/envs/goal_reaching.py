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


class GoalReachingEnv(TurtleBot3Env):
    """Goal-reaching navigation environment for TurtleBot3.
    
    The robot must navigate to a randomly sampled goal position while
    avoiding obstacles (walls and potentially other objects). The episode
    ends when the goal is reached, the robot collides, or time runs out.
    
    Observation Space:
        - lidar: LiDAR scan distances (normalized)
        - robot_state: [x, y, theta, linear_vel, angular_vel] (normalized)
        - goal: [distance_to_goal, angle_to_goal] (relative polar coordinates)
    
    Action Space:
        - Continuous: [linear_velocity, angular_velocity] in [-1, 1]
    
    Reward:
        - Dense: Improvement in distance to goal
        - Sparse: Bonus for reaching goal, penalty for collision
        - Time: Small penalty per step
    
    """
    
    def __init__(
        self,
        config: Optional[GoalReachingConfig] = None,
        render_mode: Optional[str] = None,
        use_ros2: bool = False,
    ):
        
        self.task_config = config or GoalReachingConfig()
        super().__init__(self.task_config, render_mode, use_ros2)
        
        # Goal state
        self._goal_position = np.zeros(2, dtype=np.float32)
        self._previous_distance = 0.0
        self._goal_reached = False
        
        # Extend observation space to include goal
        self._setup_goal_observation_space()
    
    def _setup_goal_observation_space(self) -> None:
        """Extend observation space to include goal information.

        """
        task_info_low  = np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        task_info_high = np.array([1.0,  1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "lidar":       self.observation_space["lidar"],
            "robot_state": self.observation_space["robot_state"],
            "task_info":   spaces.Box(
                low=task_info_low, high=task_info_high, dtype=np.float32
            ),
        })
    
    def _sample_goal(self) -> np.ndarray:
        if self.task_config.fixed_goal is not None:
            return np.array(self.task_config.fixed_goal, dtype=np.float32)
        
        # Get robot position
        robot_x, robot_y = self._robot_position[:2]
        
        # Determine sampling area
        if self.task_config.goal_sampling_area is not None:
            x_range, y_range = self.task_config.goal_sampling_area
        else:
            margin = 0.5
            arena = self.task_config.arena_size
            x_range = (-arena + margin, arena - margin)
            y_range = (-arena + margin, arena - margin)
        
        # Sample goal at appropriate distance from robot
        for _ in range(100):  # Max attempts
            goal_x = np.random.uniform(x_range[0], x_range[1])
            goal_y = np.random.uniform(y_range[0], y_range[1])
            
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            if (self.task_config.min_goal_distance <= distance <= 
                self.task_config.max_goal_distance):
                return np.array([goal_x, goal_y], dtype=np.float32)
        
        # Fallback: place goal at max distance in random direction
        angle = np.random.uniform(-np.pi, np.pi)
        dist = self.task_config.max_goal_distance
        goal_x = robot_x + dist * np.cos(angle)
        goal_y = robot_y + dist * np.sin(angle)
        
        # Clip to arena bounds
        margin = 0.3
        arena = self.task_config.arena_size
        goal_x = np.clip(goal_x, -arena + margin, arena - margin)
        goal_y = np.clip(goal_y, -arena + margin, arena - margin)
        
        return np.array([goal_x, goal_y], dtype=np.float32)
    
    def _get_goal_observation(self) -> np.ndarray:
        """Compute canonical task_info for goal_reaching (5 dims).

        """
        dx = self._goal_position[0] - self._robot_position[0]
        dy = self._goal_position[1] - self._robot_position[1]

        distance      = float(np.sqrt(dx ** 2 + dy ** 2))
        global_angle  = float(np.arctan2(dy, dx))
        relative_angle = global_angle - float(self._robot_position[2])
        relative_angle = float(np.arctan2(
            np.sin(relative_angle), np.cos(relative_angle)
        ))

        # arena diagonal = arena_size * √2
        diag = self.task_config.arena_size * np.sqrt(2.0)
        norm_dist  = float(np.clip(distance / diag, 0.0, 1.0))
        norm_angle = float(relative_angle / np.pi)

        return np.array(
            [norm_dist, norm_angle, 0.0, 0.0, 0.0], dtype=np.float32
        )
    
    def _get_observation(self) -> dict:
        """Get full observation including canonical task_info."""
        base_obs = super()._get_observation()
        base_obs["task_info"] = self._get_goal_observation()
        return base_obs
    
    def _compute_reward(
        self, action: np.ndarray, observation: Dict[str, np.ndarray]
    ) -> float:
        """
        Reward components:
        1. Distance improvement (dense)
        2. Goal reached bonus (sparse)
        3. Collision penalty (sparse)
        4. Time penalty (dense)
        
        """
        reward = 0.0
        
        # Compute current distance to goal
        dx = self._goal_position[0] - self._robot_position[0]
        dy = self._goal_position[1] - self._robot_position[1]
        current_distance = np.sqrt(dx**2 + dy**2)
        
        # Dense reward: improvement in distance
        distance_improvement = self._previous_distance - current_distance
        reward += distance_improvement * self.task_config.distance_reward_scale
        
        # Update previous distance
        self._previous_distance = current_distance
        
        # Goal reached bonus
        if current_distance < self.task_config.goal_threshold:
            reward += self.task_config.goal_reward
            self._goal_reached = True
        
        # Collision penalty
        if self._collision:
            reward += self.task_config.collision_penalty
        
        # Time penalty
        reward += self.task_config.time_penalty
        
        return reward
    
    def _is_terminated(self) -> bool:
        return self._goal_reached or self._collision
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional episode information."""
        info = super()._get_info()
        
        # Add goal-specific info
        dx = self._goal_position[0] - self._robot_position[0]
        dy = self._goal_position[1] - self._robot_position[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        info.update({
            "goal_position": self._goal_position.copy(),
            "distance_to_goal": distance_to_goal,
            "goal_reached": self._goal_reached,
            "is_success": self._goal_reached,
        })
        
        return info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment with new goal.
        
        """
        # Handle fixed start position
        if options is None:
            options = {}
        
        if self.task_config.fixed_start is not None and "position" not in options:
            options["position"] = self.task_config.fixed_start[:2]
            if "orientation" not in options:
                options["orientation"] = self.task_config.fixed_start[2]
        
        # Reset base environment
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset goal state
        self._goal_reached = False
        
        # Sample or set goal
        if options and "goal_position" in options:
            self._goal_position = np.array(options["goal_position"], dtype=np.float32)
        else:
            self._goal_position = self._sample_goal()
        
        # Initialize previous distance
        dx = self._goal_position[0] - self._robot_position[0]
        dy = self._goal_position[1] - self._robot_position[1]
        self._previous_distance = np.sqrt(dx**2 + dy**2)
        
        # Get full observation with goal
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def set_goal(self, goal_position: tuple) -> None:
        """Set the goal position explicitly."""
        self._goal_position = np.array(goal_position, dtype=np.float32)
        dx = self._goal_position[0] - self._robot_position[0]
        dy = self._goal_position[1] - self._robot_position[1]
        self._previous_distance = float(np.sqrt(dx ** 2 + dy ** 2))

    def set_dynamics(self, dynamics) -> None:
        """Attach a DynamicsConfig to this environment (delegates to base)."""
        super().set_dynamics(dynamics)  # TurtleBot3Env.set_dynamics

    def get_goal_position(self) -> np.ndarray:
        """Get current goal position."""
        return self._goal_position.copy()


class FlatGoalReachingEnv(gym.ObservationWrapper):
    """Flatten GoalReachingEnv observations to the canonical 370-dim vector.

    Layout: lidar(360) + robot_state(5) + task_info(5) = 370 dims.
    """

    def __init__(self, env: GoalReachingEnv):
        super().__init__(env)
        lidar_size     = env.observation_space["lidar"].shape[0]       # 360
        state_size     = env.observation_space["robot_state"].shape[0] # 5
        task_info_size = env.observation_space["task_info"].shape[0]   # 5
        total          = lidar_size + state_size + task_info_size      # 370
        assert total == 370, f"Expected 370 dims, got {total}"
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(total,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        """Concatenate lidar, robot_state, task_info → (370,) array."""
        return np.concatenate([
            obs["lidar"], obs["robot_state"], obs["task_info"]
        ])


def make_goal_reaching_env(
    config=None,
    flat_obs: bool = True,
    use_ros2: bool = False,
) -> gym.Env:
    """Factory function to create goal-reaching environment."""
    env = GoalReachingEnv(config=config, use_ros2=use_ros2)
    if flat_obs:
        env = FlatGoalReachingEnv(env)
    return env
