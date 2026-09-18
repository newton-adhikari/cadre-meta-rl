"""Path following navigation task for TurtleBot3.

This module implements a path following environment where the robot
must follow a reference trajectory defined by waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from meta_rl_tb3.envs.turtlebot3_env import TurtleBot3Env, TurtleBot3EnvConfig


class PathType(Enum):
    """Types of paths that can be generated."""
    STRAIGHT = "straight"
    CIRCULAR = "circular"
    SINE_WAVE = "sine_wave"
    SQUARE = "square"
    FIGURE_EIGHT = "figure_eight"
    RANDOM_WAYPOINTS = "random_waypoints"
    CUSTOM = "custom"


@dataclass
class PathFollowingConfig(TurtleBot3EnvConfig):
    """Configuration for path following task.
    
    Attributes:
        path_type: Type of path to generate
        num_waypoints: Number of waypoints in path
        waypoint_threshold: Distance to consider waypoint reached
        lookahead_points: Number of future waypoints in observation
        cross_track_penalty_scale: Penalty scale for cross-track error
        heading_penalty_scale: Penalty scale for heading error
        progress_reward_scale: Reward scale for path progress
        completion_reward: Bonus for completing the path
        collision_penalty: Penalty for collision
        time_penalty: Penalty per time step
        loop_path: Whether path should loop
        path_radius: Radius for circular paths
        path_amplitude: Amplitude for sine wave paths
        path_frequency: Frequency for sine wave paths
        custom_waypoints: Custom waypoint list [(x, y), ...]
    """
    path_type: PathType = PathType.RANDOM_WAYPOINTS
    num_waypoints: int = 10
    waypoint_threshold: float = 0.3
    lookahead_points: int = 5
    cross_track_penalty_scale: float = 0.5  # Reduced from 2.0
    heading_penalty_scale: float = 0.2  # Reduced from 1.0
    progress_reward_scale: float = 0.5   # Was 5.0 — path-following was 5-10x goal-reaching scale
    completion_reward: float = 10.0
    collision_penalty: float = -10.0
    time_penalty: float = -0.01  # Reduced from -0.05
    loop_path: bool = False
    path_radius: float = 2.0
    path_amplitude: float = 1.5
    path_frequency: float = 1.0
    custom_waypoints: Optional[List[Tuple[float, float]]] = None


class PathFollowingEnv(TurtleBot3Env):
    """Path following navigation environment for TurtleBot3.
    
    The robot must follow a reference path defined by waypoints.
    The task rewards staying close to the path and making progress.
    
    Observation Space:
        - lidar: LiDAR scan distances (normalized)
        - robot_state: [x, y, theta, linear_vel, angular_vel] (normalized)
        - path: Next N waypoints in robot frame (flattened)
        - path_progress: [current_waypoint_idx / total, cross_track_error]
    
    Action Space:
        - Continuous: [linear_velocity, angular_velocity] in [-1, 1]
    
    Reward:
        - Cross-track error: Negative reward for deviation from path
        - Heading alignment: Reward for facing along path direction
        - Progress: Reward for advancing along path
        - Completion: Bonus for completing the path
    """
    
    def __init__(
        self,
        config: Optional[PathFollowingConfig] = None,
        render_mode: Optional[str] = None,
        use_ros2: bool = False,
    ):
        """Initialize path following environment."""
        self.task_config = config or PathFollowingConfig()
        super().__init__(self.task_config, render_mode, use_ros2)
        
        # Path state
        self._waypoints: np.ndarray = np.array([])
        self._current_waypoint_idx: int = 0
        self._path_completed: bool = False
        self._total_path_length: float = 0.0
        self._distance_traveled: float = 0.0
        self._previous_position: np.ndarray = np.zeros(2)
        
        # Setup observation space
        self._setup_path_observation_space()
    
    def _setup_path_observation_space(self) -> None:

        task_info_low  = np.array([-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        task_info_high = np.array([ 1.0,  1.0, 1.0,  1.0,  1.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "lidar":       self.observation_space["lidar"],
            "robot_state": self.observation_space["robot_state"],
            "task_info":   spaces.Box(
                low=task_info_low, high=task_info_high, dtype=np.float32
            ),
        })
    
    def _generate_path(self) -> np.ndarray:
        """Generate path based on configuration.
        
        """
        path_type = self.task_config.path_type
        
        if path_type == PathType.CUSTOM and self.task_config.custom_waypoints:
            return np.array(self.task_config.custom_waypoints, dtype=np.float32)
        
        if path_type == PathType.STRAIGHT:
            return self._generate_straight_path()
        elif path_type == PathType.CIRCULAR:
            return self._generate_circular_path()



    def _generate_straight_path(self) -> np.ndarray:
        """Generate a straight line path."""
        robot_pos = self._robot_position[:2]
        robot_theta = self._robot_position[2]
        
        # Path in front of robot
        waypoints = []
        distance = 0.5
        for i in range(self.task_config.num_waypoints):
            x = robot_pos[0] + distance * np.cos(robot_theta)
            y = robot_pos[1] + distance * np.sin(robot_theta)
            waypoints.append([x, y])
            distance += 0.5
        
        return np.array(waypoints, dtype=np.float32)

    def _generate_circular_path(self) -> np.ndarray:
        radius = self.task_config.path_radius
        n = self.task_config.num_waypoints
        
        # Center the circle in the arena
        center_x, center_y = 0.0, 0.0
        
        waypoints = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoints.append([x, y])
        
        return np.array(waypoints, dtype=np.float32)
        
