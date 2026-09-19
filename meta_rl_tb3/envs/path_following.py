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
        elif path_type == PathType.SINE_WAVE:
            return self._generate_sine_path()
        elif path_type == PathType.SQUARE:
            return self._generate_square_path()
        elif path_type == PathType.FIGURE_EIGHT:
            return self._generate_figure_eight_path()
        else:  # RANDOM_WAYPOINTS
            return self._generate_random_path()
    
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
    
    def _generate_sine_path(self) -> np.ndarray:
        """Generate a sinusoidal path."""
        amplitude = self.task_config.path_amplitude
        frequency = self.task_config.path_frequency
        arena = self.task_config.arena_size
        n = self.task_config.num_waypoints
        
        waypoints = []
        for i in range(n):
            x = -arena + 0.5 + (2 * arena - 1) * i / (n - 1)
            y = amplitude * np.sin(2 * np.pi * frequency * (x + arena) / (2 * arena))
            waypoints.append([x, y])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _generate_square_path(self) -> np.ndarray:
        """Generate a square path."""
        size = self.task_config.path_radius
        n = self.task_config.num_waypoints
        
        # Create square corners
        corners = [
            [-size, -size],
            [size, -size],
            [size, size],
            [-size, size],
        ]
        
        # Interpolate between corners
        waypoints = []
        points_per_side = max(1, n // 4)
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            for j in range(points_per_side):
                t = j / points_per_side
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                waypoints.append([x, y])
        
        return np.array(waypoints[:n], dtype=np.float32)
    
    def _generate_figure_eight_path(self) -> np.ndarray:
        """Generate a figure-eight path."""
        radius = self.task_config.path_radius / 2
        n = self.task_config.num_waypoints
        
        waypoints = []
        for i in range(n):
            t = 2 * np.pi * i / n
            x = radius * np.sin(t)
            y = radius * np.sin(2 * t) / 2
            waypoints.append([x, y])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _generate_random_path(self) -> np.ndarray:
        """Generate a random waypoint path."""
        arena = self.task_config.arena_size
        margin = 0.5
        n = self.task_config.num_waypoints
        
        # Start from robot position
        waypoints = [self._robot_position[:2].copy()]
        
        for _ in range(n - 1):
            # Sample next waypoint not too far from previous
            prev = waypoints[-1]
            for _ in range(50):  # Max attempts
                # Random direction and distance
                angle = np.random.uniform(-np.pi, np.pi)
                distance = np.random.uniform(0.5, 1.5)
                
                x = prev[0] + distance * np.cos(angle)
                y = prev[1] + distance * np.sin(angle)
                
                # Check bounds
                if -arena + margin < x < arena - margin and \
                   -arena + margin < y < arena - margin:
                    waypoints.append([x, y])
                    break
            else:
                # Fallback: move toward center
                direction = -prev / (np.linalg.norm(prev) + 1e-6)
                x = prev[0] + 0.5 * direction[0]
                y = prev[1] + 0.5 * direction[1]
                waypoints.append([x, y])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _compute_path_length(self) -> float:
        """Compute total path length."""
        if len(self._waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self._waypoints) - 1):
            diff = self._waypoints[i + 1] - self._waypoints[i]
            total += np.linalg.norm(diff)
        
        return total
    
    def _get_cross_track_error(self) -> float:
        # Compute cross-track error (perpendicular distance to path).
 
        if len(self._waypoints) < 2 or self._current_waypoint_idx >= len(self._waypoints):
            return 0.0
        
        robot_pos = self._robot_position[:2]
        
        # Get current path segment
        wp_idx = min(self._current_waypoint_idx, len(self._waypoints) - 2)
        p1 = self._waypoints[wp_idx]
        p2 = self._waypoints[min(wp_idx + 1, len(self._waypoints) - 1)]
        
        # Vector from p1 to p2
        path_vec = p2 - p1
        path_length = np.linalg.norm(path_vec)
        
        if path_length < 1e-6:
            return np.linalg.norm(robot_pos - p1)
        
        # Vector from p1 to robot
        to_robot = robot_pos - p1
        
        # Project onto path
        t = np.clip(np.dot(to_robot, path_vec) / (path_length ** 2), 0, 1)
        closest_point = p1 + t * path_vec
        
        # Cross-track error
        error = np.linalg.norm(robot_pos - closest_point)
        
        # Sign based on which side of path
        cross = path_vec[0] * to_robot[1] - path_vec[1] * to_robot[0]
        return error * np.sign(cross)
    
    def _get_heading_error(self) -> float:
        # Compute heading error relative to path direction.

        if len(self._waypoints) < 2 or self._current_waypoint_idx >= len(self._waypoints):
            return 0.0
        
        # Get path direction
        wp_idx = min(self._current_waypoint_idx, len(self._waypoints) - 2)
        p1 = self._waypoints[wp_idx]
        p2 = self._waypoints[min(wp_idx + 1, len(self._waypoints) - 1)]
        
        path_direction = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        robot_heading = self._robot_position[2]
        
        # Compute error
        error = path_direction - robot_heading
        error = np.arctan2(np.sin(error), np.cos(error))
        
        return error
    
    def _get_lookahead_waypoints(self) -> np.ndarray:

        n = self.task_config.lookahead_points
        robot_pos = self._robot_position[:2]
        robot_theta = self._robot_position[2]
        
        # Rotation matrix to robot frame
        cos_theta = np.cos(-robot_theta)
        sin_theta = np.sin(-robot_theta)
        
        waypoints_robot_frame = []
        
        for i in range(n):
            wp_idx = self._current_waypoint_idx + i
            
            if wp_idx < len(self._waypoints):
                wp = self._waypoints[wp_idx]
            elif self.task_config.loop_path and len(self._waypoints) > 0:
                wp = self._waypoints[wp_idx % len(self._waypoints)]
            else:
                # Use last waypoint
                wp = self._waypoints[-1] if len(self._waypoints) > 0 else robot_pos
            
            # Transform to robot frame
            dx = wp[0] - robot_pos[0]
            dy = wp[1] - robot_pos[1]
            
            x_robot = cos_theta * dx - sin_theta * dy
            y_robot = sin_theta * dx + cos_theta * dy
            
            # Normalize to reasonable range
            max_dist = self.task_config.arena_size * 2
            x_robot = np.clip(x_robot / max_dist, -1, 1)
            y_robot = np.clip(y_robot / max_dist, -1, 1)
            
            waypoints_robot_frame.extend([x_robot, y_robot])
        
        return np.array(waypoints_robot_frame, dtype=np.float32)
    
    def _update_waypoint_progress(self) -> None:
        if self._current_waypoint_idx >= len(self._waypoints):
            self._path_completed = True
            return
        
        robot_pos = self._robot_position[:2]
        current_wp = self._waypoints[self._current_waypoint_idx]
        
        distance = np.linalg.norm(robot_pos - current_wp)
        
        if distance < self.task_config.waypoint_threshold:
            self._current_waypoint_idx += 1
            
            if self._current_waypoint_idx >= len(self._waypoints):
                if self.task_config.loop_path:
                    self._current_waypoint_idx = 0
                else:
                    self._path_completed = True
    
    def _get_observation(self) -> dict:
        """Get observation including canonical task_info."""
        base_obs = super()._get_observation()
        base_obs["task_info"] = self._get_task_info()
        return base_obs

    def _get_task_info(self) -> np.ndarray:
        # Compute canonical task_info for path following (5 dims).

        L = self.task_config.arena_size

        # Cross-track error normalised to [-1, 1]
        cte = self._get_cross_track_error()
        norm_cte = float(np.clip(cte / L, -1.0, 1.0))

        # Heading error normalised to [-1, 1]
        he = self._get_heading_error()
        norm_he = float(np.clip(he / np.pi, -1.0, 1.0))

        # Progress ratio [0, 1]
        n_wps = max(1, len(self._waypoints) - 1)
        progress = float(np.clip(self._current_waypoint_idx / n_wps, 0.0, 1.0))

        # Next waypoint in robot frame, normalised by arena_size
        robot_pos   = self._robot_position[:2]
        robot_theta = self._robot_position[2]
        cos_t = float(np.cos(-robot_theta))
        sin_t = float(np.sin(-robot_theta))

        wp_idx = min(self._current_waypoint_idx, len(self._waypoints) - 1) \
            if len(self._waypoints) > 0 else 0
        if len(self._waypoints) > 0:
            wp = self._waypoints[wp_idx]
            dx_g = float(wp[0] - robot_pos[0])
            dy_g = float(wp[1] - robot_pos[1])
            dx_r = (cos_t * dx_g - sin_t * dy_g) / L
            dy_r = (sin_t * dx_g + cos_t * dy_g) / L
            dx_r = float(np.clip(dx_r, -1.0, 1.0))
            dy_r = float(np.clip(dy_r, -1.0, 1.0))
        else:
            dx_r = dy_r = 0.0

        return np.array([norm_cte, norm_he, progress, dx_r, dy_r],
                        dtype=np.float32)
    
    def _compute_reward(
        self, action: np.ndarray, observation: Dict[str, np.ndarray]
    ) -> float:
        """Compute reward for path following task.

        All components are normalised to a [-1, +1] per-step range so
        the total per-step reward stays in a reasonable band regardless
        of path shape or arena size.  This prevents the diverging
        negative rewards seen when cross-track errors accumulate.
        """
        reward = 0.0

        # ── Cross-track error penalty ──────────────────────────────
        # Normalise by arena_size so the penalty is always in [0, 1].
        cross_track = abs(self._get_cross_track_error())
        max_error = self.task_config.arena_size
        normalised_cte = min(cross_track / max_error, 1.0)
        reward -= self.task_config.cross_track_penalty_scale * normalised_cte

        # ── Heading error penalty ──────────────────────────────────
        # heading error ∈ [0, π], so dividing by π → [0, 1].
        heading_error = abs(self._get_heading_error())
        normalised_he = heading_error / np.pi
        reward -= self.task_config.heading_penalty_scale * normalised_he

        # ── Progress reward ────────────────────────────────────────
        # Reward forward progress regardless of lateral position so
        # the robot always gets a gradient toward the next waypoint.
        robot_pos = self._robot_position[:2]
        distance_moved = np.linalg.norm(robot_pos - self._previous_position)
        self._previous_position = robot_pos.copy()
        reward += self.task_config.progress_reward_scale * distance_moved

        # ── Waypoint reached bonus ─────────────────────────────────
        old_idx = self._current_waypoint_idx
        self._update_waypoint_progress()
        if self._current_waypoint_idx > old_idx:
            reward += 5.0

        # ── Path completion bonus ──────────────────────────────────
        if self._path_completed:
            reward += self.task_config.completion_reward

        # ── Collision penalty ──────────────────────────────────────
        if self._collision:
            reward += self.task_config.collision_penalty

        # ── Time penalty ───────────────────────────────────────────
        reward += self.task_config.time_penalty

        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        return self._collision or self._path_completed
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode information."""
        info = super()._get_info()
        
        info.update({
            "path": self._waypoints.tolist() if len(self._waypoints) > 0 else [],
            "current_waypoint_idx": self._current_waypoint_idx,
            "path_completed": self._path_completed,
            "cross_track_error": self._get_cross_track_error(),
            "heading_error": self._get_heading_error(),
            "progress": self._current_waypoint_idx / max(1, len(self._waypoints)),
            "is_success": self._path_completed and not self._collision,
        })
        
        return info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment with new path.
        
        Args:
            seed: Random seed
            options: Additional options including 'path', 'path_type'
            
        Returns:
            observation: Initial observation
            info: Episode information
        """
        # Reset base environment
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset path state
        self._current_waypoint_idx = 0
        self._path_completed = False
        self._distance_traveled = 0.0
        self._previous_position = self._robot_position[:2].copy()
        
        # Set path type if specified
        if options and "path_type" in options:
            self.task_config.path_type = PathType(options["path_type"])
        
        # Generate or set path
        if options and "path" in options:
            self._waypoints = np.array(options["path"], dtype=np.float32)
        else:
            self._waypoints = self._generate_path()
        
        # Compute path length
        self._total_path_length = self._compute_path_length()
        
        # Position robot at start of path
        if len(self._waypoints) > 0 and options is None:
            # Optionally start robot at first waypoint
            pass
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def set_path(self, waypoints: List[Tuple[float, float]]) -> None:

        self._waypoints = np.array(waypoints, dtype=np.float32)
        self._current_waypoint_idx = 0
        self._path_completed = False
        self._total_path_length = self._compute_path_length()


class FlatPathFollowingEnv(gym.ObservationWrapper):
    """Flatten path following observations to canonical 370-dim vector.

    Layout: lidar(360) + robot_state(5) + task_info(5) = 370 dims.
    """

    def __init__(self, env: PathFollowingEnv):
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


def make_path_following_env(
    config=None,
    flat_obs: bool = True,
    use_ros2: bool = False,
) -> gym.Env:
    env = PathFollowingEnv(config=config, use_ros2=use_ros2)
    if flat_obs:
        env = FlatPathFollowingEnv(env)
    return env
