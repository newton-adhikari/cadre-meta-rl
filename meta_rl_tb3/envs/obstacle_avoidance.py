"""Obstacle avoidance navigation task for TurtleBot3.

This module implements an obstacle avoidance environment where the robot
must navigate through an environment with static obstacles while optionally
reaching a goal position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from meta_rl_tb3.envs.turtlebot3_env import TurtleBot3Env, TurtleBot3EnvConfig

# ROS2 imports for obstacle spawning
try:
    from gazebo_msgs.srv import SpawnEntity, DeleteEntity
    from geometry_msgs.msg import Pose
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


@dataclass
class Obstacle:
    """Represents a static obstacle in the environment.
    
    Attributes:
        position: (x, y) position of obstacle center
        size: (width, height, depth) dimensions
        obstacle_type: Type of obstacle ('box', 'cylinder', 'wall')
        name: Unique name for Gazebo entity
    """
    position: Tuple[float, float]
    size: Tuple[float, float, float] = (0.3, 0.3, 0.5)
    obstacle_type: str = "box"
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"obstacle_{id(self)}"


@dataclass 
class ObstacleAvoidanceConfig(TurtleBot3EnvConfig):
    """Configuration for obstacle avoidance task.
    
    Attributes:
        collision_penalty: Penalty for colliding with obstacles
        progress_reward_scale: Reward scale for forward progress
        safety_reward_scale: Reward scale for maintaining safe distance
        time_penalty: Penalty per time step
        min_obstacle_distance: Minimum safe distance from obstacles
        num_obstacles: Number of random obstacles to spawn
        obstacle_size_range: (min_size, max_size) for random obstacles
        fixed_obstacles: List of fixed obstacle configurations
        include_goal: Whether to include goal-reaching component
        goal_reward: Bonus for reaching goal (if include_goal)
        goal_threshold: Distance to consider goal reached
        goal_position: Fixed goal position (if include_goal)
        obstacle_types: List of allowed obstacle types
    """
    collision_penalty: float = -10.0  # Reduced from -50 for consistent scale
    progress_reward_scale: float = 0.1  # Reduced from 1.0
    safety_reward_scale: float = 0.1  # Reduced from 0.5
    time_penalty: float = -0.01  # Reduced from -0.1
    min_obstacle_distance: float = 0.3
    num_obstacles: int = 5
    obstacle_size_range: Tuple[float, float] = (0.2, 0.5)
    fixed_obstacles: Optional[List[Dict[str, Any]]] = None
    include_goal: bool = True
    goal_reward: float = 10.0  # Reduced from 100
    goal_threshold: float = 0.25
    goal_position: Optional[Tuple[float, float]] = None
    obstacle_types: List[str] = field(default_factory=lambda: ["box", "cylinder"])


class ObstacleAvoidanceEnv(TurtleBot3Env):
    """Obstacle avoidance navigation environment for TurtleBot3.
    
    The robot must navigate through an environment with static obstacles.
    Optionally includes goal-reaching component for combined task.
    
    Observation Space:
        - lidar: LiDAR scan distances (normalized) - shows obstacles
        - robot_state: [x, y, theta, linear_vel, angular_vel] (normalized)
        - goal: [distance, angle] (if include_goal is True)
    
    Action Space:
        - Continuous: [linear_velocity, angular_velocity] in [-1, 1]
    
    Reward:
        - Safety: Reward for maintaining distance from obstacles
        - Progress: Forward motion reward (and goal approach if include_goal)
        - Collision: Large penalty for hitting obstacles
        - Time: Small penalty per step
    
    """
    
    def __init__(
        self,
        config: Optional[ObstacleAvoidanceConfig] = None,
        render_mode: Optional[str] = None,
        use_ros2: bool = False,
    ):
        """Initialize obstacle avoidance environment.
        
        Args:
            config: Task configuration
            render_mode: Rendering mode
            use_ros2: Whether to use ROS2/Gazebo
        """
        self.task_config = config or ObstacleAvoidanceConfig()
        super().__init__(self.task_config, render_mode, use_ros2)
        
        # Obstacle state
        self._obstacles: List[Obstacle] = []
        self._spawned_obstacle_names: List[str] = []
        
        # Goal state (if included)
        self._goal_position = np.zeros(2, dtype=np.float32)
        self._goal_reached = False
        self._previous_distance = 0.0
        
        # Gazebo service clients
        self._spawn_client = None
        self._delete_client = None
        
        if self.use_ros2 and self._node is not None:
            self._setup_obstacle_services()
        
        # Setup observation space
        self._setup_obstacle_observation_space()
    
    def _setup_obstacle_services(self) -> None:
        """Set up Gazebo services for obstacle management."""
        self._spawn_client = self._node.create_client(
            SpawnEntity, "/spawn_entity"
        )
        self._delete_client = self._node.create_client(
            DeleteEntity, "/delete_entity"
        )
    
    def _setup_obstacle_observation_space(self) -> None:
        """Set up canonical observation space for obstacle avoidance.

        task_info layout (5 dims):
            [dist_to_goal/diag, angle_to_goal/π, min_lidar_norm, 0.0, 0.0]
        where min_lidar_norm is the minimum normalised LiDAR reading
        (proxy for nearest obstacle distance).
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
    
    def _generate_obstacle_sdf(self, obstacle: Obstacle) -> str:
        """Generate SDF string for spawning obstacle in Gazebo.
        
        Args:
            obstacle: Obstacle configuration
            
        Returns:
            SDF XML string
        """
        w, h, d = obstacle.size
        
        if obstacle.obstacle_type == "box":
            geometry = f"""
            <geometry>
                <box><size>{w} {h} {d}</size></box>
            </geometry>
            """
        elif obstacle.obstacle_type == "cylinder":
            radius = w / 2
            geometry = f"""
            <geometry>
                <cylinder>
                    <radius>{radius}</radius>
                    <length>{d}</length>
                </cylinder>
            </geometry>
            """
        else:
            # Default to box
            geometry = f"""
            <geometry>
                <box><size>{w} {h} {d}</size></box>
            </geometry>
            """
        
        sdf = f"""
        <?xml version="1.0"?>
        <sdf version="1.6">
            <model name="{obstacle.name}">
                <static>true</static>
                <link name="link">
                    <collision name="collision">
                        {geometry}
                    </collision>
                    <visual name="visual">
                        {geometry}
                        <material>
                            <ambient>0.5 0.5 0.5 1</ambient>
                            <diffuse>0.5 0.5 0.5 1</diffuse>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>
        """
        return sdf
    
    def _spawn_obstacle(self, obstacle: Obstacle) -> bool:
        """Spawn an obstacle in Gazebo.
        
        Args:
            obstacle: Obstacle to spawn
            
        Returns:
            True if successful
        """
        if not self.use_ros2 or self._spawn_client is None:
            return False
        
        if not self._spawn_client.wait_for_service(timeout_sec=1.0):
            return False
        
        request = SpawnEntity.Request()
        request.name = obstacle.name
        request.xml = self._generate_obstacle_sdf(obstacle)
        request.initial_pose = Pose()
        request.initial_pose.position.x = obstacle.position[0]
        request.initial_pose.position.y = obstacle.position[1]
        request.initial_pose.position.z = obstacle.size[2] / 2  # Half height
        
        future = self._spawn_client.call_async(request)
        # Don't wait for response to avoid blocking
        
        self._spawned_obstacle_names.append(obstacle.name)
        return True
    
    def _delete_obstacle(self, name: str) -> bool:
        """Delete an obstacle from Gazebo.
        
        Args:
            name: Name of obstacle to delete
            
        Returns:
            True if successful
        """
        if not self.use_ros2 or self._delete_client is None:
            return False
        
        if not self._delete_client.wait_for_service(timeout_sec=1.0):
            return False
        
        request = DeleteEntity.Request()
        request.name = name
        
        future = self._delete_client.call_async(request)
        return True
    
    def _clear_all_obstacles(self) -> None:
        """Remove all spawned obstacles from Gazebo."""
        for name in self._spawned_obstacle_names:
            self._delete_obstacle(name)
        
        self._spawned_obstacle_names.clear()
        self._obstacles.clear()
        
        # Give Gazebo time to process deletions
        if self.use_ros2:
            time.sleep(0.5)
    
    def _sample_obstacles(self) -> List[Obstacle]:
        """Sample random obstacles for the environment.
        
        Returns:
            List of obstacle configurations
        """
        obstacles = []
        
        # Add fixed obstacles if configured
        if self.task_config.fixed_obstacles:
            for obs_config in self.task_config.fixed_obstacles:
                obstacles.append(Obstacle(
                    position=obs_config.get("position", (0, 0)),
                    size=obs_config.get("size", (0.3, 0.3, 0.5)),
                    obstacle_type=obs_config.get("type", "box"),
                    name=obs_config.get("name"),
                ))
        
        # Add random obstacles
        arena = self.task_config.arena_size
        margin = 0.5
        robot_pos = self._robot_position[:2]
        
        for i in range(self.task_config.num_obstacles):
            # Sample position away from robot
            for _ in range(50):  # Max attempts
                x = np.random.uniform(-arena + margin, arena - margin)
                y = np.random.uniform(-arena + margin, arena - margin)
                
                # Check distance from robot
                dist_to_robot = np.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
                
                # Check distance from other obstacles
                min_obs_dist = float('inf')
                for obs in obstacles:
                    dist = np.sqrt((x - obs.position[0])**2 + (y - obs.position[1])**2)
                    min_obs_dist = min(min_obs_dist, dist)
                
                # Check distance from goal if applicable
                if self.task_config.include_goal:
                    dist_to_goal = np.sqrt(
                        (x - self._goal_position[0])**2 + 
                        (y - self._goal_position[1])**2
                    )
                else:
                    dist_to_goal = float('inf')
                
                if dist_to_robot > 1.0 and min_obs_dist > 0.8 and dist_to_goal > 0.8:
                    break
            
            # Sample size
            size_min, size_max = self.task_config.obstacle_size_range
            size = np.random.uniform(size_min, size_max)
            
            # Sample type
            obs_type = np.random.choice(self.task_config.obstacle_types)
            
            obstacles.append(Obstacle(
                position=(x, y),
                size=(size, size, 0.5),
                obstacle_type=obs_type,
                name=f"obstacle_{i}_{self._episode_count}",
            ))
        
        return obstacles
    
    def _spawn_all_obstacles(self) -> None:
        """Spawn all obstacles in Gazebo."""
        for obstacle in self._obstacles:
            self._spawn_obstacle(obstacle)
        
        # Give Gazebo time to spawn
        if self.use_ros2:
            time.sleep(0.5)
    
    def _update_standalone_obstacles(self) -> None:
        """Update lidar readings to include obstacles (standalone mode)."""
        if not self._obstacles:
            return
        
        angles = np.linspace(0, 2 * np.pi, self.task_config.lidar_points, endpoint=False)
        angles += self._robot_position[2]
        
        robot_x, robot_y = self._robot_position[:2]
        
        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            min_dist = self._lidar_data[i] * self.task_config.lidar_max_range
            
            for obstacle in self._obstacles:
                obs_x, obs_y = obstacle.position
                obs_radius = max(obstacle.size[0], obstacle.size[1]) / 2
                
                # Simple ray-circle intersection
                to_obs = np.array([obs_x - robot_x, obs_y - robot_y])
                proj = np.dot(to_obs, ray_dir)
                
                if proj > 0:
                    closest = proj * ray_dir
                    dist_to_center = np.linalg.norm(to_obs - closest)
                    
                    if dist_to_center < obs_radius:
                        hit_dist = proj - np.sqrt(obs_radius**2 - dist_to_center**2)
                        if 0 < hit_dist < min_dist:
                            min_dist = hit_dist
            
            self._lidar_data[i] = min_dist / self.task_config.lidar_max_range
    
    def _get_goal_observation(self) -> np.ndarray:
        """Compute canonical task_info for obstacle avoidance (5 dims).

        Returns
        -------
        np.ndarray of shape (5,):
            [dist/diag, angle/π, min_lidar, 0.0, 0.0]
        """
        if self.task_config.include_goal:
            dx = self._goal_position[0] - self._robot_position[0]
            dy = self._goal_position[1] - self._robot_position[1]
            distance      = float(np.sqrt(dx ** 2 + dy ** 2))
            global_angle  = float(np.arctan2(dy, dx))
            relative_angle = global_angle - float(self._robot_position[2])
            relative_angle = float(np.arctan2(
                np.sin(relative_angle), np.cos(relative_angle)
            ))
            diag = self.task_config.arena_size * np.sqrt(2.0)
            norm_dist  = float(np.clip(distance / diag, 0.0, 1.0))
            norm_angle = float(relative_angle / np.pi)
        else:
            norm_dist  = 0.0
            norm_angle = 0.0

        min_lidar = float(np.min(self._lidar_data))
        return np.array([norm_dist, norm_angle, min_lidar, 0.0, 0.0],
                        dtype=np.float32)
    
    def _get_observation(self) -> dict:
        """Get observation including obstacles and canonical task_info."""
        # Update lidar for standalone mode (adds obstacle intersections)
        if not self.use_ros2:
            self._update_standalone_obstacles()

        obs = super()._get_observation()
        obs["task_info"] = self._get_goal_observation()
        return obs
    
    def _compute_safety_reward(self) -> float:
        """Compute reward based on maintaining safe distance from obstacles.
        
        Returns:
            Safety reward value
        """
        min_lidar = np.min(self._lidar_data) * self.task_config.lidar_max_range
        
        if min_lidar < self.task_config.min_obstacle_distance:
            # Penalty for being too close
            return -self.task_config.safety_reward_scale * (
                self.task_config.min_obstacle_distance - min_lidar
            )
        else:
            # Small reward for maintaining safe distance
            return self.task_config.safety_reward_scale * 0.1
    
    def _compute_reward(
        self, action: np.ndarray, observation: Dict[str, np.ndarray]
    ) -> float:
        """Compute reward for obstacle avoidance task."""
        reward = 0.0
        
        # Safety reward
        reward += self._compute_safety_reward()
        
        # Progress reward (forward motion)
        linear_vel = action[0] * self.task_config.max_linear_vel
        reward += self.task_config.progress_reward_scale * max(0, linear_vel)
        
        # Goal-related rewards
        if self.task_config.include_goal:
            dx = self._goal_position[0] - self._robot_position[0]
            dy = self._goal_position[1] - self._robot_position[1]
            current_distance = np.sqrt(dx**2 + dy**2)
            
            # Distance improvement (scaled to match other rewards)
            distance_improvement = self._previous_distance - current_distance
            reward += distance_improvement * 1.0  # Reduced from 5.0
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
        """Check if episode should terminate."""
        if self._collision:
            return True
        if self.task_config.include_goal and self._goal_reached:
            return True
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode information."""
        info = super()._get_info()
        
        # Add obstacle info
        info["obstacles"] = [
            {"position": obs.position, "size": obs.size, "type": obs.obstacle_type}
            for obs in self._obstacles
        ]
        info["num_obstacles"] = len(self._obstacles)
        
        # Add goal info if applicable
        if self.task_config.include_goal:
            dx = self._goal_position[0] - self._robot_position[0]
            dy = self._goal_position[1] - self._robot_position[1]
            info["goal_position"] = self._goal_position.copy()
            info["distance_to_goal"] = np.sqrt(dx**2 + dy**2)
            info["goal_reached"] = self._goal_reached
            info["is_success"] = self._goal_reached
        else:
            info["is_success"] = not self._collision
        
        return info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment with new obstacles.
        
        Args:
            seed: Random seed
            options: Additional options including 'obstacles', 'goal_position'
            
        Returns:
            observation: Initial observation
            info: Episode information
        """
        # Clear existing obstacles
        self._clear_all_obstacles()
        
        # Reset base environment
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset goal state
        self._goal_reached = False
        
        # Set goal if applicable
        if self.task_config.include_goal:
            if options and "goal_position" in options:
                self._goal_position = np.array(options["goal_position"], dtype=np.float32)
            elif self.task_config.goal_position:
                self._goal_position = np.array(self.task_config.goal_position, dtype=np.float32)
            else:
                # Sample goal
                arena = self.task_config.arena_size
                margin = 0.5
                self._goal_position = np.array([
                    np.random.uniform(-arena + margin, arena - margin),
                    np.random.uniform(-arena + margin, arena - margin),
                ], dtype=np.float32)
            
            # Initialize previous distance
            dx = self._goal_position[0] - self._robot_position[0]
            dy = self._goal_position[1] - self._robot_position[1]
            self._previous_distance = np.sqrt(dx**2 + dy**2)
        
        # Spawn obstacles
        if options and "obstacles" in options:
            self._obstacles = options["obstacles"]
        else:
            self._obstacles = self._sample_obstacles()
        
        self._spawn_all_obstacles()
        
        # Get observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def close(self) -> None:
        """Clean up resources."""
        self._clear_all_obstacles()
        super().close()


class FlatObstacleAvoidanceEnv(gym.ObservationWrapper):
    """Flatten obstacle avoidance observations to canonical 370-dim vector.

    Layout: lidar(360) + robot_state(5) + task_info(5) = 370 dims.
    """

    def __init__(self, env: ObstacleAvoidanceEnv):
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


def make_obstacle_avoidance_env(
    config=None,
    flat_obs: bool = True,
    use_ros2: bool = False,
) -> gym.Env:
    """Factory function to create obstacle avoidance environment."""
    env = ObstacleAvoidanceEnv(config=config, use_ros2=use_ros2)
    if flat_obs:
        env = FlatObstacleAvoidanceEnv(env)
    return env
