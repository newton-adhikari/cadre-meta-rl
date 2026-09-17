"""Base TurtleBot3 Gymnasium environment with ROS2/Gazebo integration.

This module provides the core bridge between ROS2/Gazebo and the Gymnasium
interface, enabling reinforcement learning experiments with TurtleBot3.

"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List
import threading

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from meta_rl_tb3.envs.physics import DynamicsConfig

@dataclass
class TurtleBot3EnvConfig:
    """Configuration for TurtleBot3 environment.
    
    Attributes:
        lidar_points: Number of LiDAR scan points (default 360, can downsample)
        lidar_max_range: Maximum LiDAR range in meters
        lidar_min_range: Minimum LiDAR range in meters
        max_linear_vel: Maximum forward/backward velocity (m/s)
        max_angular_vel: Maximum rotational velocity (rad/s)
        collision_distance: Distance threshold for collision detection
        max_episode_steps: Maximum steps per episode
        step_duration: Duration of each simulation step in seconds
        arena_size: Size of the arena (half-width/height from center)
        robot_radius: Radius of the robot for collision detection
        use_sim_time: Whether to use Gazebo simulation time
        cmd_vel_topic: Topic name for velocity commands
        scan_topic: Topic name for LiDAR scans
        odom_topic: Topic name for odometry
        node_name: Name for the ROS2 node
        robot_model: Gazebo model name for the robot
            ('turtlebot3_burger', 'turtlebot3_waffle', 'turtlebot3_waffle_pi')
    """
    lidar_points: int = 360
    lidar_max_range: float = 3.5
    lidar_min_range: float = 0.12
    max_linear_vel: float = 0.22
    max_angular_vel: float = 2.84
    collision_distance: float = 0.15
    max_episode_steps: int = 500
    step_duration: float = 0.1
    arena_size: float = 4.0
    robot_radius: float = 0.105
    use_sim_time: bool = True
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    node_name: str = "turtlebot3_env"
    robot_model: str = "turtlebot3_burger"
    # Dynamics perturbation configuration (standalone mode only).
    # None = nominal physics (no perturbation).
    dynamics: Optional[DynamicsConfig] = None

