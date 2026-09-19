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

# ROS2 imports - these will be available when running with ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from std_srvs.srv import Empty
    from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity
    from gazebo_msgs.msg import EntityState
    import tf_transformations
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Running in simulation-only mode.")


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


class TurtleBot3Env(gym.Env):
    """Base Gymnasium environment for TurtleBot3 with ROS2/Gazebo.
    
    This environment has been referenced from my previous projects.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[TurtleBot3EnvConfig] = None,
        render_mode: Optional[str] = None,
        use_ros2: bool = True,
    ):
        """Initialize the TurtleBot3 environment.
        
        """
        super().__init__()
        
        self.config = config or TurtleBot3EnvConfig()
        self.render_mode = render_mode
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # State variables
        self._current_step = 0
        self._episode_count = 0
        self._lidar_data = np.zeros(self.config.lidar_points, dtype=np.float32)
        self._robot_position = np.zeros(3, dtype=np.float32)  # x, y, theta
        self._robot_velocity = np.zeros(2, dtype=np.float32)  # linear, angular
        self._collision = False
        self._data_received = {"scan": False, "odom": False}

        # ── Dynamics perturbation state ────────────────────────────────────
        # Active DynamicsConfig — None means nominal (unperturbed) physics.
        self._dynamics: Optional[DynamicsConfig] = self.config.dynamics

        # Control latency: FIFO queue of pending velocity commands.
        # Size = control_latency_steps + 1 so the head is applied this step.
        _latency = (self._dynamics.control_latency_steps + 1
                    if self._dynamics is not None else 1)
        self._cmd_queue: deque = deque(
            [(0.0, 0.0)] * _latency, maxlen=_latency
        )

        # Observation latency: FIFO queue of recent robot_state observations.
        _obs_latency = (self._dynamics.obs_latency_steps + 1
                        if self._dynamics is not None else 1)
        self._obs_queue: deque = deque(
            [np.zeros(5, dtype=np.float32)] * _obs_latency,
            maxlen=_obs_latency,
        )
        
        # ROS2 components
        self._node: Optional[Node] = None
        self._cmd_pub = None
        self._scan_sub = None
        self._odom_sub = None
        self._reset_world_client = None
        self._reset_sim_client = None
        self._set_entity_client = None
        self._spin_thread = None
        self._stop_spin = threading.Event()  # signals the spin thread to exit
        self._lock = threading.Lock()
        
        if self.use_ros2:
            self._init_ros2()
    
    def _setup_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space: LiDAR + robot state
        # LiDAR: normalized distances [0, 1]
        # Robot state: [x, y, theta, linear_vel, angular_vel] normalized
        
        lidar_low = np.zeros(self.config.lidar_points, dtype=np.float32)
        lidar_high = np.ones(self.config.lidar_points, dtype=np.float32)
        
        # Robot state bounds (normalized to roughly [-1, 1])
        state_low = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        state_high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=lidar_low, high=lidar_high, dtype=np.float32),
            "robot_state": spaces.Box(low=state_low, high=state_high, dtype=np.float32),
        })
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
    
    def _init_ros2(self) -> None:
        """Initialize ROS2 node and subscribers/publishers."""
        if not rclpy.ok():
            rclpy.init()
        
        self._node = rclpy.create_node(
            f"{self.config.node_name}_{id(self)}",
            parameter_overrides=[
                rclpy.Parameter("use_sim_time", value=self.config.use_sim_time)
            ] if hasattr(rclpy, 'Parameter') else []
        )
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Velocity command publisher
        self._cmd_pub = self._node.create_publisher(
            Twist, self.config.cmd_vel_topic, 10
        )
        
        # LiDAR subscriber
        self._scan_sub = self._node.create_subscription(
            LaserScan,
            self.config.scan_topic,
            self._scan_callback,
            sensor_qos
        )
        
        # Odometry subscriber
        self._odom_sub = self._node.create_subscription(
            Odometry,
            self.config.odom_topic,
            self._odom_callback,
            sensor_qos
        )
        
        # Gazebo service clients
        self._reset_world_client = self._node.create_client(
            Empty, "/reset_world"
        )
        self._reset_sim_client = self._node.create_client(
            Empty, "/reset_simulation"
        )
        self._set_entity_client = self._node.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        
        # Start spinning in background thread
        self._spin_thread = threading.Thread(target=self._spin_ros2, daemon=True)
        self._spin_thread.start()
        
        # Wait for initial data
        self._wait_for_data(timeout=5.0)
    
    def _spin_ros2(self) -> None:
        """Background thread for spinning ROS2 node."""
        while not self._stop_spin.is_set() and rclpy.ok():
            if self._node is None:
                break
            rclpy.spin_once(self._node, timeout_sec=0.01)
    
    def _scan_callback(self, msg: 'LaserScan') -> None:
        """Process incoming LiDAR scan data."""
        with self._lock:
            # Convert to numpy and handle inf/nan
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=self.config.lidar_max_range, 
                                   posinf=self.config.lidar_max_range)
            
            # Downsample if needed
            if len(ranges) != self.config.lidar_points:
                indices = np.linspace(0, len(ranges) - 1, 
                                      self.config.lidar_points, dtype=int)
                ranges = ranges[indices]
            
            # Clip and normalize
            ranges = np.clip(ranges, self.config.lidar_min_range, 
                            self.config.lidar_max_range)
            self._lidar_data = ranges / self.config.lidar_max_range
            
            # Check for collision
            min_distance = np.min(ranges)
            self._collision = min_distance < self.config.collision_distance
            self._data_received["scan"] = True
    
    def _odom_callback(self, msg: 'Odometry') -> None:
        """Process incoming odometry data."""
        with self._lock:
            # Extract position
            self._robot_position[0] = msg.pose.pose.position.x
            self._robot_position[1] = msg.pose.pose.position.y
            
            # Extract orientation (yaw from quaternion)
            q = msg.pose.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )
            self._robot_position[2] = yaw
            
            # Extract velocities
            self._robot_velocity[0] = msg.twist.twist.linear.x
            self._robot_velocity[1] = msg.twist.twist.angular.z
            self._data_received["odom"] = True
    
    def _wait_for_data(self, timeout: float = 5.0) -> bool:
        """Wait for initial sensor data.
        
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if all(self._data_received.values()):
                    return True
            time.sleep(0.1)
        
        self._node.get_logger().warn(
            f"Timeout waiting for sensor data. Received: {self._data_received}"
        )
        return False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary with optional sensor perturbations.

        """
        with self._lock:
            lidar = self._lidar_data.copy()

            # Current (ground-truth) robot state
            robot_state = np.array([
                self._robot_position[0] / self.config.arena_size,
                self._robot_position[1] / self.config.arena_size,
                self._robot_position[2] / np.pi,
                self._robot_velocity[0] / self.config.max_linear_vel,
                self._robot_velocity[1] / self.config.max_angular_vel,
            ], dtype=np.float32)
            robot_state = np.clip(robot_state, -1.0, 1.0)

        # ── Observation latency ───────────────────────────────────────────
        # Push current state; pop stale reading (FIFO).
        self._obs_queue.append(robot_state.copy())
        stale_state = self._obs_queue[0].copy()

        # ── Observation noise ─────────────────────────────────────────────
        dyn = self._dynamics
        if dyn is not None and dyn.obs_noise_std > 0.0:
            stale_state += np.random.normal(
                0.0, dyn.obs_noise_std, size=stale_state.shape
            ).astype(np.float32)
            stale_state = np.clip(stale_state, -1.0, 1.0)

        return {"lidar": lidar, "robot_state": stale_state}
    
    # ------------------------------------------------------------------
    # Dynamics perturbation starts
    # ------------------------------------------------------------------

    def set_dynamics(self, dynamics: Optional[DynamicsConfig]) -> None:
        self._dynamics = dynamics

    def get_dynamics(self) -> Optional[DynamicsConfig]:
        return self._dynamics

    # ------------------------------------------------------------------
    # Velocity command dispatch
    # ------------------------------------------------------------------

    def _send_velocity_command(self, linear_vel: float, angular_vel: float) -> None:

        if self.use_ros2 and self._cmd_pub is not None:
            msg = Twist()
            msg.linear.x = float(linear_vel)
            msg.angular.z = float(angular_vel)
            self._cmd_pub.publish(msg)
        else:
            # Push new command; pop the oldest (which is now "due")
            self._cmd_queue.append((linear_vel, angular_vel))
            effective_lin, effective_ang = self._cmd_queue[0]
            self._update_standalone_physics(effective_lin, effective_ang)

    # ------------------------------------------------------------------
    # Standalone physics (perturbed)
    # ------------------------------------------------------------------

    def _update_standalone_physics(self, linear_vel: float, angular_vel: float) -> None:
        """Update robot state with optional dynamics perturbations.

        Perturbation application order
        --------------------------------
        1. Actuator noise  — Gaussian noise on the command signal.
        2. Velocity scaling — multiplicative scale on max achievable vel.
        3. Wheel slip      — reduces effective linear velocity.
        4. Friction        — scales angular acceleration response.
        5. Payload         — scales both linear and angular inertia.
        6. Euler integration of unicycle kinematics.
        7. Arena boundary clamping / collision detection.
        8. LiDAR generation (with sensor perturbations).

        The payload and friction models use a first-order lag:
            v_actual[t] = v_actual[t-1] + (v_cmd - v_actual[t-1]) / tau
        where tau = payload_factor / friction.  When tau=1 (nominal),
        the robot tracks the command exactly in one step.
        """
        dyn = self._dynamics
        dt  = self.config.step_duration

        if dyn is not None:
            # 1. Actuator noise
            if dyn.actuator_noise_std > 0.0:
                linear_vel  += float(np.random.normal(0.0, dyn.actuator_noise_std))
                angular_vel += float(np.random.normal(0.0, dyn.actuator_noise_std))

            # 2. Velocity scale (motor degradation / incline)
            eff_max_lin = self.config.max_linear_vel * dyn.lin_vel_scale
            eff_max_ang = self.config.max_angular_vel * dyn.ang_vel_scale
            linear_vel  = np.clip(linear_vel,  -eff_max_lin, eff_max_lin)
            angular_vel = np.clip(angular_vel, -eff_max_ang, eff_max_ang)

            # 3. Wheel slip (linear velocity attenuation)
            linear_vel *= dyn.wheel_slip

            # 4 & 5. Payload-inertia + friction lag model.
            # tau = payload_factor / friction.  Larger tau → slower response.
            tau_lin = dyn.payload_factor / max(dyn.friction, 1e-3)
            tau_ang = dyn.payload_factor / max(dyn.friction, 1e-3)
            v_lin_prev = self._robot_velocity[0]
            v_ang_prev = self._robot_velocity[1]
            # Clamp tau to [1, payload_factor*3] so step never overshoots
            tau_lin = float(np.clip(tau_lin, 1.0, dyn.payload_factor * 3.0))
            tau_ang = float(np.clip(tau_ang, 1.0, dyn.payload_factor * 3.0))
            linear_vel  = v_lin_prev + (linear_vel  - v_lin_prev) / tau_lin
            angular_vel = v_ang_prev + (angular_vel - v_ang_prev) / tau_ang
        else:
            # Nominal: no perturbation — command is applied directly
            eff_max_lin = self.config.max_linear_vel
            eff_max_ang = self.config.max_angular_vel
            linear_vel  = np.clip(linear_vel,  -eff_max_lin, eff_max_lin)
            angular_vel = np.clip(angular_vel, -eff_max_ang, eff_max_ang)

        # 6. Euler integration (unicycle kinematics)
        theta = self._robot_position[2]
        self._robot_position[0] += linear_vel * np.cos(theta) * dt
        self._robot_position[1] += linear_vel * np.sin(theta) * dt
        self._robot_position[2] += angular_vel * dt

        # Normalise angle to [-π, π]
        self._robot_position[2] = np.arctan2(
            np.sin(self._robot_position[2]),
            np.cos(self._robot_position[2]),
        )

        # Store effective velocities (used by observations)
        self._robot_velocity[0] = float(linear_vel)
        self._robot_velocity[1] = float(angular_vel)

        # 7. Arena boundary
        arena = self.config.arena_size
        if (abs(self._robot_position[0]) > arena
                or abs(self._robot_position[1]) > arena):
            self._collision = True
            self._robot_position[0] = np.clip(self._robot_position[0], -arena, arena)
            self._robot_position[1] = np.clip(self._robot_position[1], -arena, arena)

        # 8. LiDAR (with sensor perturbations)
        self._generate_standalone_lidar()

    def _generate_standalone_lidar(self) -> None:
        """Vectorised wall ray-casting with optional sensor perturbations.

        Perturbation pipeline
        ---------------------
        1. Compute ideal wall distances (ray-wall intersection).
        2. Apply range scale (models lens degradation / range limiting).
        3. Add Gaussian range noise.
        4. Apply beam dropout (set dropped beams to max range).
        5. Clip and normalise to [0, 1].
        """
        n      = self.config.lidar_points
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles += self._robot_position[2]

        x, y  = self._robot_position[0], self._robot_position[1]
        arena = self.config.arena_size

        dx = np.cos(angles)
        dy = np.sin(angles)

        with np.errstate(divide='ignore', invalid='ignore'):
            t_xn = np.where(dx != 0, (-arena - x) / dx, np.inf)
            t_xp = np.where(dx != 0, ( arena - x) / dx, np.inf)
            t_yn = np.where(dy != 0, (-arena - y) / dy, np.inf)
            t_yp = np.where(dy != 0, ( arena - y) / dy, np.inf)

        t_all = np.stack([t_xn, t_xp, t_yn, t_yp], axis=0)
        t_all = np.where(t_all > 1e-9, t_all, np.inf)
        dist  = np.min(t_all, axis=0)  # ideal wall distance per beam

        dyn = self._dynamics
        if dyn is not None:
            # 1. Range scale
            effective_max = self.config.lidar_max_range * dyn.lidar_range_scale
            dist = np.minimum(dist, effective_max)

            # 2. Gaussian range noise
            if dyn.lidar_noise_std > 0.0:
                dist += np.random.normal(0.0, dyn.lidar_noise_std, size=n)

            # 3. Beam dropout — set dropped beams to max range
            if dyn.lidar_dropout_prob > 0.0:
                mask = np.random.random(n) < dyn.lidar_dropout_prob
                dist[mask] = self.config.lidar_max_range
        
        dist = np.clip(dist, self.config.lidar_min_range, self.config.lidar_max_range)
        self._lidar_data = (dist / self.config.lidar_max_range).astype(np.float32)

    def _reset_robot_pose(
        self, 
        position: Optional[Tuple[float, float]] = None,
        orientation: Optional[float] = None
    ) -> None:
        """Reset robot to specified or random pose.
        
        Args:
            position: (x, y) position. Random if None.
            orientation: Yaw angle. Random if None.
        """
        if position is None:
            # Random position within arena (not too close to edges)
            margin = 0.5
            x = np.random.uniform(-self.config.arena_size + margin,
                                  self.config.arena_size - margin)
            y = np.random.uniform(-self.config.arena_size + margin,
                                  self.config.arena_size - margin)
            position = (x, y)
        
        if orientation is None:
            orientation = np.random.uniform(-np.pi, np.pi)
        
        if self.use_ros2 and self._set_entity_client is not None:
            # Use Gazebo service to reset robot pose
            self._set_robot_gazebo_pose(position, orientation)
        else:
            # Direct state update for standalone mode
            self._robot_position[0] = position[0]
            self._robot_position[1] = position[1]
            self._robot_position[2] = orientation
            self._robot_velocity[:] = 0
    
    def _set_robot_gazebo_pose(
        self, 
        position: Tuple[float, float], 
        orientation: float
    ) -> None:
        """Set robot pose in Gazebo using service call."""
        if not self._set_entity_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn("Set entity state service not available")
            return
        
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = self.config.robot_model
        request.state.pose.position.x = position[0]
        request.state.pose.position.y = position[1]
        request.state.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        q = tf_transformations.quaternion_from_euler(0, 0, orientation)
        request.state.pose.orientation.x = q[0]
        request.state.pose.orientation.y = q[1]
        request.state.pose.orientation.z = q[2]
        request.state.pose.orientation.w = q[3]
        
        # Zero velocities
        request.state.twist.linear.x = 0.0
        request.state.twist.linear.y = 0.0
        request.state.twist.linear.z = 0.0
        request.state.twist.angular.x = 0.0
        request.state.twist.angular.y = 0.0
        request.state.twist.angular.z = 0.0
        
        future = self._set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=1.0)
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Action array [linear_vel, angular_vel] in [-1, 1]
            
        Returns:
            observation: Current observation
            reward: Step reward (0 for base class, override in subclasses)
            terminated: Whether episode ended due to terminal state
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        self._current_step += 1
        
        # Scale action to actual velocity commands
        linear_vel = action[0] * self.config.max_linear_vel
        angular_vel = action[1] * self.config.max_angular_vel
        
        # Send command
        self._send_velocity_command(linear_vel, angular_vel)
        
        # Wait for simulation step
        if self.use_ros2:
            time.sleep(self.config.step_duration)
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward (base implementation returns 0)
        reward = self._compute_reward(action, observation)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._current_step >= self.config.max_episode_steps
        
        # Build info dict
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment.
        
        """
        super().reset(seed=seed)
        
        # Reset state
        self._current_step = 0
        self._episode_count += 1
        self._collision = False

        # Re-initialise latency queues in case dynamics changed since __init__
        _ctrl_latency = (
            self._dynamics.control_latency_steps + 1
            if self._dynamics is not None else 1
        )
        self._cmd_queue = deque(
            [(0.0, 0.0)] * _ctrl_latency, maxlen=_ctrl_latency
        )
        _obs_latency = (
            self._dynamics.obs_latency_steps + 1
            if self._dynamics is not None else 1
        )
        self._obs_queue = deque(
            [np.zeros(5, dtype=np.float32)] * _obs_latency,
            maxlen=_obs_latency,
        )
        
        # Stop the robot
        self._send_velocity_command(0.0, 0.0)
        
        # Get reset options
        position = options.get("position") if options else None
        orientation = options.get("orientation") if options else None
        
        # Reset robot pose
        self._reset_robot_pose(position, orientation)
        
        # Wait for state to settle
        if self.use_ros2:
            time.sleep(0.5)
            self._wait_for_data(timeout=2.0)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _compute_reward(
        self, action: np.ndarray, observation: Dict[str, np.ndarray]
    ) -> float:

        return 0.0
    
    def _is_terminated(self) -> bool:

        return self._collision
    
    def _get_info(self) -> Dict[str, Any]:

        with self._lock:
            return {
                "step": self._current_step,
                "episode": self._episode_count,
                "collision": self._collision,
                "position": self._robot_position.copy(),
                "velocity": self._robot_velocity.copy(),
            }

    # will implement this later
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        """
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            return None
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        # Stop the robot
        self._send_velocity_command(0.0, 0.0)
        
        if self.use_ros2:
            # Signal the spin thread to stop and wait for it to exit cleanly
            # before we destroy the node, eliminating the race condition where
            # the thread calls spin_once on a destroyed node.
            self._stop_spin.set()
            if self._spin_thread is not None and self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        
        if self.use_ros2 and self._node is not None:
            self._node.destroy_node()
            self._node = None
        
        # Note: Don't call rclpy.shutdown() as other environments might be using it
    
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state.
        """
        with self._lock:
            return {
                "position": self._robot_position.copy(),
                "velocity": self._robot_velocity.copy(),
            }
    
    def get_lidar_data(self) -> np.ndarray:
        """Get current LiDAR readings.

        """
        with self._lock:
            return self._lidar_data.copy()


# Flat observation wrapper for algorithms that prefer flat arrays
class FlatObservationWrapper(gym.ObservationWrapper):
    """Wrapper to flatten Dict observations to a single array."""
    
    def __init__(self, env: TurtleBot3Env):
        super().__init__(env)
        
        # Calculate total observation size
        lidar_size = env.observation_space["lidar"].shape[0]
        state_size = env.observation_space["robot_state"].shape[0]
        total_size = lidar_size + state_size
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_size,), dtype=np.float32
        )
    
    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten observation dictionary."""
        return np.concatenate([obs["lidar"], obs["robot_state"]])
