"""
Observation builder for the meta-rl-tb3 Gazebo <-> policy bridge.

Challenge/risk:
  Training placed the robot anywhere in [-arena, arena]^2 with the goal in the
  same frame. Gazebo /odom starts at (0,0) at the spawn pose. The bridge maps
  odom -> arena frame by adding the spawn offset (SPAWN_XY), so the policy sees
  the same normalized position distribution it trained on. Set SPAWN_XY to the
  robot's spawn pose in the arena frame (default (0,0): spawn == arena center).
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ObsSpec:
    """Normalization constants."""
    n_rays: int = 360
    lidar_max_range: float = 3.5
    lidar_min_range: float = 0.12
    arena_size: float = 4.0            # half-width; training default
    max_linear_velocity: float = 0.22
    max_angular_velocity: float = 2.84
    # Spawn pose in the ARENA frame (x, y). odom is offset by this so the
    # policy sees arena-centered coordinates as in training.
    spawn_xy: Tuple[float, float] = (0.0, 0.0)

    @property
    def diag(self) -> float:
        """Arena diagonal used by GoalReachingEnv._get_goal_observation."""
        return self.arena_size * math.sqrt(2.0)

    @property
    def ray_angles(self) -> np.ndarray:
        """Body-frame ray angles, matching _generate_standalone_lidar (fov=360)."""
        return np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False)


def resample_scan_to_rays(
    scan_ranges: Sequence[float],
    scan_angle_min: float,
    scan_angle_increment: float,
    spec: ObsSpec,
) -> np.ndarray:
    """Resample a raw ROS LaserScan into the 360 body-frame rays the policy expects.

    Non-finite / non-positive "no return" values map to max_range.
    """
    scan = np.asarray(scan_ranges, dtype=np.float64)
    n = scan.shape[0]
    if n == 0:
        return np.full(spec.n_rays, spec.lidar_max_range, dtype=np.float32)

    beam_angles = scan_angle_min + np.arange(n) * scan_angle_increment
    beam_angles = np.mod(beam_angles, 2.0 * np.pi)

    # If the scan already has exactly n_rays beams aligned to our grid, the
    # nearest-neighbor loop still returns them; but the fast path avoids O(n^2).
    target = np.mod(spec.ray_angles, 2.0 * np.pi)
    out = np.empty(spec.n_rays, dtype=np.float64)
    # Vectorized nearest-neighbor: for each target, min circular distance.
    for k, t in enumerate(target):
        d = np.abs(beam_angles - t)
        d = np.minimum(d, 2.0 * np.pi - d)
        out[k] = scan[int(np.argmin(d))]

    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = spec.lidar_max_range
    out = np.clip(out, spec.lidar_min_range, spec.lidar_max_range)
    return out.astype(np.float32)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about z) from a quaternion, in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def build_observation(
    scan_ranges: Sequence[float],
    scan_angle_min: float,
    scan_angle_increment: float,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    v_linear: float,
    v_angular: float,
    goal_x: float,
    goal_y: float,
    spec: Optional[ObsSpec] = None,
) -> np.ndarray:
    """Assemble the 370-dim normalized observation from raw ROS inputs.

    robot_x, robot_y, goal_x, goal_y are in the ARENA frame (see ObsSpec.spawn_xy;
    the caller is responsible for adding the spawn offset to odom).
    Mirrors TurtleBot3Env._get_observation() + GoalReachingEnv._get_goal_observation().
    """
    spec = spec or ObsSpec()

    # 1) LiDAR (360), normalized to [0,1] by max_range.
    lidar = resample_scan_to_rays(scan_ranges, scan_angle_min, scan_angle_increment, spec)
    lidar_norm = np.clip(lidar / spec.lidar_max_range, 0.0, 1.0)

    # 2) robot_state (5): arena-normalized pose + normalized velocity.
    robot_state = np.array([
        robot_x / spec.arena_size,
        robot_y / spec.arena_size,
        robot_yaw / math.pi,
        v_linear / spec.max_linear_velocity,
        v_angular / spec.max_angular_velocity,
    ], dtype=np.float32)
    robot_state = np.clip(robot_state, -1.0, 1.0)

    # 3) task_info (5): goal_reaching layout.
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = math.hypot(dx, dy)
    rel_angle = wrap_to_pi(math.atan2(dy, dx) - robot_yaw)
    task_info = np.array([
        float(np.clip(distance / spec.diag, 0.0, 1.0)),
        rel_angle / math.pi,
        0.0, 0.0, 0.0,
    ], dtype=np.float32)

    obs = np.concatenate([lidar_norm, robot_state, task_info]).astype(np.float32)
    assert obs.shape[0] == 370, f"expected 370-dim obs, got {obs.shape[0]}"
    return obs


def action_to_twist(action: Sequence[float], spec: Optional[ObsSpec] = None):
    """Map a policy action in [-1,1]^2 to (linear_x, angular_z) commands.

    Matches TurtleBot3Env.step:
        linear_x  = action[0] * max_linear_velocity   -> [-max, max] (bidirectional)
        angular_z = action[1] * max_angular_velocity  -> [-max, max]
    NOTE: unlike some diff-drive setups, meta-rl-tb3 allows reverse motion.
    """
    spec = spec or ObsSpec()
    a0 = float(np.clip(action[0], -1.0, 1.0))
    a1 = float(np.clip(action[1], -1.0, 1.0))
    linear_x = a0 * spec.max_linear_velocity
    angular_z = a1 * spec.max_angular_velocity
    return linear_x, angular_z
