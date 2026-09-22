#!/usr/bin/env python3
"""
Gazebo <-> trained-policy bridge for meta-rl-tb3 (ROS2 Humble).

Runs a CADRE-ctx or FOMAML policy trained in the standalone TurtleBot3Env
against a Gazebo TurtleBot3 Burger, ZERO-SHOT, by reconstructing the exact
370-dim observation the policy expects from live /scan and /odom, and
publishing /cmd_vel at 10 Hz (matching training step_duration=0.1).

Node structure adapted from a known-working sim-to-sim bridge
(research/adaptive_nav/gazebo_bridge): subscribe /scan+/odom, spin, publish
/cmd_vel on a fixed-rate timer, infer collision from nearest scan return,
grace period after reset. The policy-loading and observation contract are
meta-rl-tb3 specific (370-dim, causal GRU context).

Usage (inside the ROS2/Gazebo container, after sourcing ROS2):
    # 1. Launch the world (see launch/replicated_world.launch.py or
    #    turtlebot3_gazebo empty_world.launch.py)
    # 2. Run this node:
    python3 gazebo_bridge/gazebo_policy_bridge.py \
        --ckpt results/ablations/cadre_seed42_best.pt \
        --method cadre_ctx --goal 2.0 2.0 --max-steps 200

Frame handling: /odom starts at (0,0) at spawn. --spawn-xy sets the spawn pose
in the ARENA frame so the policy sees arena-centered coordinates as in
training. Goal is given in the ARENA frame.
"""

import argparse
import math
import sys
import time
from collections import deque, OrderedDict
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gazebo_bridge.obs_builder import (  # noqa: E402
    ObsSpec, build_observation, yaw_from_quaternion, action_to_twist,
)

CONTROL_HZ = 10.0            # matches training step_duration = 0.1 s
GOAL_THRESHOLD = 0.25        # m, matches GoalReachingConfig.goal_threshold
ROBOT_RADIUS = 0.105         # m, TB3 Burger
COLLISION_MARGIN = 0.03      # m; collision if nearest obstacle < radius + margin
COLLISION_GRACE_STEPS = 5    # skip collision checks for first N steps after reset
CONTEXT_WINDOW = 5           # matches ContextEncoderConfig.context_window


class _CADREPolicy:
    """Loads a meta-rl-tb3 CADRE/FOMAML checkpoint and runs it zero-shot with
    the causal context protocol used in evaluate_cadre_at_budget.

    - cadre_ctx: encoder_type="gru", context inferred from a rolling buffer of
      the last K=5 transitions (built AFTER each action, matching training).
    - fomaml:    encoder_type="none", context_dim=0, obs used directly.
    """

    def __init__(self, ckpt_path: str, method: str, device: str = "cpu"):
        import torch
        from meta_rl_tb3.algos.cadre import CADRE, CADREConfig
        from meta_rl_tb3.algos.networks import ContextEncoderConfig, NetworkConfig
        from meta_rl_tb3.envs.goal_reaching import (
            GoalReachingEnv, GoalReachingConfig, FlatGoalReachingEnv,
        )
        from meta_rl_tb3.envs.wrappers import CANONICAL_OBS_DIM

        self._torch = torch
        self.method = method
        self.device = device

        encoder_type = "gru" if method == "cadre_ctx" else "none"

        def _env_fn():
            return FlatGoalReachingEnv(
                GoalReachingEnv(
                    GoalReachingConfig(max_episode_steps=200, goal_threshold=0.25),
                    use_ros2=False,
                )
            )

        cfg = CADREConfig(
            meta_lr=3e-4, inner_lr=0.01, num_inner_steps=0,
            meta_batch_size=5, num_support_episodes=3, num_query_episodes=3,
            max_episode_steps=200, encoder_type=encoder_type,
            context_encoder_config=ContextEncoderConfig(
                obs_dim=CANONICAL_OBS_DIM, action_dim=2, context_dim=16,
                context_window=CONTEXT_WINDOW, gru_hidden_dim=64, num_gru_layers=2,
            ),
            first_order=True,
            network_config=NetworkConfig(hidden_sizes=[256, 256], activation="tanh"),
            device=device, seed=0,
        )
        self.agent = CADRE(_env_fn, cfg)
        self.agent.load(ckpt_path)

        # Clone meta-params once (zero-shot: same params every step).
        self.params = OrderedDict(
            (n, p.clone()) for n, p in self.agent.actor_critic.named_parameters()
        )
        self.enc_params = (dict(self.agent.encoder.named_parameters())
                           if self.agent.encoder is not None else {})
        self._trans_buf = deque(maxlen=CONTEXT_WINDOW)
        self._prev_obs = None

    def reset(self):
        self._trans_buf.clear()
        self._prev_obs = None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            z = self.agent._encode_context(self._trans_buf, self.enc_params)
            z_np = z.cpu().numpy().squeeze(0)
            p_obs = (np.concatenate([obs, z_np])
                     if self.agent.context_dim > 0 else obs)
            obs_t = torch.from_numpy(p_obs).float().unsqueeze(0).to(self.device)
            act_t, _, _, _ = self.agent._forward_policy(obs_t, self.params)
        return act_t.cpu().numpy().squeeze(0)

    def record(self, obs, action, reward, next_obs):
        """Append the transition AFTER acting (causal protocol)."""
        self._trans_buf.append(
            self.agent._build_transition(obs, action, reward, next_obs)
        )


class PolicyBridge(Node):
    def __init__(self, args):
        super().__init__("meta_rl_tb3_gazebo_bridge")
        self.args = args
        self.spec = ObsSpec(
            arena_size=args.arena_size,
            spawn_xy=(args.spawn_xy[0], args.spawn_xy[1]),
        )
        self.goal = (float(args.goal[0]), float(args.goal[1]))   # arena frame
        self.max_steps = args.max_steps

        self._scan = None
        self._odom = None
        self.step_count = 0
        self.done = False
        self.result = None

        self._prev_obs = None
        self._prev_action = np.zeros(2, dtype=np.float32)

        self.policy = _CADREPolicy(args.ckpt, args.method, device=args.device)
        self.policy.reset()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5,
        )
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, sensor_qos)
        self.create_subscription(Odometry, args.odom_topic, self._on_odom, sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.get_logger().info(
            f"Bridge up. method={args.method} goal={self.goal} "
            f"arena={self.spec.arena_size} spawn_xy={self.spec.spawn_xy} "
            f"scan={args.scan_topic} odom={args.odom_topic} rate={CONTROL_HZ}Hz"
        )

    def _on_scan(self, msg: LaserScan):
        self._scan = (np.asarray(msg.ranges, dtype=np.float64),
                      float(msg.angle_min), float(msg.angle_increment))

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        tw = msg.twist.twist
        self._odom = (p.x, p.y, yaw, tw.linear.x, tw.angular.z)

    def _control_step(self):
        if self.done:
            return
        if self._scan is None or self._odom is None:
            return

        ranges, angle_min, angle_inc = self._scan
        ox, oy, yaw, v_lin, v_ang = self._odom
        # Map odom -> arena frame (spawn offset).
        rx = ox + self.spec.spawn_xy[0]
        ry = oy + self.spec.spawn_xy[1]

        dist_to_goal = math.hypot(self.goal[0] - rx, self.goal[1] - ry)
        finite = np.isfinite(ranges) & (ranges > 0)
        nearest = float(np.min(ranges[finite])) if np.any(finite) else self.spec.lidar_max_range

        # One-time startup diagnostic: print the first odom/goal geometry so a
        # false "goal at step 0" (frame/units bug) is immediately visible.
        if self.step_count == 0:
            self.get_logger().info(
                f"[startup] odom=({ox:.2f},{oy:.2f}) yaw={yaw:.2f} "
                f"arena_pos=({rx:.2f},{ry:.2f}) goal={self.goal} "
                f"dist_to_goal={dist_to_goal:.2f} nearest_scan={nearest:.2f} "
                f"n_scan={len(ranges)} angle_min={angle_min:.3f}"
            )

        # Guard against a spurious step-0 success: require at least one control
        # step before accepting a goal, so an unpopulated/degenerate first odom
        # cannot register as "reached" before the robot has moved.
        if dist_to_goal < GOAL_THRESHOLD and self.step_count > 0:
            self._finish("goal"); return
        if self.step_count >= COLLISION_GRACE_STEPS and \
                nearest < (ROBOT_RADIUS + COLLISION_MARGIN):
            self._finish("collision"); return
        if self.step_count >= self.max_steps:
            self._finish("timeout"); return

        obs = build_observation(
            scan_ranges=ranges, scan_angle_min=angle_min,
            scan_angle_increment=angle_inc,
            robot_x=rx, robot_y=ry, robot_yaw=yaw,
            v_linear=v_lin, v_angular=v_ang,
            goal_x=self.goal[0], goal_y=self.goal[1], spec=self.spec,
        )

        action = self.policy.predict(obs)

        # Record transition for the NEXT step's context (reward not needed for
        # goal-reaching context at eval; use 0.0 — matches zero-shot eval where
        # the transition's reward field is present but the encoder was trained
        # with dense rewards. We approximate with a progress signal to stay
        # close to training statistics).
        if self._prev_obs is not None:
            reward = float(getattr(self, "_last_progress", 0.0))
            self.policy.record(self._prev_obs, self._prev_action, reward, obs)
        self._prev_obs = obs.copy()
        self._prev_action = np.asarray(action, dtype=np.float32).copy()
        self._last_progress = -dist_to_goal   # rough progress proxy

        lin_x, ang_z = action_to_twist(action, self.spec)
        cmd = Twist(); cmd.linear.x = float(lin_x); cmd.angular.z = float(ang_z)
        self.cmd_pub.publish(cmd)

        self.step_count += 1
        if self.step_count % 10 == 0:
            self.get_logger().info(
                f"step={self.step_count} dist_goal={dist_to_goal:.2f} "
                f"near={nearest:.2f} v={lin_x:.2f} w={ang_z:.2f}"
            )

    def _finish(self, result: str):
        self.done = True
        self.result = result
        self.cmd_pub.publish(Twist())
        self.get_logger().info(f"EPISODE DONE: result={result} steps={self.step_count}")


def main():
    p = argparse.ArgumentParser(description="meta-rl-tb3 Gazebo policy bridge")
    p.add_argument("--ckpt", required=True, help="CADRE/FOMAML .pt checkpoint")
    p.add_argument("--method", default="cadre_ctx", choices=["cadre_ctx", "fomaml"])
    p.add_argument("--goal", nargs=2, type=float, required=True,
                   metavar=("X", "Y"), help="Goal in ARENA frame (m)")
    p.add_argument("--arena-size", type=float, default=4.0,
                   help="Arena half-width (training default 4.0)")
    p.add_argument("--spawn-xy", nargs=2, type=float, default=[0.0, 0.0],
                   metavar=("X", "Y"), help="Spawn pose in arena frame")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--odom-topic", default="/odom")
    p.add_argument("--cmd-topic", default="/cmd_vel")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rclpy.init()
    node = PolicyBridge(args)
    result = "interrupted"
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        result = node.result or "interrupted"
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.get_logger().info(f"Shutting down. Final result: {result}")
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if result == "goal" else 1)


if __name__ == "__main__":
    main()
