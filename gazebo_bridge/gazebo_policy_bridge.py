#!/usr/bin/env python3
"""
Gazebo <-> trained-policy bridge for meta-rl-tb3 (ROS2 Humble).

Runs a CADRE-ctx or FOMAML policy trained in the standalone TurtleBot3Env
against a Gazebo TurtleBot3 Burger, ZERO-SHOT, by reconstructing the exact
370-dim observation the policy expects from live /scan and /odom, and
publishing /cmd_vel at 10 Hz (matching training step_duration=0.1).


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

