#!/usr/bin/env python3
"""Gazebo validation: zero-shot evaluation of trained CADRE-ctx / FOMAML
checkpoints on a TurtleBot3 Burger in the Gazebo physics simulator.

This does NOT retrain anything. It loads a checkpoint trained in the
standalone simulator and evaluates it on Gazebo physics under NOMINAL
dynamics (no synthetic perturbation).

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_rl_tb3.algos.cadre import CADRE, CADREConfig
from meta_rl_tb3.algos.networks import ContextEncoderConfig, NetworkConfig
from meta_rl_tb3.envs.goal_reaching import GoalReachingEnv, GoalReachingConfig, FlatGoalReachingEnv
from meta_rl_tb3.envs.wrappers import CANONICAL_OBS_DIM

MAX_EP_STEPS = 200
CONTEXT_WINDOW = 5


def parse_args():
    p = argparse.ArgumentParser(description="Gazebo zero-shot validation of CADRE/FOMAML.")
    p.add_argument("--ckpt", required=True, help="Path to *_best.pt checkpoint.")
    p.add_argument("--method", required=True, choices=["cadre_ctx", "fomaml"],
                   help="cadre_ctx -> gru encoder; fomaml -> no encoder.")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of evaluation episodes (default 20).")
    p.add_argument("--seed", type=int, default=0,
                   help="Eval seed for goal/start sampling (default 0).")
    p.add_argument("--output-dir", type=str, default="results/gazebo",
                   help="Where to write metrics + raw episode log.")
    p.add_argument("--no-ros2", action="store_true",
                   help="Run standalone physics instead of Gazebo (smoke test).")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def build_agent(method: str, device: str) -> CADRE:
    """Instantiate a CADRE with the SAME architecture used at training time. """
    encoder_type = "gru" if method == "cadre_ctx" else "none"

    # A throwaway env_fn just so CADRE can infer obs/action dims at init.
    def _env_fn():
        return FlatGoalReachingEnv(
            GoalReachingEnv(
                GoalReachingConfig(max_episode_steps=MAX_EP_STEPS,
                                   goal_threshold=0.25),
                use_ros2=False,   # init only; the eval env is built separately
            )
        )

    cfg = CADREConfig(
        meta_lr=3e-4, inner_lr=0.01,
        num_inner_steps=0,               # zero-shot: no gradient adaptation
        meta_batch_size=5,
        num_support_episodes=3,
        num_query_episodes=3,
        max_episode_steps=MAX_EP_STEPS,
        encoder_type=encoder_type,
        context_encoder_config=ContextEncoderConfig(
            obs_dim=CANONICAL_OBS_DIM, action_dim=2, context_dim=16,
            context_window=CONTEXT_WINDOW, gru_hidden_dim=64, num_gru_layers=2,
        ),
        first_order=True,
        network_config=NetworkConfig(hidden_sizes=[256, 256], activation="tanh"),
        device=device, seed=0,
    )
    return CADRE(_env_fn, cfg)


def make_eval_env(use_ros2: bool):
    """Build the Gazebo (or standalone) goal-reaching env with nominal dynamics."""
    env = FlatGoalReachingEnv(
        GoalReachingEnv(
            GoalReachingConfig(max_episode_steps=MAX_EP_STEPS, goal_threshold=0.25),
            use_ros2=use_ros2,
        )
    )
    # Nominal dynamics: no synthetic perturbation. In Gazebo mode this means
    # the real Gazebo physics are used unmodified; in standalone it means the
    # unperturbed unicycle model.
    env.unwrapped.set_dynamics(None)
    return env


def run_episode(agent: CADRE, env, params, enc_params) -> dict:
    """Run one zero-shot episode with the causal context protocol."""
    obs_raw, info = env.reset()
    obs = agent._flatten_obs(obs_raw)
    trans_buf: deque = deque(maxlen=CONTEXT_WINDOW)

    total_reward = 0.0
    collided = False
    steps = 0

    for _ in range(MAX_EP_STEPS):
        with torch.no_grad():
            z = agent._encode_context(trans_buf, enc_params)
            z_np = z.cpu().numpy().squeeze(0)
            p_obs = np.concatenate([obs, z_np]) if agent.context_dim > 0 else obs
            obs_t = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
            act_t, _, _, _ = agent._forward_policy(obs_t, params)
        act_np = act_t.cpu().numpy().squeeze(0)

        next_raw, rew, term, trunc, info = env.step(act_np)
        next_obs = agent._flatten_obs(next_raw)
        total_reward += float(rew)
        steps += 1
        if info.get("collision", False):
            collided = True

        trans_buf.append(agent._build_transition(obs, act_np, rew, next_obs))
        obs = next_obs
        if term or trunc:
            break

    return {
        "success": bool(info.get("is_success", False)),
        "collision": collided,
        "steps": steps,
        "final_dist": float(info.get("distance_to_goal", float("nan"))),
        "reward": total_reward,
    }


def main():
    args = parse_args()
    use_ros2 = not args.no_ros2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "GAZEBO (ROS2)" if use_ros2 else "STANDALONE (smoke test)"
    print("=" * 60)
    print(f"Gazebo validation — {args.method}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Mode:       {mode}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Dynamics:   nominal (no synthetic perturbation)")
    print("=" * 60)

    # ── Load checkpoint ────────────────────────────────────────────────────
    agent = build_agent(args.method, args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    agent.load(str(ckpt_path))
    print(f"  Loaded checkpoint ({args.method}).")

    # Clone meta-params once; zero-shot => no adaptation, same params all eps.
    from collections import OrderedDict
    params = OrderedDict(
        (name, p.clone()) for name, p in agent.actor_critic.named_parameters()
    )
    enc_params = (dict(agent.encoder.named_parameters())
                  if agent.encoder is not None else {})

    # ── Build the eval env ─────────────────────────────────────────────────
    env = make_eval_env(use_ros2)

    # ── Run episodes ───────────────────────────────────────────────────────
    raw = []
    t0 = time.time()
    for ep in range(args.episodes):
        r = run_episode(agent, env, params, enc_params)
        r.update({"method": args.method, "episode": ep, "mode": mode})
        raw.append(r)
        print(f"  ep {ep+1:2d}/{args.episodes} | "
              f"success={int(r['success'])} collision={int(r['collision'])} "
              f"steps={r['steps']:3d} final_dist={r['final_dist']:.3f} "
              f"({time.time()-t0:.0f}s)")
    env.close()

    # ── Aggregate ──────────────────────────────────────────────────────────
    n = len(raw)
    sr = float(np.mean([e["success"] for e in raw]))
    cr = float(np.mean([e["collision"] for e in raw]))
    mean_steps = float(np.mean([e["steps"] for e in raw]))
    mean_dist = float(np.nanmean([e["final_dist"] for e in raw]))

    tag = "gazebo" if use_ros2 else "standalone"
    raw_path = out_dir / f"raw_{args.method}_{tag}.jsonl"
    with open(raw_path, "w") as f:
        for e in raw:
            f.write(json.dumps(e) + "\n")

    metrics = {
        "method": args.method,
        "mode": tag,
        "dynamics": "nominal",
        "n_episodes": n,
        "success_rate": sr,
        "collision_rate": cr,
        "mean_steps": mean_steps,
        "mean_final_dist": mean_dist,
        "checkpoint": str(ckpt_path),
    }
    metrics_path = out_dir / f"metrics_{args.method}_{tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("-" * 60)
    print(f"  {args.method}[{tag}] | SR={sr:.3f}  CR={cr:.3f}  "
          f"mean_steps={mean_steps:.1f}  mean_final_dist={mean_dist:.3f}  (n={n})")
    print(f"  Raw     -> {raw_path}")
    print(f"  Metrics -> {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
