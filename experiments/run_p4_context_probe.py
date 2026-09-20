#!/usr/bin/env python3
"""P4: Context Identifiability Experiment.

To test whether the CADRE encoder's latent context vector z contains
extractable information about hidden dynamics/environment conditions
that the encoder was never directly told about.

Main question
-------------------
Does f_φ(τ_{1:K}) — the GRU encoder applied to transition history —
produce representations that correlate with the true (hidden) physics
parameters used to generate those transitions?


Protocol
--------
1. Load each CADRE best-checkpoint.
2. Roll out the policy on 500 evaluation episodes spanning all
   dynamics splits (train, ood_dyn_interp, ood_dyn_extrap).
3. After each episode, record:
   - z = encoder output after K transitions (the context vector)
   - The ground-truth DynamicsConfig parameters for that episode
4. Fit a linear probe (Ridge regression) from z → each scalar param.
5. Report R² and Pearson r per parameter, pooled across seeds.
6. Generate:
   - PCA of z colored by each dynamics parameter (Figure 6a)
   - UMAP of z colored by each dynamics parameter (Figure 6b)
   - Bar chart of R² per parameter (Figure 6c)

Key design constraint
---------------------
The encoder NEVER sees the DynamicsConfig — only (s, a, r, s') tuples.
The probe is purely diagnostic; it has no effect on training.

"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings("ignore", message="n_jobs")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_rl_tb3.algos.cadre import CADRE
from meta_rl_tb3.algos.networks import ContextEncoderConfig, NetworkConfig
from meta_rl_tb3.envs.physics import DynamicsConfig, sample_dynamics, DYNAMICS_RANGES
from meta_rl_tb3.tasks import GoalReachingDistribution, GoalReachingDistributionConfig
from experiments.run_p2_adaptation_curves import _make_cadre_agent, MAX_EP_STEPS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS_DEFAULT    = [7, 13, 99, 1337, 42]
N_EPISODES       = 300          # episodes per seed (300 × 5 seeds = 1500 total)
CKPT_DIR         = Path("results/p2_full")
OUTPUT_DIR       = Path("results/p4")
DYNAMICS_SPLITS  = ["train", "ood_dyn_interp", "ood_dyn_extrap"]
# Parameters to probe — continuous scalars only
PROBE_PARAMS = [
    "wheel_slip",
    "friction",
    "payload_factor",
    "lin_vel_scale",
    "ang_vel_scale",
    "actuator_noise_std",
    "lidar_noise_std",
    "lidar_dropout_prob",
    "lidar_range_scale",
    "obs_noise_std",
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",       type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--n-episodes",  type=int, default=N_EPISODES)
    p.add_argument("--ckpt-dir",    type=str, default=str(CKPT_DIR))
    p.add_argument("--output-dir",  type=str, default=str(OUTPUT_DIR))
    p.add_argument("--skip-umap",   action="store_true",
                   help="Skip UMAP (faster; use when umap-learn is not installed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Context collection
# ---------------------------------------------------------------------------

def collect_context_episodes(
    agent:      CADRE,
    n_episodes: int,
    dyn_splits: List[str],
    task_dist,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out the agent and collect (z, dynamics_params) pairs."""
    n_per_split = max(1, n_episodes // len(dyn_splits))
    z_list:     List[np.ndarray] = []
    param_list: List[np.ndarray] = []

    rng = np.random.RandomState(seed_offset + 100)

    for split in dyn_splits:
        for ep_idx in range(n_per_split):
            # Sample fresh dynamics and task for this episode
            dyn   = sample_dynamics(split, rng=rng)
            task  = task_dist.sample()
            task.config.dynamics = dyn
            env   = task._default_env_fn()

            try:
                K         = agent.config.context_encoder_config.context_window
                trans_buf = deque(maxlen=K)

                obs_raw, _ = env.reset()
                obs = agent._flatten_obs(obs_raw)
                done  = False
                step  = 0

                while not done and step < MAX_EP_STEPS:
                    with torch.no_grad():
                        z_t  = agent._encode_context(trans_buf,
                                dict(agent.encoder.named_parameters()))
                        z_np = z_t.cpu().numpy().squeeze(0)
                        p_obs = (np.concatenate([obs, z_np])
                                 if agent.context_dim > 0 else obs)
                        obs_t = torch.from_numpy(p_obs).float().unsqueeze(0).to(agent.device)
                        act_t, _, _, _ = agent._forward_policy(
                            obs_t,
                            dict(agent.actor_critic.named_parameters()),
                        )
                    act_np = act_t.cpu().numpy().squeeze(0)
                    next_raw, rew, term, trunc, _ = env.step(act_np)
                    done = bool(term or trunc)
                    next_obs = agent._flatten_obs(next_raw)
                    trans_buf.append(
                        agent._build_transition(obs, act_np, rew, next_obs)
                    )
                    obs = next_obs
                    step += 1

                # Record final z after full episode
                with torch.no_grad():
                    z_final = agent._encode_context(
                        trans_buf,
                        dict(agent.encoder.named_parameters()),
                    )
                z_list.append(z_final.cpu().numpy().squeeze(0))

                # Record true dynamics params
                params_vec = np.array(
                    [getattr(dyn, p) for p in PROBE_PARAMS], dtype=np.float32
                )
                param_list.append(params_vec)

            finally:
                env.close()

    z_array     = np.array(z_list,     dtype=np.float32)   # (N, d_z)
    param_array = np.array(param_list, dtype=np.float32)   # (N, n_params)
    return z_array, param_array


# ---------------------------------------------------------------------------
# Linear probe fitting
# ---------------------------------------------------------------------------

def fit_linear_probes(
    z:      np.ndarray,
    params: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Fit one Ridge regression probe per dynamics parameter."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    results = {}

    for i, param_name in enumerate(PROBE_PARAMS):
        y = params[:, i]

        # Skip near-constant parameters (no variance to regress on)
        if y.std() < 1e-6:
            results[param_name] = {
                "r2": 0.0, "r2_cv": 0.0, "r2_std": 0.0,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "note": "near-constant — no variance",
            }
            continue

        probe = Ridge(alpha=1.0)

        # Cross-validated R²
        cv_scores = cross_val_score(probe, z_scaled, y, cv=cv, scoring="r2")
        r2_cv  = float(cv_scores.mean())
        r2_std = float(cv_scores.std())

        # Pearson r on full data (fit once for correlation)
        probe.fit(z_scaled, y)
        y_pred = probe.predict(z_scaled)
        r, p_val = pearsonr(y, y_pred)

        results[param_name] = {
            "r2_cv":      max(0.0, r2_cv),   # clip negative R² to 0 for reporting
            "r2_std":     r2_std,
            "pearson_r":  float(r),
            "pearson_p":  float(p_val),
            "raw_r2_cv":  r2_cv,             # keep the true value for diagnostics
        }

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_r2_bars(
    probe_results: Dict[str, float],
    save_path: Path,
    title: str = "Context Identifiability: R² per dynamics parameter",
) -> None:
    """Bar chart of cross-validated R² per dynamics parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable — skipping R² bar chart")
        return

    params   = list(probe_results.keys())
    r2_vals  = [max(0.0, probe_results[p]["r2_cv"]) for p in params]
    r2_stds  = [probe_results[p]["r2_std"] for p in params]
    pearson  = [abs(probe_results[p]["pearson_r"]) for p in params]

    # Friendly labels
    label_map = {
        "wheel_slip": "Wheel slip",
        "friction": "Friction",
        "payload_factor": "Payload",
        "lin_vel_scale": "Lin. vel. scale",
        "ang_vel_scale": "Ang. vel. scale",
        "actuator_noise_std": "Actuator noise",
        "lidar_noise_std": "LiDAR noise",
        "lidar_dropout_prob": "LiDAR dropout",
        "lidar_range_scale": "LiDAR range scale",
        "obs_noise_std": "Obs. noise",
    }
    labels = [label_map.get(p, p) for p in params]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(params))
    bars = ax.bar(x, r2_vals, color="#2196F3", alpha=0.8, label="R² (CV)")
    ax.errorbar(x, r2_vals, yerr=r2_stds, fmt="none", color="black",
                capsize=4, linewidth=1.5)
    ax.plot(x, pearson, "o--", color="#FF9800", markersize=6,
            label="|Pearson r| (train)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Coefficient of determination (R²)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.10, color="gray", linestyle=":", linewidth=1,
               label="R²=0.10 threshold")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  R² bar chart → {save_path}")


def plot_pca_colored(
    z:          np.ndarray,
    params:     np.ndarray,
    param_names: List[str],
    save_dir:   Path,
    top_k:      int = 4,
    probe_results: Dict = None,
) -> None:
    """PCA scatter plots of z colored by top-k most identifiable parameters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn/matplotlib unavailable — skipping PCA plots")
        return

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    pca = PCA(n_components=2, random_state=0)
    z_2d = pca.fit_transform(z_scaled)
    var_exp = pca.explained_variance_ratio_

    # Select top-k params by R² if probe_results given, else use all
    if probe_results is not None:
        ranked = sorted(param_names,
                        key=lambda p: -probe_results.get(p, {}).get("r2_cv", 0.0))
        top_params = ranked[:top_k]
    else:
        top_params = param_names[:top_k]

    fig, axes = plt.subplots(1, len(top_params),
                              figsize=(4 * len(top_params), 4))
    if len(top_params) == 1:
        axes = [axes]

    label_map = {
        "wheel_slip": "Wheel slip", "friction": "Friction",
        "payload_factor": "Payload", "lin_vel_scale": "Lin. vel. scale",
        "ang_vel_scale": "Ang. vel. scale", "actuator_noise_std": "Act. noise",
        "lidar_noise_std": "LiDAR noise", "lidar_dropout_prob": "LiDAR dropout",
        "lidar_range_scale": "LiDAR range", "obs_noise_std": "Obs. noise",
    }

    for ax, param in zip(axes, top_params):
        idx = param_names.index(param)
        c   = params[:, idx]
        sc  = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=c, cmap="viridis",
                         alpha=0.6, s=15)
        plt.colorbar(sc, ax=ax)
        r2 = probe_results.get(param, {}).get("r2_cv", float("nan")) \
            if probe_results else float("nan")
        ax.set_title(f"{label_map.get(param, param)}\n(R²={r2:.2f})",
                     fontsize=11)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("PCA of CADRE context vectors z", fontsize=13, y=1.02)
    fig.tight_layout()
    out = save_dir / "fig6a_pca_context.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA figure → {out}")


def plot_umap_colored(
    z:          np.ndarray,
    params:     np.ndarray,
    param_names: List[str],
    save_dir:   Path,
    probe_results: Dict = None,
) -> None:
    """UMAP scatter plots colored by top dynamics parameters."""
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  umap-learn not available — skipping UMAP (install: pip install umap-learn)")
        return

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    print("  Running UMAP (may take ~30s)...")
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15,
                        min_dist=0.1, metric="euclidean")
    z_2d = reducer.fit_transform(z_scaled)

    # Top-2 most identifiable params
    if probe_results is not None:
        ranked    = sorted(param_names,
                           key=lambda p: -probe_results.get(p, {}).get("r2_cv", 0))
        top_params = ranked[:2]
    else:
        top_params = param_names[:2]

    label_map = {
        "wheel_slip": "Wheel slip", "friction": "Friction",
        "payload_factor": "Payload", "lin_vel_scale": "Lin. vel. scale",
        "ang_vel_scale": "Ang. vel. scale", "actuator_noise_std": "Act. noise",
        "lidar_noise_std": "LiDAR noise", "lidar_dropout_prob": "LiDAR dropout",
        "lidar_range_scale": "LiDAR range", "obs_noise_std": "Obs. noise",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, param in zip(axes, top_params):
        idx = param_names.index(param)
        c   = params[:, idx]
        sc  = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=c, cmap="plasma",
                         alpha=0.5, s=12)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        r2 = probe_results.get(param, {}).get("r2_cv", float("nan")) \
            if probe_results else float("nan")
        ax.set_title(f"{label_map.get(param, param)}  (R²={r2:.2f})",
                     fontsize=12)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle("UMAP of CADRE context vectors z", fontsize=13)
    fig.tight_layout()
    out = save_dir / "fig6b_umap_context.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP figure → {out}")

