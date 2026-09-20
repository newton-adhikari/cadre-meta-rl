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
    pass