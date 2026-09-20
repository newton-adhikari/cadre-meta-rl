#!/usr/bin/env python3
"""Run one seed of the P2 experiment.

Each invocation trains one method for one seed and runs the full
evaluation protocol.

"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Re-use everything from the main orchestrator
from experiments.run_p2_adaptation_curves import (
    BUDGET_SCHEDULE,
    N_EVAL_EPISODES_PER_TASK,
    N_EVAL_TASKS,
    MAX_EP_STEPS,
    DYNAMICS_SPLIT,
    DynGoalFactory,
    _make_cadre_agent,
    train_cadre_agent,
    train_ppo_ft_agent,
    evaluate_cadre_at_budget,
    evaluate_ppo_ft_at_budget,
    plot_adaptation_curves,
)
from meta_rl_tb3.tasks import (
    GoalReachingDistribution,
    GoalReachingDistributionConfig,
    DynamicsDistribution,
    get_fixed_test_tasks,
)
from meta_rl_tb3.evaluation.metrics import compute_curve_metrics, aggregate_seeds, format_summary_row


def parse_args():
    p = argparse.ArgumentParser(description="P2 single-seed run")
    p.add_argument("--method",    required=True, choices=["cadre", "fomaml", "ppo_ft"])
    p.add_argument("--seed",      type=int, required=True)
    p.add_argument("--num-iters", type=int, default=1500)
    p.add_argument("--output-dir", type=str, default="results/p2_full")
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--budget-schedule", type=int, nargs="+",
                   default=BUDGET_SCHEDULE,
                   help="Override budget schedule (default: %(default)s)")
    p.add_argument("--num-inner-steps", type=int, default=None,
                   help="Override num_inner_steps (default: from config). "
                        "Set to 0 for context-only ablation.")
    p.add_argument("--encoder-type", type=str, default=None,
                   choices=["gru", "mlp_avg", "none"],
                   help="Override encoder type. Set 'gru' + --num-inner-steps 0 for context-only.")
    return p.parse_args()

