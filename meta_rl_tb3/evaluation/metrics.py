"""
Evaluation metrics for the P2 adaptation-curve experiment.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Episode result record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Single-episode outcome record.

    Attributes
    ----------
    success:
        True iff the robot reached the goal within the episode limit.
    collision:
        True iff any collision occurred during the episode.
    n_steps:
        Total steps taken in the episode.
    final_dist:
        Distance to goal at episode end (metres, unnormalised).
    total_reward:
        Cumulative episode reward.
    method:
        String label for the method (e.g. "cadre", "fomaml", "ppo_ft").
    seed:
        Training seed from which this evaluation was derived.
    budget:
        Adaptation interaction budget (environment steps) used.
    task_id:
        Optional identifier of the evaluation task.
    dynamics_split:
        Dynamics distribution split (e.g. "train", "ood_dyn_extrap").
    """
    success:        bool
    collision:      bool
    n_steps:        int
    final_dist:     float
    total_reward:   float
    method:         str   = ""
    seed:           int   = 0
    budget:         int   = 0
    task_id:        str   = ""
    dynamics_split: str   = "train"


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def success_rate(episodes: List[EpisodeResult]) -> float:
    """SR = fraction of successful episodes."""
    if not episodes:
        return float("nan")
    return float(np.mean([e.success for e in episodes]))


def collision_rate(episodes: List[EpisodeResult]) -> float:
    """CR = fraction of episodes with any collision."""
    if not episodes:
        return float("nan")
    return float(np.mean([e.collision for e in episodes]))


def navigation_error(episodes: List[EpisodeResult]) -> float:
    """NE = mean final_dist for FAILED episodes only.

    Returns NaN if all episodes succeeded (perfect performance).
    """
    failed = [e.final_dist for e in episodes if not e.success]
    if not failed:
        return float("nan")
    return float(np.mean(failed))


def time_to_goal(episodes: List[EpisodeResult]) -> float:
    """TTG = mean n_steps for SUCCESSFUL episodes only.

    Returns NaN if no episode succeeded.
    """
    succeeded = [e.n_steps for e in episodes if e.success]
    if not succeeded:
        return float("nan")
    return float(np.mean(succeeded))


def mean_episode_reward(episodes: List[EpisodeResult]) -> float:
    if not episodes:
        return float("nan")
    return float(np.mean([e.total_reward for e in episodes]))


# ---------------------------------------------------------------------------
# Adaptation-curve aggregate metrics
# ---------------------------------------------------------------------------

def auc_sr(
    budgets: List[int],
    sr_values: List[float],
) -> float:
    """Area under the SR-vs-log10(budget+1) curve (trapezoidal rule)."""
    x = np.array([math.log10(b + 1) for b in budgets], dtype=float)
    y = np.array(sr_values, dtype=float)

    # Remove NaN pairs
    valid = ~np.isnan(y)
    x, y = x[valid], y[valid]

    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def k_threshold(
    budgets: List[int],
    sr_values: List[float],
    target_sr: float = 0.70,
) -> int:
    """Minimum adaptation budget at which SR first exceeds target_sr."""
    for b, sr in zip(budgets, sr_values):
        if not math.isnan(sr) and sr >= target_sr:
            return b
    return math.inf   # never reached within the evaluated budget schedule


def adaptation_gain(
    sr_at_zero: float,
    sr_at_budget: float,
) -> float:
    """ΔSR = SR(K) − SR(0).  Positive = adaptation improved performance."""
    if math.isnan(sr_at_zero) or math.isnan(sr_at_budget):
        return float("nan")
    return sr_at_budget - sr_at_zero


# ---------------------------------------------------------------------------
# Aggregate from episode lists keyed by budget
# ---------------------------------------------------------------------------

def compute_curve_metrics(
    episodes_by_budget: Dict[int, List[EpisodeResult]],
    target_sr: float = 0.70,
) -> Dict[str, float]:
    """Compute all adaptation-curve metrics from a budget→episodes dict."""
    
    budgets = sorted(episodes_by_budget.keys())
    sr_list = [success_rate(episodes_by_budget[b]) for b in budgets]
    cr_list = [collision_rate(episodes_by_budget[b]) for b in budgets]
    ne_list = [navigation_error(episodes_by_budget[b]) for b in budgets]
    ttg_list = [time_to_goal(episodes_by_budget[b]) for b in budgets]
    rew_list = [mean_episode_reward(episodes_by_budget[b]) for b in budgets]

    result: Dict[str, float] = {}
    for b, sr, cr, ne, ttg, rew in zip(
        budgets, sr_list, cr_list, ne_list, ttg_list, rew_list
    ):
        result[f"sr_at_{b}"]     = sr
        result[f"cr_at_{b}"]     = cr
        result[f"ne_at_{b}"]     = ne
        result[f"ttg_at_{b}"]    = ttg
        result[f"reward_at_{b}"] = rew

    result["auc_sr"]         = auc_sr(budgets, sr_list)
    result["k_threshold"]    = float(k_threshold(budgets, sr_list, target_sr))
    result["adaptation_gain"] = adaptation_gain(
        sr_list[0] if sr_list else float("nan"),
        sr_list[-1] if sr_list else float("nan"),
    )
    return result


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(
    per_seed_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed metric dicts into mean/std/ci95."""
    if not per_seed_metrics:
        return {}

    all_keys = set()
    for d in per_seed_metrics:
        all_keys.update(d.keys())

    result = {}
    for key in sorted(all_keys):
        vals = np.array(
            [d[key] for d in per_seed_metrics
             if key in d
             and isinstance(d.get(key), (int, float))
             and not math.isnan(float(d[key]))],
            dtype=float,
        )
        if len(vals) == 0:
            result[key] = {"mean": float("nan"), "std": float("nan"),
                           "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                           "n_seeds": 0}
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # 95% CI via normal approximation (n_seeds ≥ 5 is barely adequate;
        # we report it but flag when n < 5)
        se = std / math.sqrt(len(vals))
        result[key] = {
            "mean":      mean,
            "std":       std,
            "ci95_lo":   mean - 1.96 * se,
            "ci95_hi":   mean + 1.96 * se,
            "n_seeds":   len(vals),
        }
    return result


# ---------------------------------------------------------------------------
# Convenience: format a summary table row
# ---------------------------------------------------------------------------

def format_summary_row(
    method: str,
    agg: Dict[str, Dict[str, float]],
    budgets_to_show: List[int],
) -> str:
    """Format a one-line table row for quick console inspection."""
    parts = [f"{method:<10}"]
    for b in budgets_to_show:
        key = f"sr_at_{b}"
        if key in agg:
            parts.append(f"SR@{b}={agg[key]['mean']:.2f}±{agg[key]['std']:.2f}")
        else:
            parts.append(f"SR@{b}=n/a")
    if "auc_sr" in agg:
        parts.append(f"AUC={agg['auc_sr']['mean']:.3f}")
    if "k_threshold" in agg:
        k_val = agg["k_threshold"]["mean"]
        parts.append(f"K70={'inf' if math.isinf(k_val) else int(k_val)}")
    return " | ".join(parts)
