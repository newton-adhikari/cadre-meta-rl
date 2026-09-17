"""Dynamics perturbation model for TurtleBot3 standalone simulation.

This module defines DynamicsConfig — a dataclass that parameterises
every source of dynamics and sensor variability in the standalone
(non-Gazebo) simulation.  Its purpose is to let CADRE and other
meta-RL algorithms train across a distribution of robot conditions
so that the context encoder learns to infer the current condition
from online transition data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DynamicsConfig:
    """Per-episode dynamics and sensor configuration.
    """

    # ── Dynamics ─────────────────────────────────────────────────────────────
    wheel_slip:              float = 1.0   # Training: Uniform[0.70, 1.00]
    friction:                float = 1.0   # Training: Uniform[0.60, 1.40]
    payload_factor:          float = 1.0   # Training: Uniform[1.00, 1.80]
    lin_vel_scale:           float = 1.0   # Training: Uniform[0.70, 1.10]
    ang_vel_scale:           float = 1.0   # Training: Uniform[0.70, 1.20]
    control_latency_steps:   int   = 0     # Training: randint[0, 1]
    actuator_noise_std:      float = 0.0   # Training: Uniform[0.000, 0.020]

    # ── Sensor ────────────────────────────────────────────────────────────────
    lidar_noise_std:         float = 0.0   # Training: Uniform[0.000, 0.015]
    lidar_dropout_prob:      float = 0.0   # Training: Uniform[0.000, 0.040]
    lidar_range_scale:       float = 1.0   # Training: Uniform[0.920, 1.080]
    obs_noise_std:           float = 0.0   # Training: Uniform[0.000, 0.010]
    obs_latency_steps:       int   = 0     # Training: randint[0, 1]

    def is_nominal(self) -> bool:
        """Return True iff all parameters are at their nominal (identity) values."""
        return (
            self.wheel_slip == 1.0
            and self.friction == 1.0
            and self.payload_factor == 1.0
            and self.lin_vel_scale == 1.0
            and self.ang_vel_scale == 1.0
            and self.control_latency_steps == 0
            and self.actuator_noise_std == 0.0
            and self.lidar_noise_std == 0.0
            and self.lidar_dropout_prob == 0.0
            and self.lidar_range_scale == 1.0
            and self.obs_noise_std == 0.0
            and self.obs_latency_steps == 0
        )

    def to_vector(self) -> list:
        """Return parameters as a list of floats (useful for logging)."""
        return [
            self.wheel_slip,
            self.friction,
            self.payload_factor,
            self.lin_vel_scale,
            self.ang_vel_scale,
            float(self.control_latency_steps),
            self.actuator_noise_std,
            self.lidar_noise_std,
            self.lidar_dropout_prob,
            self.lidar_range_scale,
            self.obs_noise_std,
            float(self.obs_latency_steps),
        ]

    @classmethod
    def param_names(cls) -> list:
        """Return ordered list of parameter names (matches to_vector())."""
        return [
            "wheel_slip",
            "friction",
            "payload_factor",
            "lin_vel_scale",
            "ang_vel_scale",
            "control_latency_steps",
            "actuator_noise_std",
            "lidar_noise_std",
            "lidar_dropout_prob",
            "lidar_range_scale",
            "obs_noise_std",
            "obs_latency_steps",
        ]


#: Sampling ranges for every split.
#: Each value is (low, high) for continuous params or (lo, hi) int for latency.
DYNAMICS_RANGES: dict = {
    "train": {
        "wheel_slip":            (0.70, 1.00),
        "friction":              (0.60, 1.40),
        "payload_factor":        (1.00, 1.80),
        "lin_vel_scale":         (0.70, 1.10),
        "ang_vel_scale":         (0.70, 1.20),
        "control_latency_steps": (0, 1),        # inclusive integer range
        "actuator_noise_std":    (0.000, 0.020),
        "lidar_noise_std":       (0.000, 0.015),
        "lidar_dropout_prob":    (0.000, 0.040),
        "lidar_range_scale":     (0.920, 1.080),
        "obs_noise_std":         (0.000, 0.010),
        "obs_latency_steps":     (0, 1),
    },
    "val_iid": {
        # Same ranges as train — sampled independently with a different seed.
        "wheel_slip":            (0.70, 1.00),
        "friction":              (0.60, 1.40),
        "payload_factor":        (1.00, 1.80),
        "lin_vel_scale":         (0.70, 1.10),
        "ang_vel_scale":         (0.70, 1.20),
        "control_latency_steps": (0, 1),
        "actuator_noise_std":    (0.000, 0.020),
        "lidar_noise_std":       (0.000, 0.015),
        "lidar_dropout_prob":    (0.000, 0.040),
        "lidar_range_scale":     (0.920, 1.080),
        "obs_noise_std":         (0.000, 0.010),
        "obs_latency_steps":     (0, 1),
    },
    "ood_dyn_interp": {
        # Combinations of training-range extremes seen together — interpolation OOD.
        "wheel_slip":            (0.70, 0.75),
        "friction":              (1.25, 1.40),
        "payload_factor":        (1.60, 1.80),
        "lin_vel_scale":         (0.70, 0.75),
        "ang_vel_scale":         (0.70, 0.75),
        "control_latency_steps": (1, 1),
        "actuator_noise_std":    (0.015, 0.020),
        "lidar_noise_std":       (0.012, 0.015),
        "lidar_dropout_prob":    (0.035, 0.040),
        "lidar_range_scale":     (0.920, 0.940),
        "obs_noise_std":         (0.008, 0.010),
        "obs_latency_steps":     (1, 1),
    },
    "ood_dyn_extrap": {
        # Outside training ranges — extrapolation OOD.
        "wheel_slip":            (0.30, 0.69),
        "friction":              (0.20, 0.59),
        "payload_factor":        (1.81, 2.50),
        "lin_vel_scale":         (0.30, 0.69),
        "ang_vel_scale":         (0.30, 0.69),
        "control_latency_steps": (2, 3),
        "actuator_noise_std":    (0.030, 0.060),
        "lidar_noise_std":       (0.025, 0.060),
        "lidar_dropout_prob":    (0.060, 0.200),
        "lidar_range_scale":     (0.700, 0.890),
        "obs_noise_std":         (0.020, 0.050),
        "obs_latency_steps":     (2, 3),
    },
    "ood_sens": {
        # Sensor degradation only; dynamics at nominal.
        "wheel_slip":            (1.0, 1.0),
        "friction":              (1.0, 1.0),
        "payload_factor":        (1.0, 1.0),
        "lin_vel_scale":         (1.0, 1.0),
        "ang_vel_scale":         (1.0, 1.0),
        "control_latency_steps": (0, 0),
        "actuator_noise_std":    (0.000, 0.000),
        "lidar_noise_std":       (0.025, 0.060),
        "lidar_dropout_prob":    (0.060, 0.200),
        "lidar_range_scale":     (0.700, 0.890),
        "obs_noise_std":         (0.020, 0.050),
        "obs_latency_steps":     (2, 3),
    },
    "ood_comp": {
        # All factors simultaneously at OOD-extrap levels.
        "wheel_slip":            (0.30, 0.69),
        "friction":              (0.20, 0.59),
        "payload_factor":        (1.81, 2.50),
        "lin_vel_scale":         (0.30, 0.69),
        "ang_vel_scale":         (0.30, 0.69),
        "control_latency_steps": (2, 3),
        "actuator_noise_std":    (0.030, 0.060),
        "lidar_noise_std":       (0.025, 0.060),
        "lidar_dropout_prob":    (0.060, 0.200),
        "lidar_range_scale":     (0.700, 0.890),
        "obs_noise_std":         (0.020, 0.050),
        "obs_latency_steps":     (2, 3),
    },
    "nominal": {
        # No perturbation — used as zero-shot baseline reference.
        "wheel_slip":            (1.0, 1.0),
        "friction":              (1.0, 1.0),
        "payload_factor":        (1.0, 1.0),
        "lin_vel_scale":         (1.0, 1.0),
        "ang_vel_scale":         (1.0, 1.0),
        "control_latency_steps": (0, 0),
        "actuator_noise_std":    (0.000, 0.000),
        "lidar_noise_std":       (0.000, 0.000),
        "lidar_dropout_prob":    (0.000, 0.000),
        "lidar_range_scale":     (1.000, 1.000),
        "obs_noise_std":         (0.000, 0.000),
        "obs_latency_steps":     (0, 0),
    },
}

#: Integer parameters — sampled with randint(lo, hi+1).
_INTEGER_PARAMS = frozenset({"control_latency_steps", "obs_latency_steps"})


def sample_dynamics(split: str = "train", rng=None) -> DynamicsConfig:
    """Sample a DynamicsConfig from the named split distribution.

    DynamicsConfig with all fields sampled from the split's ranges.
    """
    import numpy as _np

    if rng is None:
        rng = _np.random

    if split not in DYNAMICS_RANGES:
        raise ValueError(
            f"Unknown dynamics split '{split}'. "
            f"Available: {sorted(DYNAMICS_RANGES.keys())}"
        )

    ranges = DYNAMICS_RANGES[split]
    kwargs = {}

    for param, (lo, hi) in ranges.items():
        if param in _INTEGER_PARAMS:
            # randint is exclusive at hi — add 1 to get inclusive range
            kwargs[param] = int(rng.randint(int(lo), int(hi) + 1))
        else:
            if lo == hi:
                kwargs[param] = float(lo)
            else:
                kwargs[param] = float(rng.uniform(lo, hi))

    return DynamicsConfig(**kwargs)
