"""
Observation wrapper for multi-task CADRE .

"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Canonical dimension ───────────────────────────────────────────────────────
#   lidar(360) + robot_state(5) + task_info(5) = 370
CANONICAL_OBS_DIM: int = 370

# Backward-compat alias (was 377 in the old HELM branch; now 370)
MIXED_OBS_DIM: int = CANONICAL_OBS_DIM


class CanonicalObsWrapper(gym.ObservationWrapper):
    """Pad/trim a flat Box observation to exactly CANONICAL_OBS_DIM.

    this wrapper is a no-op for well-formed environments.
    It is kept as an explicit contract-enforcement layer: if an env
    produces the wrong dimension the wrapper raises a clear error

    """

    def __init__(self, env: gym.Env, target_dim: int = CANONICAL_OBS_DIM):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "requires a flat Box observation space. "
                f"Got: {type(env.observation_space)}"
            )
        src = env.observation_space.shape[0]
        if src > target_dim:
            raise ValueError(
                f"Source obs dim ({src}) exceeds CANONICAL_OBS_DIM ({target_dim}). "
                "Increase CANONICAL_OBS_DIM or fix the task environment."
            )
        self._source_dim = src
        self._target_dim = target_dim
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(target_dim,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Zero-pad observation to target_dim (no-op when dims already match)."""
        if self._source_dim == self._target_dim:
            return obs
        padded = np.zeros(self._target_dim, dtype=np.float32)
        padded[: self._source_dim] = obs
        return padded


# Backward-compat alias
PaddedObsWrapper = CanonicalObsWrapper
