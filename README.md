# CADRE — Context Encoding Improves Few-Shot Adaptation Consistency in Meta-RL for Robot Navigation


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# cadre-meta-rl
A meta-reinforcement learning (meta-RL) method that combines FOMAML with a causal GRU context encoder to learn from recent transition history and rapidly adapt to changing environment dynamics, without requiring privileged physics labels or a reconstruction loss.

## Results so far

On single-task goal-reaching with structured dynamics perturbations (wheel slip, friction, payload, velocity scaling, sensor noise):

| Method | SR@0 | SR@2ep | SR@20ep | AUC-SR |
|--------|------|--------|---------|--------|
| FOMAML (n=7) | 0.950 ± 0.073 | 0.832 ± 0.147 | 0.701 ± 0.215 | 3.13 |
| CADRE (n=7) | 0.779 ± 0.266 | 0.891 ± 0.083 | 0.697 ± 0.262 | 3.03 |
| **CADRE-ctx** (n=7) | **0.961 ± 0.076** | **0.972 ± 0.041**★ | **0.934 ± 0.115** | **3.42** |
| PPO-FT (n=5) | 0.979 ± 0.044 | 1.000 ± 0.000 | 0.980 ± 0.045 | 3.55 |


**Sim-to-sim transfer:** Both methods transfer zero-shot to Gazebo Classic (TurtleBot3 Burger, ODE physics): CADRE-ctx 20/20, FOMAML 18/20 on goal-reaching.