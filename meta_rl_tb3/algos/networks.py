"""This is the core Neural network architectures for policy and value functions.

This module provides modular neural network components that can be used
with various RL algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class NetworkConfig:
    """Configuration for neural networks.
    
    Attributes:
        hidden_sizes: List of hidden layer sizes
        activation: Activation function name ('relu', 'tanh', 'elu')
        output_activation: Output activation (None for linear)
        init_std: Initial standard deviation for output layers
        use_layer_norm: Whether to use layer normalization
        use_orthogonal_init: Use orthogonal weight initialization
        log_std_bounds: Bounds for log standard deviation (actor)
    """
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    output_activation: Optional[str] = None
    init_std: float = 0.5
    use_layer_norm: bool = False
    use_orthogonal_init: bool = True
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)


def get_activation(name: str) -> nn.Module:
    # Get activation function by name
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
    }
    return activations.get(name.lower(), nn.ReLU())


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    # Apply orthogonal initialization to a layer.

    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.
    
    A flexible MLP implementation that supports:
    - Variable number of hidden layers
    - Different activation functions
    - Optional layer normalization
    - Orthogonal initialization
    
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        
        hidden_sizes = hidden_sizes or [256, 256]
        self.input_size = input_size
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        if use_orthogonal_init:
            self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                # Use smaller gain for output layer
                if i == len(self.network) - 1 or (
                    i == len(self.network) - 2 and 
                    not isinstance(self.network[-1], nn.Linear)
                ):
                    orthogonal_init(layer, gain=0.01)
                else:
                    orthogonal_init(layer, gain=math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        """
        return self.network(x)
