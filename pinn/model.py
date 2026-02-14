from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor] | nn.Module:
    mapping = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "sin": lambda x: torch.sin(x),
    }
    return mapping.get(name, nn.Tanh())


class PINNModel(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: str = "tanh"):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.activation_name = activation
        self.activation = get_activation(activation)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                if self.activation_name == "sin":
                    x = torch.sin(x)
                else:
                    x = self.activation(x)
        return x
