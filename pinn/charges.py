from __future__ import annotations

import torch


class ChargeFunction:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SquareCharge(ChargeFunction):
    def __init__(self):
        super().__init__("square")

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = (torch.abs(x) <= 0.3) & (torch.abs(y) <= 0.3)
        return mask.float() * 5.0


class GaussianCharge(ChargeFunction):
    def __init__(self):
        super().__init__("gaussian")

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r_squared = x**2 + y**2
        return 10.0 * torch.exp(-r_squared / (2 * 0.2**2))


class RingCharge(ChargeFunction):
    def __init__(self):
        super().__init__("ring")

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(x**2 + y**2)
        mask = (r >= 0.3) & (r <= 0.5)
        return mask.float() * 8.0


CHARGE_FUNCTIONS = {
    "square": SquareCharge,
    "gaussian": GaussianCharge,
    "ring": RingCharge,
}


def get_charge_function(charge_type: str) -> ChargeFunction:
    if charge_type not in CHARGE_FUNCTIONS:
        raise ValueError(f"unknown charge type: {charge_type}")
    return CHARGE_FUNCTIONS[charge_type]()
