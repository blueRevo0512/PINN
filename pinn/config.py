from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    layer_sizes: list[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    activation: str = "tanh"


@dataclass
class OptimizationConfig:
    iterations: int = 12000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    scheduler_patience: int = 500
    scheduler_factor: float = 0.8
    early_stop_patience: int = 2000
    checkpoint_every: int = 500


@dataclass
class RuntimeConfig:
    seed: int = 42
    domain_size: float = 1.0
    grid_resolution: int = 100
    domain_points: int = 4000
    boundary_points: int = 400
    output_dir: str = "./outputs"
    run_name: str = "default_run"
    log_level: str = "INFO"
    log_every: int = 100


@dataclass
class ExperimentConfig:
    mode: str = "matrix"
    charge_types: list[str] = field(default_factory=lambda: ["gaussian", "square"])
    measurement_points: list[int] = field(default_factory=lambda: [200, 400, 800, 1500])
    noise_percents: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0])
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    force_retrain_forward: bool = False
    force_retrain_inverse: bool = False
    resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config_from_json(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    model = ModelConfig(**raw.get("model", {}))
    optimization = OptimizationConfig(**raw.get("optimization", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))

    root_keys = {
        "mode",
        "charge_types",
        "measurement_points",
        "noise_percents",
        "force_retrain_forward",
        "force_retrain_inverse",
        "resume",
    }
    root_values = {k: raw[k] for k in root_keys if k in raw}

    return ExperimentConfig(
        **root_values,
        model=model,
        optimization=optimization,
        runtime=runtime,
    )
