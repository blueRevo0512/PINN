from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


class IOManager:
    def __init__(self, config: ExperimentConfig):
        run_tag = config.runtime.run_name or "run"
        output_root = Path(config.runtime.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        if config.resume:
            candidates = sorted(output_root.glob(f"{run_tag}_*"))
            if candidates:
                self.root = candidates[-1]
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.root = output_root / f"{run_tag}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root = output_root / f"{run_tag}_{timestamp}"
        self.config_dir = self.root / "config"
        self.models_dir = self.root / "models"
        self.checkpoints_dir = self.root / "checkpoints"
        self.metrics_dir = self.root / "metrics"
        self.arrays_dir = self.root / "arrays"
        self.plots_dir = self.root / "plots"
        self.logs_dir = self.root / "logs"

        for p in [
            self.root,
            self.config_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.arrays_dir,
            self.plots_dir,
            self.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: ExperimentConfig) -> Path:
        target = self.config_dir / "experiment_config.json"
        target.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def save_json(self, data: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def append_csv_row(self, path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
