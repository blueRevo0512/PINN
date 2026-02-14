from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .charges import get_charge_function
from .config import ExperimentConfig
from .io_manager import IOManager
from .logging_utils import setup_logger
from .training import ForwardTrainer, InverseTrainer
from .visualization import Visualizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.io = IOManager(config)
        self.io.save_config(config)
        log_file = self.io.logs_dir / "run.log"
        logger_name = f"pinn.{self.io.root.name}"
        self.logger = setup_logger(logger_name, log_file, config.runtime.log_level)
        set_seed(config.runtime.seed)
        self.visualizer = Visualizer()
        self.logger.info("Run root: %s", self.io.root)
        self.logger.info("Seed=%d Device=%s", config.runtime.seed, "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(
            "Mode=%s Charges=%s Points=%s Noise(%%)=%s Iter=%d",
            config.mode,
            config.charge_types,
            config.measurement_points,
            config.noise_percents,
            config.optimization.iterations,
        )

    def _create_grid(self) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        n = self.config.runtime.grid_resolution
        ds = self.config.runtime.domain_size
        x = np.linspace(-ds, ds, n)
        y = np.linspace(-ds, ds, n)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        return X, Y, torch.tensor(points, device=getattr(torch, "device", lambda _: None)("cpu"))

    def _export_forward_plot(self, charge_type: str, forward_trainer: ForwardTrainer) -> str:
        n = self.config.runtime.grid_resolution
        ds = self.config.runtime.domain_size
        x = torch.linspace(-ds, ds, n, device=forward_trainer.device)
        y = torch.linspace(-ds, ds, n, device=forward_trainer.device)
        X_t, Y_t = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([X_t.flatten(), Y_t.flatten()], dim=1)

        forward_trainer.model.eval()
        with torch.no_grad():
            phi_pred = forward_trainer.model(points).cpu().numpy().reshape(X_t.shape)
        rho_true = (
            forward_trainer.charge_function(points[:, 0:1], points[:, 1:2])
            .cpu()
            .numpy()
            .reshape(X_t.shape)
        )

        X = X_t.cpu().numpy()
        Y = Y_t.cpu().numpy()
        plot_path = self.io.plots_dir / charge_type / "forward.png"
        self.visualizer.plot_forward(X, Y, phi_pred, rho_true, charge_type, plot_path)
        return str(plot_path)

    def _export_inverse_plot(
        self,
        charge_type: str,
        key: str,
        array_path: Path,
        num_points: int,
        noise_percent: float,
        mse: float,
        correlation: float,
    ) -> str:
        data = np.load(array_path)
        X = data["X"]
        Y = data["Y"]
        rho_pred = data["rho_pred"]
        rho_true = data["rho_true"]
        plot_path = self.io.plots_dir / charge_type / f"inverse_{key}.png"
        self.visualizer.plot_inverse(
            X,
            Y,
            rho_pred,
            rho_true,
            charge_type,
            num_points,
            noise_percent,
            mse,
            correlation,
            plot_path,
        )
        return str(plot_path)

    def _run_forward(self, charge_type: str) -> tuple[ForwardTrainer, dict[str, Any]]:
        charge = get_charge_function(charge_type)
        trainer = ForwardTrainer(self.config, charge, logger=self.logger)

        model_path = self.io.models_dir / charge_type / "forward_best.pth"
        checkpoint_path = self.io.checkpoints_dir / charge_type / "forward_last.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if model_path.exists() and not self.config.force_retrain_forward:
            self.logger.info("[Forward][%s] loading best model: %s", charge_type, model_path)
            trainer.load_best(model_path)
            plot_path = self._export_forward_plot(charge_type, trainer)
            return trainer, {"loaded": True, "model_path": str(model_path), "plot_path": plot_path}

        self.logger.info("[Forward][%s] start training", charge_type)
        result = trainer.train(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            resume=self.config.resume,
        )
        result["plot_path"] = self._export_forward_plot(charge_type, trainer)
        self.io.save_json(result, self.io.metrics_dir / charge_type / "forward_metrics.json")
        return trainer, result

    def _run_inverse_matrix(self, charge_type: str, forward_trainer: ForwardTrainer) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        fields = [
            "charge_type",
            "num_points",
            "noise_percent",
            "mse",
            "mae",
            "correlation",
            "best_loss",
            "model_path",
            "plot_path",
        ]
        csv_path = self.io.metrics_dir / charge_type / "inverse_matrix_metrics.csv"

        for num_points in self.config.measurement_points:
            for noise_percent in self.config.noise_percents:
                key = f"pts_{num_points}_noise_{noise_percent:g}"
                inv_trainer = InverseTrainer(
                    self.config,
                    forward_trainer.charge_function,
                    forward_trainer.model,
                    logger=self.logger,
                )

                model_path = self.io.models_dir / charge_type / f"inverse_{key}_best.pth"
                checkpoint_path = self.io.checkpoints_dir / charge_type / f"inverse_{key}_last.pth"
                array_path = self.io.arrays_dir / charge_type / f"inverse_{key}.npz"

                model_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                array_path.parent.mkdir(parents=True, exist_ok=True)

                if model_path.exists() and not self.config.force_retrain_inverse:
                    self.logger.info("[Inverse][%s] loading best model: %s", key, model_path)
                    checkpoint = torch.load(model_path, map_location=inv_trainer.device)
                    metrics = checkpoint.get("metrics", {})
                    loaded = torch.load(model_path, map_location=inv_trainer.device)
                    inv_trainer.model.load_state_dict(loaded["model_state_dict"])
                    inv_trainer.export_arrays(array_path)
                    plot_path = self._export_inverse_plot(
                        charge_type,
                        key,
                        array_path,
                        num_points,
                        noise_percent,
                        float(metrics.get("mse", 0.0) or 0.0),
                        float(metrics.get("correlation", 0.0) or 0.0),
                    )
                    row = {
                        "charge_type": charge_type,
                        "num_points": num_points,
                        "noise_percent": noise_percent,
                        "mse": metrics.get("mse"),
                        "mae": metrics.get("mae"),
                        "correlation": metrics.get("correlation"),
                        "best_loss": checkpoint.get("best_loss"),
                        "model_path": str(model_path),
                        "plot_path": plot_path,
                    }
                else:
                    self.logger.info(
                        "[Inverse][%s] start training charge=%s points=%d noise=%g%%",
                        key,
                        charge_type,
                        num_points,
                        noise_percent,
                    )
                    result = inv_trainer.train(
                        num_points=num_points,
                        noise_percent=noise_percent,
                        model_path=model_path,
                        checkpoint_path=checkpoint_path,
                        resume=self.config.resume,
                    )
                    inv_trainer.export_arrays(array_path)
                    plot_path = self._export_inverse_plot(
                        charge_type,
                        key,
                        array_path,
                        num_points,
                        noise_percent,
                        float(result["metrics"]["mse"]),
                        float(result["metrics"]["correlation"]),
                    )
                    self.io.save_json(
                        result,
                        self.io.metrics_dir / charge_type / f"inverse_{key}_metrics.json",
                    )
                    row = {
                        "charge_type": charge_type,
                        "num_points": num_points,
                        "noise_percent": noise_percent,
                        "mse": result["metrics"]["mse"],
                        "mae": result["metrics"]["mae"],
                        "correlation": result["metrics"]["correlation"],
                        "best_loss": result["best_loss"],
                        "model_path": result["model_path"],
                        "plot_path": plot_path,
                    }

                self.io.append_csv_row(csv_path, fields, row)
                records.append(row)

        self.io.save_json(
            {"charge_type": charge_type, "records": records},
            self.io.metrics_dir / charge_type / "inverse_matrix_metrics.json",
        )
        return records

    def run(self) -> dict[str, Any]:
        output: dict[str, Any] = {"run_root": str(self.io.root), "charges": {}}
        for charge_type in self.config.charge_types:
            self.logger.info("Start charge pipeline: %s", charge_type)
            forward_trainer, forward_result = self._run_forward(charge_type)

            charge_result: dict[str, Any] = {"forward": forward_result}
            if self.config.mode in ["inverse", "full", "matrix"]:
                matrix = self._run_inverse_matrix(charge_type, forward_trainer)
                charge_result["inverse_matrix"] = matrix
            output["charges"][charge_type] = charge_result
            self.logger.info("Done charge pipeline: %s", charge_type)

        self.io.save_json(output, self.io.metrics_dir / "run_summary.json")
        self.logger.info("Run finished. Summary: %s", self.io.metrics_dir / "run_summary.json")
        return output
