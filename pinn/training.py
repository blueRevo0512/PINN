from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .charges import ChargeFunction
from .config import ExperimentConfig
from .model import PINNModel, get_device


class DataBuilder:
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device

    def interior_and_boundary(self) -> tuple[torch.Tensor, torch.Tensor]:
        domain_size = self.config.runtime.domain_size
        n_domain = self.config.runtime.domain_points
        n_boundary = self.config.runtime.boundary_points

        x_domain = torch.rand(n_domain, 1, device=self.device) * 2 * domain_size - domain_size
        y_domain = torch.rand(n_domain, 1, device=self.device) * 2 * domain_size - domain_size
        domain_points = torch.cat([x_domain, y_domain], dim=1)

        boundary = []
        for _ in range(n_boundary // 4):
            boundary.extend(
                [
                    [-domain_size, torch.rand(1).item() * 2 * domain_size - domain_size],
                    [domain_size, torch.rand(1).item() * 2 * domain_size - domain_size],
                    [torch.rand(1).item() * 2 * domain_size - domain_size, -domain_size],
                    [torch.rand(1).item() * 2 * domain_size - domain_size, domain_size],
                ]
            )
        boundary_points = torch.tensor(boundary, device=self.device, dtype=torch.float32)
        return domain_points, boundary_points

    def measurement_points(self, num_points: int) -> torch.Tensor:
        domain_size = self.config.runtime.domain_size
        try:
            from scipy.stats import qmc

            sample = qmc.LatinHypercube(d=2).random(n=num_points)
        except ImportError:
            sample = np.random.random((num_points, 2))
        sample = sample * 2 * domain_size - domain_size
        return torch.tensor(sample, device=self.device, dtype=torch.float32)


class ForwardTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        charge_function: ChargeFunction,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.charge_function = charge_function
        self.device = get_device()
        self.model = PINNModel(config.model.layer_sizes, config.model.activation).to(self.device)
        self.data_builder = DataBuilder(config, self.device)
        self.logger = logger or logging.getLogger(__name__)

    def _pde_loss(self, domain_points: torch.Tensor) -> torch.Tensor:
        domain_points.requires_grad_(True)
        phi = self.model(domain_points)

        grad_phi = torch.autograd.grad(
            outputs=phi.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0]
        phi_x = grad_phi[:, 0:1]
        phi_y = grad_phi[:, 1:2]

        phi_xx = torch.autograd.grad(
            outputs=phi_x.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        phi_yy = torch.autograd.grad(
            outputs=phi_y.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0][:, 1:2]

        x_coords = domain_points[:, 0:1]
        y_coords = domain_points[:, 1:2]
        rho = self.charge_function(x_coords, y_coords)
        residual = -(phi_xx + phi_yy) - rho
        return torch.mean(residual**2)

    def _boundary_loss(self, boundary_points: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.model(boundary_points) ** 2)

    def train(
        self,
        model_path: Path,
        checkpoint_path: Path,
        resume: bool,
    ) -> dict[str, Any]:
        opt_cfg = self.config.optimization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=opt_cfg.scheduler_patience,
            factor=opt_cfg.scheduler_factor,
            min_lr=1e-7,
        )

        start_epoch = 0
        best_loss = float("inf")
        loss_history: list[float] = []

        if resume and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = float(ckpt.get("best_loss", best_loss))
            loss_history = list(ckpt.get("loss_history", []))
            self.logger.info(
                "[Forward][%s] Resume from epoch=%d, best_loss=%.3e",
                self.charge_function.name,
                start_epoch,
                best_loss,
            )

        domain_points, boundary_points = self.data_builder.interior_and_boundary()
        no_improve = 0

        self.model.train()
        log_every = max(1, int(self.config.runtime.log_every))
        for epoch in range(start_epoch, opt_cfg.iterations):
            optimizer.zero_grad()
            pde_loss = self._pde_loss(domain_points)
            bc_loss = self._boundary_loss(boundary_points)
            total_loss = pde_loss + 10.0 * bc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            scheduler.step(total_loss.detach())

            loss_value = float(total_loss.item())
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "charge_type": self.charge_function.name,
                        "model_config": asdict(self.config.model),
                        "best_loss": best_loss,
                    },
                    model_path,
                )
            else:
                no_improve += 1

            if (epoch + 1) % opt_cfg.checkpoint_every == 0 or epoch == opt_cfg.iterations - 1:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_loss": best_loss,
                        "loss_history": loss_history,
                    },
                    checkpoint_path,
                )
                self.logger.info(
                    "[Forward][%s] checkpoint saved at epoch=%d -> %s",
                    self.charge_function.name,
                    epoch,
                    checkpoint_path,
                )

            if epoch == start_epoch or (epoch + 1) % log_every == 0 or epoch == opt_cfg.iterations - 1:
                current_lr = optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "[Forward][%s] epoch=%d/%d total=%.3e pde=%.3e bc=%.3e best=%.3e lr=%.2e",
                    self.charge_function.name,
                    epoch,
                    opt_cfg.iterations - 1,
                    loss_value,
                    float(pde_loss.item()),
                    float(bc_loss.item()),
                    best_loss,
                    current_lr,
                )

            if no_improve >= opt_cfg.early_stop_patience:
                self.logger.info(
                    "[Forward][%s] early stop at epoch=%d (patience=%d)",
                    self.charge_function.name,
                    epoch,
                    opt_cfg.early_stop_patience,
                )
                break

        return {
            "best_loss": best_loss,
            "last_epoch": len(loss_history) - 1,
            "loss_history": loss_history,
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
        }

    def load_best(self, model_path: Path) -> None:
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])


class InverseTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        charge_function: ChargeFunction,
        forward_model: PINNModel,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.charge_function = charge_function
        self.device = get_device()
        layer_sizes = config.model.layer_sizes.copy()
        layer_sizes[-1] = 2
        self.model = PINNModel(layer_sizes, config.model.activation).to(self.device)
        self.forward_model = forward_model.to(self.device)
        self.data_builder = DataBuilder(config, self.device)
        self.logger = logger or logging.getLogger(__name__)

    def _inverse_pde_loss(self, domain_points: torch.Tensor) -> torch.Tensor:
        domain_points.requires_grad_(True)
        outputs = self.model(domain_points)
        phi_pred = outputs[:, 0:1]
        rho_pred = outputs[:, 1:2]

        grad_phi = torch.autograd.grad(
            outputs=phi_pred.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0]
        phi_x = grad_phi[:, 0:1]
        phi_y = grad_phi[:, 1:2]

        phi_xx = torch.autograd.grad(
            outputs=phi_x.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        phi_yy = torch.autograd.grad(
            outputs=phi_y.sum(), inputs=domain_points, create_graph=True, retain_graph=True
        )[0][:, 1:2]

        residual = -(phi_xx + phi_yy) - rho_pred
        return torch.mean(residual**2)

    def _boundary_loss(self, boundary_points: torch.Tensor) -> torch.Tensor:
        outputs = self.model(boundary_points)
        return torch.mean(outputs[:, 0:1] ** 2)

    def _data_loss(self, m_points: torch.Tensor, phi_measured: torch.Tensor) -> torch.Tensor:
        pred = self.model(m_points)[:, 0:1]
        return torch.mean((pred - phi_measured) ** 2)

    def _build_measurements(self, num_points: int, noise_percent: float) -> tuple[torch.Tensor, torch.Tensor]:
        m_points = self.data_builder.measurement_points(num_points)
        self.forward_model.eval()
        with torch.no_grad():
            phi_true = self.forward_model(m_points)
        noise_std = float(phi_true.detach().std().item()) * (noise_percent / 100.0)
        noise = torch.randn_like(phi_true) * noise_std
        return m_points, phi_true + noise

    def _evaluate(self) -> dict[str, float]:
        x = torch.linspace(-self.config.runtime.domain_size, self.config.runtime.domain_size, 100, device=self.device)
        y = torch.linspace(-self.config.runtime.domain_size, self.config.runtime.domain_size, 100, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

        self.model.eval()
        with torch.no_grad():
            rho_pred = self.model(grid_points)[:, 1].cpu().numpy().flatten()
            rho_true = self.charge_function(grid_points[:, 0:1], grid_points[:, 1:2]).cpu().numpy().flatten()

        mse = float(np.mean((rho_pred - rho_true) ** 2))
        mae = float(np.mean(np.abs(rho_pred - rho_true)))
        corr = float(np.corrcoef(rho_pred, rho_true)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        return {"mse": mse, "mae": mae, "correlation": corr}

    def train(
        self,
        num_points: int,
        noise_percent: float,
        model_path: Path,
        checkpoint_path: Path,
        resume: bool,
    ) -> dict[str, Any]:
        opt_cfg = self.config.optimization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=opt_cfg.scheduler_patience,
            factor=opt_cfg.scheduler_factor,
            min_lr=1e-7,
        )

        start_epoch = 0
        best_loss = float("inf")
        loss_history: list[float] = []

        if resume and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = float(ckpt.get("best_loss", best_loss))
            loss_history = list(ckpt.get("loss_history", []))
            self.logger.info(
                "[Inverse][%s|pts=%d|noise=%g%%] Resume from epoch=%d, best_loss=%.3e",
                self.charge_function.name,
                num_points,
                noise_percent,
                start_epoch,
                best_loss,
            )

        domain_points, boundary_points = self.data_builder.interior_and_boundary()
        m_points, phi_measured = self._build_measurements(num_points, noise_percent)

        no_improve = 0
        self.model.train()
        log_every = max(1, int(self.config.runtime.log_every))
        for epoch in range(start_epoch, opt_cfg.iterations):
            optimizer.zero_grad()
            pde_loss = self._inverse_pde_loss(domain_points)
            bc_loss = self._boundary_loss(boundary_points)
            data_loss = self._data_loss(m_points, phi_measured)
            total_loss = pde_loss + 10.0 * bc_loss + 100.0 * data_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            scheduler.step(total_loss.detach())

            loss_value = float(total_loss.item())
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                no_improve = 0
                metrics = self._evaluate()
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "model_config": asdict(self.config.model),
                        "num_points": num_points,
                        "noise_percent": noise_percent,
                        "metrics": metrics,
                        "best_loss": best_loss,
                    },
                    model_path,
                )
            else:
                no_improve += 1

            if (epoch + 1) % opt_cfg.checkpoint_every == 0 or epoch == opt_cfg.iterations - 1:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_loss": best_loss,
                        "loss_history": loss_history,
                    },
                    checkpoint_path,
                )
                self.logger.info(
                    "[Inverse][%s|pts=%d|noise=%g%%] checkpoint saved at epoch=%d -> %s",
                    self.charge_function.name,
                    num_points,
                    noise_percent,
                    epoch,
                    checkpoint_path,
                )

            if epoch == start_epoch or (epoch + 1) % log_every == 0 or epoch == opt_cfg.iterations - 1:
                current_lr = optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "[Inverse][%s|pts=%d|noise=%g%%] epoch=%d/%d total=%.3e pde=%.3e bc=%.3e data=%.3e best=%.3e lr=%.2e",
                    self.charge_function.name,
                    num_points,
                    noise_percent,
                    epoch,
                    opt_cfg.iterations - 1,
                    loss_value,
                    float(pde_loss.item()),
                    float(bc_loss.item()),
                    float(data_loss.item()),
                    best_loss,
                    current_lr,
                )

            if no_improve >= opt_cfg.early_stop_patience:
                self.logger.info(
                    "[Inverse][%s|pts=%d|noise=%g%%] early stop at epoch=%d (patience=%d)",
                    self.charge_function.name,
                    num_points,
                    noise_percent,
                    epoch,
                    opt_cfg.early_stop_patience,
                )
                break

        metrics = self._evaluate()
        return {
            "best_loss": best_loss,
            "last_epoch": len(loss_history) - 1,
            "loss_history": loss_history,
            "metrics": metrics,
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
            "num_points": num_points,
            "noise_percent": noise_percent,
        }

    def export_arrays(self, npz_path: Path) -> None:
        x = torch.linspace(-self.config.runtime.domain_size, self.config.runtime.domain_size, 100, device=self.device)
        y = torch.linspace(-self.config.runtime.domain_size, self.config.runtime.domain_size, 100, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(grid_points)
            rho_pred = outputs[:, 1].cpu().numpy().reshape(X.shape)
        rho_true = self.charge_function(grid_points[:, 0:1], grid_points[:, 1:2]).cpu().numpy().reshape(X.shape)
        np.savez_compressed(npz_path, X=X.cpu().numpy(), Y=Y.cpu().numpy(), rho_pred=rho_pred, rho_true=rho_true)
