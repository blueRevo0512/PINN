from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self):
        plt.style.use("default")

    def plot_forward(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        phi_pred: np.ndarray,
        rho_true: np.ndarray,
        charge_type: str,
        save_path: Path,
    ) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].contourf(X, Y, phi_pred, levels=20, cmap="RdBu_r")
        axes[0].set_title("Predicted Electric Potential φ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].contourf(X, Y, rho_true, levels=20, cmap="viridis")
        axes[1].set_title("True Charge Distribution ρ")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1])

        fig.suptitle(f"Forward Problem - {charge_type}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_inverse(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rho_pred: np.ndarray,
        rho_true: np.ndarray,
        charge_type: str,
        num_points: int,
        noise_percent: float,
        mse: float,
        correlation: float,
        save_path: Path,
    ) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im1 = axes[0].contourf(X, Y, rho_true, levels=20, cmap="viridis")
        axes[0].set_title("True ρ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].contourf(X, Y, rho_pred, levels=20, cmap="viridis")
        axes[1].set_title("Reconstructed ρ")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1])

        err = np.abs(rho_pred - rho_true)
        im3 = axes[2].contourf(X, Y, err, levels=20, cmap="Reds")
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[2])

        info = (
            f"Charge: {charge_type}\n"
            f"Points: {num_points}\n"
            f"Noise(%): {noise_percent:g}\n"
            f"MSE: {mse:.2e}\n"
            f"Corr: {correlation:.3f}"
        )
        fig.text(
            0.02,
            0.95,
            info,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
