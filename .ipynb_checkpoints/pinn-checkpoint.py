#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN静电学问题求解器 - 简化版（修复版）
专注于核心功能和高效训练
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ================== 设备和环境配置 ==================
def setup_device():
    """自动检测并配置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        print(f"使用GPU: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用Apple MPS")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    return device

DEVICE = setup_device()
TORCH_VERSION = torch.__version__

# ================== 配置管理 ==================
@dataclass
class ModelConfig:
    """神经网络配置"""
    layer_sizes: List[int] = None
    activation: str = "tanh"
    
    def __post_init__(self):
        if self.layer_sizes is None:
            self.layer_sizes = [2, 64, 64, 64, 1]  # 默认网络结构

@dataclass 
class OptimizationConfig:
    """优化配置"""
    adam_iterations: int = 12000  # 主要依赖Adam
    adam_lr: float = 0.001
    scheduler_patience: int = 500
    scheduler_factor: float = 0.8
    weight_decay: float = 1e-6
    grad_clip: float = 1.0

@dataclass
class ExperimentConfig:
    """实验配置"""
    mode: str = "full"  # forward, inverse, full
    charge_type: str = "square"
    output_base_dir: str = "./outputs"
    
    # 核心配置
    model_config: ModelConfig = None
    optimization_config: OptimizationConfig = None
    
    # 实验参数
    noise_levels: List[float] = None
    measurement_points: List[int] = None
    
    # 网格和批处理
    domain_size: float = 1.0
    grid_resolution: int = 100
    batch_size: int = 2048
    
    # 训练控制
    force_retrain_fwd: bool = False
    force_retrain_inv: bool = False
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.02]
        if self.measurement_points is None:
            self.measurement_points = [200, 400]

# ================== 电荷分布函数 ==================
class ChargeFunction:
    """电荷分布基类"""
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SquareCharge(ChargeFunction):
    """方形电荷分布"""
    def __init__(self):
        super().__init__("square")
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = (torch.abs(x) <= 0.3) & (torch.abs(y) <= 0.3)
        return mask.float() * 5.0

class GaussianCharge(ChargeFunction):
    """高斯电荷分布"""
    def __init__(self):
        super().__init__("gaussian")
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r_squared = x**2 + y**2
        return 10.0 * torch.exp(-r_squared / (2 * 0.2**2))

class RingCharge(ChargeFunction):
    """环形电荷分布"""
    def __init__(self):
        super().__init__("ring")
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(x**2 + y**2)
        mask = (r >= 0.3) & (r <= 0.5)
        return mask.float() * 8.0

# 电荷函数工厂
CHARGE_FUNCTIONS = {
    "square": SquareCharge,
    "gaussian": GaussianCharge, 
    "ring": RingCharge
}

def get_charge_function(charge_type: str) -> ChargeFunction:
    """获取电荷函数"""
    if charge_type not in CHARGE_FUNCTIONS:
        raise ValueError(f"未知电荷类型: {charge_type}")
    return CHARGE_FUNCTIONS[charge_type]()

# ================== 神经网络模型 ==================
def get_activation(activation_name: str):
    """获取激活函数"""
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "sin": lambda x: torch.sin(x)
    }
    return activations.get(activation_name, nn.Tanh())

class PINNModel(nn.Module):
    """PINN神经网络模型"""
    
    def __init__(self, layer_sizes: List[int], activation: str = "tanh"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.activation = get_activation(activation)
        self.activation_name = activation
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不应用激活函数
                if self.activation_name == "sin":
                    x = torch.sin(x)
                else:
                    x = self.activation(x)
        return x

# ================== 核心PINN求解器 ==================
class PINNElectrostatics:
    """PINN静电学求解器"""
    
    def __init__(self, config: ExperimentConfig, charge_function: ChargeFunction):
        self.config = config
        self.charge_function = charge_function
        self.logger = logging.getLogger(__name__)
        
        # 创建模型
        self.model = PINNModel(
            config.model_config.layer_sizes,
            config.model_config.activation
        ).to(DEVICE)
        
        self.logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def generate_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练数据"""
        domain_size = self.config.domain_size
        
        # 内部域点
        n_domain = 4000
        x_domain = torch.rand(n_domain, 1, device=DEVICE) * 2 * domain_size - domain_size
        y_domain = torch.rand(n_domain, 1, device=DEVICE) * 2 * domain_size - domain_size
        domain_points = torch.cat([x_domain, y_domain], dim=1)
        
        # 边界点
        n_boundary = 400
        boundary_points = []
        
        # 四条边界
        for _ in range(n_boundary // 4):
            # 左右边界
            boundary_points.extend([
                [-domain_size, torch.rand(1).item() * 2 * domain_size - domain_size],
                [domain_size, torch.rand(1).item() * 2 * domain_size - domain_size],
                [torch.rand(1).item() * 2 * domain_size - domain_size, -domain_size],
                [torch.rand(1).item() * 2 * domain_size - domain_size, domain_size]
            ])
        
        boundary_points = torch.tensor(boundary_points, device=DEVICE, dtype=torch.float32)
        
        return domain_points, boundary_points
    
    def compute_pde_loss(self, domain_points: torch.Tensor) -> torch.Tensor:
        """计算PDE损失"""
        domain_points.requires_grad_(True)
        
        # 前向传播
        phi = self.model(domain_points)
        
        # 计算一阶导数
        grad_phi = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0]
        
        phi_x = grad_phi[:, 0:1]
        phi_y = grad_phi[:, 1:2]
        
        # 计算二阶导数
        phi_xx = torch.autograd.grad(
            outputs=phi_x.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        phi_yy = torch.autograd.grad(
            outputs=phi_y.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # 计算电荷密度
        x_coords = domain_points[:, 0:1]
        y_coords = domain_points[:, 1:2]
        rho = self.charge_function(x_coords, y_coords)
        
        # 泊松方程: -∇²φ = ρ
        laplacian_phi = phi_xx + phi_yy
        pde_residual = -laplacian_phi - rho
        
        return torch.mean(pde_residual**2)
    
    def compute_boundary_loss(self, boundary_points: torch.Tensor) -> torch.Tensor:
        """计算边界条件损失"""
        phi_boundary = self.model(boundary_points)
        # 边界条件: φ = 0
        return torch.mean(phi_boundary**2)
    
    def train_forward_efficient(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """高效的正向问题训练"""
        self.logger.info("开始正向问题训练（高效版）")
        start_time = time.time()
        
        # 预生成训练数据
        domain_points, boundary_points = self.generate_training_data()
        
        opt_config = self.config.optimization_config
        
        # 只使用Adam优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config.adam_lr,
            weight_decay=opt_config.weight_decay
        )
        
        # 学习率调度器（兼容性修复）
        try:
            # 尝试使用verbose参数（新版本PyTorch）
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=opt_config.scheduler_patience,
                factor=opt_config.scheduler_factor,
                min_lr=1e-7,
                verbose=True
            )
        except TypeError:
            # 回退到不使用verbose参数（旧版本PyTorch）
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=opt_config.scheduler_patience,
                factor=opt_config.scheduler_factor,
                min_lr=1e-7
            )
            self.logger.info("使用兼容模式的学习率调度器")
        
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 2000
        
        self.model.train()
        
        for epoch in range(opt_config.adam_iterations):
            optimizer.zero_grad()
            
            # 计算损失
            pde_loss = self.compute_pde_loss(domain_points)
            boundary_loss = self.compute_boundary_loss(boundary_points)
            total_loss = pde_loss + 10.0 * boundary_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                opt_config.grad_clip
            )
            
            optimizer.step()
            scheduler.step(total_loss)
            
            # 记录损失
            loss_value = total_loss.item()
            loss_history.append(loss_value)
            
            # 早停检查
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 日志输出
            if epoch % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch}: Loss={loss_value:.2e}, "
                    f"PDE={pde_loss.item():.2e}, "
                    f"BC={boundary_loss.item():.2e}, "
                    f"LR={current_lr:.2e}"
                )
            
            # 收敛检查
            if loss_value < 1e-7 or patience_counter > max_patience:
                self.logger.info(f"训练在第{epoch}轮收敛或达到耐心限制")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 保存模型
        if save_path:
            self.save_model(save_path, {
                'loss_history': loss_history,
                'training_time': training_time,
                'final_loss': best_loss
            })
        
        return {
            'model': self.model,
            'loss_history': loss_history,
            'training_time': training_time,
            'final_loss': best_loss
        }
    
    def save_model(self, path: str, metadata: Dict[str, Any] = None):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': asdict(self.config.model_config),
            'charge_type': self.charge_function.name,
            'device': str(DEVICE),
            'torch_version': TORCH_VERSION
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path + '.pth')
        self.logger.info(f"模型已保存: {path}.pth")
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """加载模型"""
        checkpoint = torch.load(path + '.pth', map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型已加载: {path}.pth")
        return checkpoint

# ================== 逆向问题求解器 ==================
class InverseProblemSolver:
    """逆向问题求解器"""
    
    def __init__(self, config: ExperimentConfig, forward_pinn: PINNElectrostatics):
        self.config = config
        self.forward_pinn = forward_pinn
        self.logger = logging.getLogger(__name__)
        
        # 创建逆向模型（输入坐标，输出电势和电荷密度）
        layer_sizes = config.model_config.layer_sizes.copy()
        layer_sizes[-1] = 2  # 输出电势和电荷密度
        
        self.inverse_model = PINNModel(
            layer_sizes,
            config.model_config.activation
        ).to(DEVICE)
    
    def generate_measurement_data(self, num_points: int, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成测量数据"""
        try:
            # 尝试导入scipy
            from scipy.stats import qmc
            use_scipy = True
        except ImportError:
            use_scipy = False
            self.logger.warning("scipy不可用，使用随机采样代替拉丁超立方采样")
        
        if use_scipy:
            # 使用拉丁超立方采样
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=num_points)
        else:
            # 使用均匀随机采样
            sample = np.random.random((num_points, 2))
        
        # 映射到域范围
        domain_size = self.config.domain_size
        sample = sample * 2 * domain_size - domain_size
        
        measurement_points = torch.tensor(sample, device=DEVICE, dtype=torch.float32)
        
        # 使用正向模型预测电势
        self.forward_pinn.model.eval()
        with torch.no_grad():
            phi_true = self.forward_pinn.model(measurement_points)
        
        # 添加噪声
        noise = torch.randn_like(phi_true) * noise_level
        phi_measured = phi_true + noise
        
        return measurement_points, phi_measured
    
    def solve_inverse(self, measurement_points: torch.Tensor, phi_measured: torch.Tensor, 
                     charge_function: ChargeFunction, save_path: Optional[str] = None) -> Dict[str, Any]:
        """求解逆向问题"""
        self.logger.info(f"开始逆向问题求解，测量点数: {len(measurement_points)}")
        start_time = time.time()
        
        # 生成域内训练点
        domain_points, boundary_points = self.forward_pinn.generate_training_data()
        
        opt_config = self.config.optimization_config
        
        optimizer = optim.AdamW(
            self.inverse_model.parameters(),
            lr=opt_config.adam_lr,
            weight_decay=opt_config.weight_decay
        )
        
        # 学习率调度器（兼容性修复）
        try:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=opt_config.scheduler_patience,
                factor=opt_config.scheduler_factor,
                min_lr=1e-7,
                verbose=True
            )
        except TypeError:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=opt_config.scheduler_patience,
                factor=opt_config.scheduler_factor,
                min_lr=1e-7
            )
        
        loss_history = []
        best_loss = float('inf')
        
        self.inverse_model.train()
        
        for epoch in range(opt_config.adam_iterations):
            optimizer.zero_grad()
            
            # 计算各项损失
            pde_loss = self._compute_inverse_pde_loss(domain_points)
            boundary_loss = self._compute_inverse_boundary_loss(boundary_points)
            data_loss = self._compute_data_fitting_loss(measurement_points, phi_measured)
            
            # 总损失（加权）
            total_loss = pde_loss + 10.0 * boundary_loss + 100.0 * data_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), opt_config.grad_clip)
            optimizer.step()
            scheduler.step(total_loss)
            
            loss_value = total_loss.item()
            loss_history.append(loss_value)
            
            if loss_value < best_loss:
                best_loss = loss_value
            
            if epoch % 1000 == 0:
                self.logger.info(
                    f"Epoch {epoch}: Total={loss_value:.2e}, "
                    f"PDE={pde_loss.item():.2e}, "
                    f"BC={boundary_loss.item():.2e}, "
                    f"Data={data_loss.item():.2e}"
                )
            
            if loss_value < 1e-6:
                self.logger.info(f"逆向训练在第{epoch}轮收敛")
                break
        
        training_time = time.time() - start_time
        
        # 评估结果
        metrics = self._evaluate_solution(charge_function)
        
        # 保存模型
        if save_path:
            self._save_inverse_model(save_path, {
                'loss_history': loss_history,
                'training_time': training_time,
                'metrics': metrics
            })
        
        return {
            'model': self.inverse_model,
            'loss_history': loss_history,
            'training_time': training_time,
            'metrics': metrics
        }
    
    def _compute_inverse_pde_loss(self, domain_points: torch.Tensor) -> torch.Tensor:
        """计算逆向PDE损失"""
        domain_points.requires_grad_(True)
        
        # 预测电势和电荷密度
        outputs = self.inverse_model(domain_points)
        phi_pred = outputs[:, 0:1]
        rho_pred = outputs[:, 1:2]
        
        # 计算电势的二阶导数
        grad_phi = torch.autograd.grad(
            outputs=phi_pred.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0]
        
        phi_x = grad_phi[:, 0:1]
        phi_y = grad_phi[:, 1:2]
        
        phi_xx = torch.autograd.grad(
            outputs=phi_x.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        phi_yy = torch.autograd.grad(
            outputs=phi_y.sum(),
            inputs=domain_points,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # 泊松方程: -∇²φ = ρ
        laplacian_phi = phi_xx + phi_yy
        pde_residual = -laplacian_phi - rho_pred
        
        return torch.mean(pde_residual**2)
    
    def _compute_inverse_boundary_loss(self, boundary_points: torch.Tensor) -> torch.Tensor:
        """计算逆向边界损失"""
        outputs = self.inverse_model(boundary_points)
        phi_boundary = outputs[:, 0:1]
        return torch.mean(phi_boundary**2)
    
    def _compute_data_fitting_loss(self, measurement_points: torch.Tensor, phi_measured: torch.Tensor) -> torch.Tensor:
        """计算数据拟合损失"""
        outputs = self.inverse_model(measurement_points)
        phi_pred = outputs[:, 0:1]
        return torch.mean((phi_pred - phi_measured)**2)
    
    def _evaluate_solution(self, charge_function: ChargeFunction) -> Dict[str, float]:
        """评估求解结果"""
        # 创建评估网格
        x = torch.linspace(-self.config.domain_size, self.config.domain_size, 100, device=DEVICE)
        y = torch.linspace(-self.config.domain_size, self.config.domain_size, 100, device=DEVICE)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # 预测和真实值
        self.inverse_model.eval()
        with torch.no_grad():
            outputs = self.inverse_model(grid_points)
            rho_pred = outputs[:, 1].cpu().numpy()
            
            x_coords = grid_points[:, 0:1]
            y_coords = grid_points[:, 1:2]
            rho_true = charge_function(x_coords, y_coords).cpu().numpy().flatten()
        
        # 计算指标
        mse = np.mean((rho_pred - rho_true)**2)
        mae = np.mean(np.abs(rho_pred - rho_true))
        
        # 相关系数
        correlation = np.corrcoef(rho_pred, rho_true)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation)
        }
    
    def _save_inverse_model(self, path: str, metadata: Dict[str, Any]):
        """保存逆向模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.inverse_model.state_dict(),
            'model_config': asdict(self.config.model_config),
            'device': str(DEVICE),
            'torch_version': TORCH_VERSION
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path + '.pth')
        self.logger.info(f"逆向模型已保存: {path}.pth")

# ================== 可视化模块 ==================
class Visualizer:
    """结果可视化"""
    
    def __init__(self):
        plt.style.use('default')
    
    def create_grid(self, resolution: int, domain_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """创建预测网格"""
        x = np.linspace(-domain_size, domain_size, resolution)
        y = np.linspace(-domain_size, domain_size, resolution)
        return np.meshgrid(x, y)
    
    def predict_on_grid(self, model: nn.Module, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """在网格上预测"""
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        points_tensor = torch.tensor(points, device=DEVICE, dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            predictions = model(points_tensor).cpu().numpy().flatten()
        
        return predictions.reshape(X.shape)
    
    def predict_inverse_on_grid(self, inverse_solver, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """在网格上预测逆向模型的电荷密度"""
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        points_tensor = torch.tensor(points, device=DEVICE, dtype=torch.float32)
        
        inverse_solver.inverse_model.eval()
        with torch.no_grad():
            outputs = inverse_solver.inverse_model(points_tensor)
            predictions = outputs[:, 1].cpu().numpy()  # 电荷密度
        
        return predictions.reshape(X.shape)
    
    def plot_forward_results(self, X: np.ndarray, Y: np.ndarray, phi_pred: np.ndarray,
                           rho_true: np.ndarray, charge_type: str, save_path: str):
        """绘制正向问题结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 电势场
        im1 = axes[0].contourf(X, Y, phi_pred, levels=20, cmap='RdBu_r')
        axes[0].set_title('Predicted Electric Potential φ')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # 真实电荷分布
        im2 = axes[1].contourf(X, Y, rho_true, levels=20, cmap='viridis')
        axes[1].set_title('True Charge Distribution ρ')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        fig.suptitle(f'Forward Problem Results - {charge_type}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_inverse_results(self, X: np.ndarray, Y: np.ndarray, rho_pred: np.ndarray,
                           rho_true: np.ndarray, metrics: Dict[str, float], charge_type: str,
                           noise_level: float, num_points: int, save_path: str):
        """绘制逆向问题结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 真实分布
        im1 = axes[0].contourf(X, Y, rho_true, levels=20, cmap='viridis')
        axes[0].set_title('True ρ')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # 预测分布
        im2 = axes[1].contourf(X, Y, rho_pred, levels=20, cmap='viridis')
        axes[1].set_title('Reconstructed ρ')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # 误差分布
        error = np.abs(rho_pred - rho_true)
        im3 = axes[2].contourf(X, Y, error, levels=20, cmap='Reds')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        # 添加指标信息
        info_text = (
            f"Charge: {charge_type}\n"
            f"Points: {num_points}\n"
            f"Noise: {noise_level:.3f}\n"
            f"MSE: {metrics['mse']:.2e}\n"
            f"Correlation: {metrics['correlation']:.3f}"
        )
        fig.text(0.02, 0.95, info_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ================== 实验管理器 ==================
class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.visualizer = Visualizer()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        
        log_file = os.path.join(
            self.config.output_base_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行完整实验"""
        self.logger.info("PINN静电学实验开始")
        self.logger.info(f"使用设备: {DEVICE}")
        self.logger.info(f"PyTorch版本: {TORCH_VERSION}")
        
        start_time = time.time()
        
        # 获取电荷函数
        charge_function = get_charge_function(self.config.charge_type)
        
        # 创建输出目录
        model_dir = os.path.join(self.config.output_base_dir, "models")
        plots_dir = os.path.join(self.config.output_base_dir, "plots")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        results = {}
        
        # 正向问题
        if self.config.mode in ["forward", "full"]:
            forward_model_path = os.path.join(model_dir, f"forward_{self.config.charge_type}")
            
            pinn = PINNElectrostatics(self.config, charge_function)
            
            if self.config.force_retrain_fwd or not os.path.exists(forward_model_path + '.pth'):
                forward_result = pinn.train_forward_efficient(forward_model_path)
            else:
                self.logger.info("加载现有正向模型")
                checkpoint = pinn.load_model(forward_model_path)
                forward_result = {'model': pinn.model, 'checkpoint': checkpoint}
            
            results['forward'] = forward_result
            
            # 可视化正向结果
            X, Y = self.visualizer.create_grid(self.config.grid_resolution, self.config.domain_size)
            phi_pred = self.visualizer.predict_on_grid(pinn.model, X, Y)
            
            # 计算真实电荷分布
            X_tensor = torch.tensor(X, device=DEVICE, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, device=DEVICE, dtype=torch.float32)
            rho_true = charge_function(X_tensor, Y_tensor).cpu().numpy()
            
            forward_plot_path = os.path.join(plots_dir, f"forward_{self.config.charge_type}.png")
            self.visualizer.plot_forward_results(X, Y, phi_pred, rho_true, 
                                                charge_function.name, forward_plot_path)
        
        # 逆向问题
        if self.config.mode in ["inverse", "full"]:
            if self.config.mode == "inverse":
                # 仅逆向模式，需要加载正向模型
                forward_model_path = os.path.join(model_dir, f"forward_{self.config.charge_type}")
                pinn = PINNElectrostatics(self.config, charge_function)
                pinn.load_model(forward_model_path)
            
            inverse_results = {}
            
            for noise_level in self.config.noise_levels:
                for num_points in self.config.measurement_points:
                    self.logger.info(f"逆向求解: 噪声={noise_level:.3f}, 点数={num_points}")
                    
                    # 创建逆向求解器
                    inverse_solver = InverseProblemSolver(self.config, pinn)
                    
                    # 生成测量数据
                    measurement_points, phi_measured = inverse_solver.generate_measurement_data(
                        num_points, noise_level
                    )
                    
                    # 求解逆向问题
                    key = f"noise_{noise_level:.3f}_points_{num_points}"
                    inverse_model_path = os.path.join(model_dir, f"inverse_{key}")
                    
                    if self.config.force_retrain_inv or not os.path.exists(inverse_model_path + '.pth'):
                        result = inverse_solver.solve_inverse(
                            measurement_points, phi_measured, charge_function, inverse_model_path
                        )
                    else:
                        self.logger.info(f"跳过已存在的逆向模型: {key}")
                        continue
                    
                    inverse_results[key] = result
                    
                    # 可视化逆向结果
                    X, Y = self.visualizer.create_grid(self.config.grid_resolution, self.config.domain_size)
                    rho_pred = self.visualizer.predict_inverse_on_grid(inverse_solver, X, Y)
                    
                    X_tensor = torch.tensor(X, device=DEVICE, dtype=torch.float32)
                    Y_tensor = torch.tensor(Y, device=DEVICE, dtype=torch.float32)
                    rho_true = charge_function(X_tensor, Y_tensor).cpu().numpy()
                    
                    inverse_plot_path = os.path.join(plots_dir, f"inverse_{key}.png")
                    self.visualizer.plot_inverse_results(
                        X, Y, rho_pred, rho_true, result['metrics'],
                        charge_function.name, noise_level, num_points, inverse_plot_path
                    )
                    
                    self.logger.info(f"完成: MSE={result['metrics']['mse']:.2e}")
            
            results['inverse'] = inverse_results
        
        total_time = time.time() - start_time
        
        # 保存实验摘要
        summary = {
            'config': asdict(self.config),
            'total_time': total_time,
            'device': str(DEVICE),
            'torch_version': TORCH_VERSION,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.config.output_base_dir, "experiment_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"实验完成，总耗时: {total_time:.2f}秒")
        
        return results

# ================== 命令行接口 ==================
def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="PINN静电学实验 - 简化版")
    
    parser.add_argument("--mode", type=str, default="full",
                       choices=["forward", "inverse", "full"],
                       help="运行模式")
    parser.add_argument("--charge_type", type=str, default="square",
                       choices=list(CHARGE_FUNCTIONS.keys()),
                       help="电荷分布类型")
    parser.add_argument("--output_dir", type=str, default="./outputs_simple",
                       help="输出目录")
    parser.add_argument("--iterations", type=int, default=12000,
                       help="训练迭代次数")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                       default=[0.01, 0.02],
                       help="噪声水平列表")
    parser.add_argument("--measurement_points", type=int, nargs="+",
                       default=[200, 400],
                       help="测量点数列表")
    parser.add_argument("--force_retrain", action="store_true",
                       help="强制重新训练所有模型")
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 创建配置
    model_config = ModelConfig()
    
    optimization_config = OptimizationConfig(
        adam_iterations=args.iterations,
        adam_lr=args.learning_rate
    )
    
    config = ExperimentConfig(
        mode=args.mode,
        charge_type=args.charge_type,
        output_base_dir=args.output_dir,
        model_config=model_config,
        optimization_config=optimization_config,
        noise_levels=args.noise_levels,
        measurement_points=args.measurement_points,
        force_retrain_fwd=args.force_retrain,
        force_retrain_inv=args.force_retrain
    )
    
    # 运行实验
    try:
        experiment_manager = ExperimentManager(config)
        results = experiment_manager.run_experiment()
        return results
    except Exception as e:
        logging.error(f"实验失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
