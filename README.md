# 基于物理信息神经网络的静电荷分布逆向重建研究

本项目用于二维静电学场景下的 PINN 求解与逆问题重建：

- 正向问题：由电荷分布求解电势场
- 逆向问题：由测量电势反演电荷分布

支持批量矩阵实验，默认参数为：

- 电荷类型：`gaussian`、`square`
- 测量点数：`200 400 800 1500`
- 噪声百分点：`0 1 2 5`

## 目录说明

- `pinn/`：核心代码（配置、模型、训练、实验编排、可视化、日志）
- `run_pinn.py`：命令行入口
- `configs/outreach_matrix.json`：实验配置模板
- `outputs/full_matrix_rerun_20260214_112937/`：完整实验结果
- `ustcbeamer/`：课程汇报材料

## 运行方式

使用配置文件运行：

```bash
uv run run_pinn.py --config configs/outreach_matrix.json
```

或命令行直接指定参数：

```bash
uv run run_pinn.py --mode matrix --charge_types gaussian square --measurement_points 200 400 800 1500 --noise_percents 0 1 2 5 --log_level INFO --log_every 100
```

## 输出内容

每次运行会在 `output_dir/run_name_timestamp/` 下生成：

- `config/experiment_config.json`
- `models/`（最佳模型）
- `checkpoints/`（断点续训）
- `metrics/`（JSON + CSV）
- `arrays/`（网格数组）
- `plots/`（前向/逆向可视化）
- `logs/run.log`（运行日志）

矩阵总表路径：

- `metrics/<charge_type>/inverse_matrix_metrics.csv`

## 联系方式

- **QQ**: 2328036454
- **Email**: [jintianhao@mail.ustc.edu.cn](mailto:jintianhao@mail.ustc.edu.cn)