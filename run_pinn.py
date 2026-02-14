from __future__ import annotations

import argparse
import json

from pinn.config import ExperimentConfig, load_config_from_json
from pinn.experiment import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN electrostatics runner")
    parser.add_argument("--config", type=str, default="", help="Path to config json")
    parser.add_argument("--mode", type=str, default=None, choices=["forward", "inverse", "full", "matrix"])
    parser.add_argument("--charge_types", type=str, nargs="+", default=None)
    parser.add_argument("--measurement_points", type=int, nargs="+", default=None)
    parser.add_argument("--noise_percents", type=float, nargs="+", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--force_retrain_forward", action="store_true", default=None)
    parser.add_argument("--force_retrain_inverse", action="store_true", default=None)
    return parser


def merge_args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        config = load_config_from_json(args.config)
    else:
        config = ExperimentConfig()

    if args.mode is not None:
        config.mode = args.mode
    if args.charge_types is not None:
        config.charge_types = args.charge_types
    if args.measurement_points is not None:
        config.measurement_points = args.measurement_points
    if args.noise_percents is not None:
        config.noise_percents = args.noise_percents
    if args.iterations is not None:
        config.optimization.iterations = args.iterations
    if args.learning_rate is not None:
        config.optimization.learning_rate = args.learning_rate
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.run_name is not None:
        config.runtime.run_name = args.run_name
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.log_level is not None:
        config.runtime.log_level = args.log_level
    if args.log_every is not None:
        config.runtime.log_every = args.log_every
    if args.resume is not None:
        config.resume = args.resume
    if args.force_retrain_forward is not None:
        config.force_retrain_forward = args.force_retrain_forward
    if args.force_retrain_inverse is not None:
        config.force_retrain_inverse = args.force_retrain_inverse
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = merge_args_to_config(args)
    runner = ExperimentRunner(config)
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
