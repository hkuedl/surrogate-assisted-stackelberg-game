"""Train and evaluate one L2O surrogate model per microgrid instance.

This is the headless counterpart of ``train_l2o_model.ipynb``. Run it from the
project root, or from any working directory: paths are resolved from this file.

Expected input files:
    data/X_SB1.npy, data/Y_SB1.npy
    data/X_SB2.npy, data/Y_SB2.npy
    data/X_BG1.npy, data/Y_BG1.npy
    data/X_BG2.npy, data/Y_BG2.npy
    data/X_IP1.npy, data/Y_IP1.npy

Checkpoints and metadata are written to ``NNmodel/``.

Server example::

    python -u train_l2o_model.py --scenarios SB1 SB2 BG1 BG2 IP1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from L2Otraining import (
    SUPPORTED_INSTANCES,
    TrainingConfig,
    evaluate,
    load_data,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_SCENARIOS = tuple(SUPPORTED_INSTANCES)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate L2O surrogate models."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(ALL_SCENARIOS),
        choices=ALL_SCENARIOS,
        help="Follower instances to train. Default: SB1 SB2 BG1 BG2 IP1.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate saved checkpoints.",
    )
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help="Evaluate all five follower instances, not only the trained ones.",
    )
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> dict[str, TrainingConfig]:
    return {
        scenario: TrainingConfig(
            scenario=scenario,
            hidden_dim=args.hidden_dim,
            residual_blocks=args.residual_blocks,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        for scenario in args.scenarios
    }


def check_training_data(scenarios: list[str]) -> dict[str, dict[str, tuple[int, ...]]]:
    data_shapes = {}
    for scenario in scenarios:
        x_tensor, y_tensor = load_data(scenario)
        data_shapes[scenario] = {
            "X shape": tuple(x_tensor.shape),
            "Y shape": tuple(y_tensor.shape),
        }
        print(
            f"{scenario}: X{data_shapes[scenario]['X shape']}, "
            f"Y{data_shapes[scenario]['Y shape']}"
        )
    return data_shapes


def train_models(
    scenarios: list[str],
    configs: dict[str, TrainingConfig],
) -> dict[str, dict[str, float]]:
    training_results = {}
    for scenario in scenarios:
        print(f"\n{'=' * 20} Training {scenario} {'=' * 20}")
        training_results[scenario] = train(scenario, config=configs[scenario])
        print(f"{scenario} training result: {training_results[scenario]}")
    return training_results


def evaluate_models(scenarios: list[str]) -> dict[str, dict[str, float]]:
    evaluation_results = {}
    for scenario in scenarios:
        evaluation_results[scenario] = evaluate(scenario)
    return evaluation_results


def main() -> None:
    _configure_stdio()
    args = parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Training device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Scenarios: {args.scenarios}")

    configs = build_configs(args)
    check_training_data(args.scenarios)

    if not args.skip_train:
        train_models(args.scenarios, configs)
    else:
        print("Skipping training.")

    evaluate_scenarios = list(ALL_SCENARIOS) if args.evaluate_all else list(args.scenarios)
    print(f"\nEvaluating: {evaluate_scenarios}")
    evaluation_results = evaluate_models(evaluate_scenarios)
    print("Evaluation summary:")
    for scenario, metrics in evaluation_results.items():
        print(f"  {scenario}: {metrics}")


if __name__ == "__main__":
    main()
