"""
CIFAR-10/100 Experiment Runner
==============================
Author: Hardik Tankaria
Date: May 2026
Description:
    Standalone runner for CIFAR experiments with extra options
    (model selection, per-experiment hyperparameter tuning).

Usage:
    python -m nysadam.experiments.run_cifar --dataset cifar10 --optimizer nysadam
    python -m nysadam.experiments.run_cifar --dataset cifar100 --optimizer adamw --epochs 200
"""

import argparse
from nysadam.experiments.run_all import run_experiment


def main():
    parser = argparse.ArgumentParser(description="CIFAR Experiment")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--optimizer", type=str, nargs="+", default=None,
                        help="Optimizers to test (default: all)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nys_rank", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    override = {}
    if args.epochs:
        override["epochs"] = args.epochs
    if args.nys_rank is not None:
        override["nys_rank"] = args.nys_rank
    if args.lr is not None:
        override["lr"] = args.lr

    exp_name = args.dataset
    run_experiment(
        experiment_name=exp_name,
        optimizer_names=args.optimizer,
        seed=args.seed,
        output_dir=args.output_dir,
        config_override=override if override else None,
    )


if __name__ == "__main__":
    main()
