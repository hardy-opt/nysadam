"""
Logistic Regression Experiment Runner
=====================================
Author: Hardik Tankaria | Date: May 2026
Usage: python -m nysadam.experiments.run_logistic --optimizer nysadam adamw sgd
"""
import argparse
from nysadam.experiments.run_all import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Experiment")
    parser.add_argument("--optimizer", type=str, nargs="+", default=None)
    parser.add_argument("--dataset", type=str, default="digits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        experiment_name="logistic",
        optimizer_names=args.optimizer,
        seed=args.seed,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
