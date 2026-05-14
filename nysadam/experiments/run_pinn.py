"""
PINN Experiment Runner
======================
Author: Hardik Tankaria | Date: May 2026
Usage: python -m nysadam.experiments.run_pinn --optimizer nysadam adamw
"""
import argparse
from nysadam.experiments.run_all import run_experiment

def main():
    parser = argparse.ArgumentParser(description="PINN Burgers Experiment")
    parser.add_argument("--optimizer", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nys_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    override = {}
    if args.epochs:
        override["epochs"] = args.epochs
    if args.nys_rank is not None:
        override["nys_rank"] = args.nys_rank

    run_experiment(
        experiment_name="pinn_burgers",
        optimizer_names=args.optimizer,
        seed=args.seed,
        output_dir=args.output_dir,
        config_override=override if override else None,
    )

if __name__ == "__main__":
    main()
