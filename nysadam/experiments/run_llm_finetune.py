"""
Transformer Language Model Experiment Runner
=============================================
Author: Hardik Tankaria | Date: May 2026
Usage: python -m nysadam.experiments.run_llm_finetune --optimizer nysadam adamw sophia
"""
import argparse
from nysadam.experiments.run_all import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Transformer LM Experiment")
    parser.add_argument("--optimizer", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nys_rank", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    override = {"model_config": {"d_model": args.d_model, "n_layers": args.n_layers,
                                  "n_heads": max(1, args.d_model // 64)}}
    if args.epochs:
        override["epochs"] = args.epochs
    if args.nys_rank is not None:
        override["nys_rank"] = args.nys_rank

    run_experiment(
        experiment_name="transformer_lm",
        optimizer_names=args.optimizer,
        seed=args.seed,
        output_dir=args.output_dir,
        config_override=override,
    )

if __name__ == "__main__":
    main()
