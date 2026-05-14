#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NysAdam: Run All Experiments
# ═══════════════════════════════════════════════════════════════════
# Author: Hardik Tankaria | Date: May 2026
# Description: Master script to run all benchmark experiments and
#   generate paper figures/tables.
# Usage: bash scripts/run_all_experiments.sh [--gpu 0] [--seed 42]
# ═══════════════════════════════════════════════════════════════════

set -e

# Parse arguments
GPU=${GPU:-0}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-results}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=$GPU

echo "═══════════════════════════════════════════════════════════════"
echo "  NysAdam: Full Experiment Suite"
echo "  GPU: $GPU | Seed: $SEED | Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"

# Run unit tests first
echo ""
echo ">>> Running unit tests..."
python -m pytest tests/ -v --tb=short || {
    echo "TESTS FAILED. Fix before running experiments."
    exit 1
}
echo ">>> All tests passed!"

# Experiment 1: Logistic Regression (quick)
echo ""
echo ">>> Experiment 1: Logistic Regression"
python -m nysadam.experiments.run_all \
    --experiment logistic \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

# Experiment 2: CIFAR-10
echo ""
echo ">>> Experiment 2: CIFAR-10 ResNet-18"
python -m nysadam.experiments.run_all \
    --experiment cifar10 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

# Experiment 3: CIFAR-100
echo ""
echo ">>> Experiment 3: CIFAR-100 ResNet-18"
python -m nysadam.experiments.run_all \
    --experiment cifar100 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

# Experiment 4: PINN (Burgers' equation)
echo ""
echo ">>> Experiment 4: PINN Burgers' Equation"
python -m nysadam.experiments.run_all \
    --experiment pinn_burgers \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

# Experiment 5: Transformer Language Model
echo ""
echo ">>> Experiment 5: Transformer LM (WikiText-2)"
python -m nysadam.experiments.run_all \
    --experiment transformer_lm \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

# Generate figures and tables
echo ""
echo ">>> Generating paper figures and LaTeX tables..."
python -m nysadam.utils.visualization \
    --results_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR/figures

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: $OUTPUT_DIR/"
echo "  Figures: $OUTPUT_DIR/figures/"
echo "  Tables:  $OUTPUT_DIR/tables/"
echo "═══════════════════════════════════════════════════════════════"
