#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NysAdam: Refresh Interval Ablation
# ═══════════════════════════════════════════════════════════════════
# Author: Hardik Tankaria | Date: May 2026
# Description: Ablation over Nyström refresh interval k = {1, 5, 10, 20, 50, 100}
#   to find the optimal cost/accuracy trade-off. Larger k = cheaper but
#   staler curvature info.
# Usage: bash scripts/run_ablation_refresh.sh [--experiment cifar10]
# ═══════════════════════════════════════════════════════════════════

set -e

EXPERIMENT=${1:-cifar10}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-results/ablation_refresh}

INTERVALS=(1 5 10 20 50 100)

echo "═══════════════════════════════════════════════════════════════"
echo "  Refresh Interval Ablation: $EXPERIMENT"
echo "  Intervals: ${INTERVALS[*]}"
echo "═══════════════════════════════════════════════════════════════"

for k in "${INTERVALS[@]}"; do
    echo ""
    echo ">>> Running: refresh interval k=$k"

    python -c "
import sys, yaml, json, os
sys.path.insert(0, '.')
from nysadam.experiments.run_all import run_experiment

results = run_experiment(
    experiment_name='$EXPERIMENT',
    optimizer_names=['nysadam'],
    seed=$SEED,
    output_dir='$OUTPUT_DIR',
    config_override={'nys_refresh_interval': $k},
)

# Rename to include k
src = '$OUTPUT_DIR/${EXPERIMENT}_nysadam.json'
dst = '$OUTPUT_DIR/${EXPERIMENT}_nysadam_refresh${k}.json'
if os.path.exists(src):
    os.rename(src, dst)
"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Refresh Ablation Complete: $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════"
