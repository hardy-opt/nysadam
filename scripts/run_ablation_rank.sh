#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NysAdam: Rank Ablation Study
# ═══════════════════════════════════════════════════════════════════
# Author: Hardik Tankaria | Date: May 2026
# Description: Ablation over Nyström rank m = {0, 4, 8, 16, 32, 64, 128, 256}
#   on CIFAR-10 and PINN to demonstrate monotonic improvement and
#   saturation at m ≈ d_eff. Produces Figure 4 in the paper.
# Usage: bash scripts/run_ablation_rank.sh [--experiment cifar10]
# ═══════════════════════════════════════════════════════════════════

set -e

EXPERIMENT=${1:-cifar10}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-results/ablation_rank}

RANKS=(0 4 8 16 32 64 128 256)

echo "═══════════════════════════════════════════════════════════════"
echo "  Rank Ablation: $EXPERIMENT"
echo "  Ranks: ${RANKS[*]}"
echo "═══════════════════════════════════════════════════════════════"

for m in "${RANKS[@]}"; do
    echo ""
    echo ">>> Running: rank m=$m"
    python -m nysadam.experiments.run_all \
        --experiment $EXPERIMENT \
        --optimizers nysadam \
        --nys_rank $m \
        --seed $SEED \
        --output_dir $OUTPUT_DIR

    # Rename result file to include rank
    if [ -f "$OUTPUT_DIR/${EXPERIMENT}_nysadam.json" ]; then
        mv "$OUTPUT_DIR/${EXPERIMENT}_nysadam.json" \
           "$OUTPUT_DIR/${EXPERIMENT}_nysadam_rank${m}.json"
    fi
done

# Also run AdamW baseline
echo ""
echo ">>> Running: AdamW baseline"
python -m nysadam.experiments.run_all \
    --experiment $EXPERIMENT \
    --optimizers adamw \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Rank Ablation Complete: $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════"
