#!/usr/bin/env bash
# Gen 1 self-training: RT-DETR-l — static threshold (conf=0.25, seed=42)
set -euo pipefail

MODEL="rtdetr_l"
CONFIG="configs/rtdetr_l/self_train.yaml"
OUTPUT="experiments/rtdetr_l_st_static_42"
LOG="experiments/run_rtdetr_l_gen1_static.log"

echo "=== RT-DETR-l Gen 1 static self-training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: generate iter1 pseudo-labels ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/generate_pseudo_labels.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --weights runs/rtdetr_v3/baseline-2/weights/best.pt \
    --output "$OUTPUT/iter1/pseudo/labels" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 2: self-training loop ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --no-adaptive \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
