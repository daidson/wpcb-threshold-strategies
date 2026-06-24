#!/usr/bin/env bash
# YOLO12m noisy static self-training
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt
# Strategy: static confidence threshold (conf=0.25, no adaptation)
# Training set: data/FICS_PCB_REMAP_NOISE_TRAIN/ (75% annotations corrupted)
# Prerequisite: run_yolo12_noisy_gen0_v3.sh must have completed first.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_noisy.yaml"
OUTPUT="experiments/yolo12_st_noisy_static_v3_42"
LOG="experiments/run_yolo12_noisy_st_static_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"

echo "=== YOLO12m noisy static self-training (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Labeled set: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: self-training loop (static conf=0.25) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --no-adaptive \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
