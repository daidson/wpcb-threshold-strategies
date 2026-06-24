#!/usr/bin/env bash
# YOLO12m noisy adaptive self-training
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt
# Strategy: per-class adaptive confidence threshold (precision-target calibration, precision≥0.9 on held-out DEV)
# Training set: data/FICS_PCB_REMAP_NOISE_TRAIN/ (75% annotations corrupted)
# Prerequisite: run_yolo12_noisy_gen0_v3.sh must have completed first.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_noisy.yaml"
OUTPUT="experiments/yolo12_st_noisy_adaptive_v3_42"
LOG="experiments/run_yolo12_noisy_st_adaptive_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"

echo "=== YOLO12m noisy adaptive self-training (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Labeled set: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: self-training loop (adaptive per-class thresholds) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
