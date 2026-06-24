#!/usr/bin/env bash
# YOLOv8 noisy adaptive self-training — v2 (corrected Gen0 teacher)
# Teacher: runs/yolov8/baseline_noisy_v2/weights/best.pt (50 epochs, same budget as students)
# Strategy: per-class adaptive confidence threshold (max-F1 calibration)
# Training set: data/FICS_PCB_REMAP_NOISE/ (75% annotations corrupted)
# Prerequisite: run_yolov8_noisy_gen0_v2.sh must have completed first.
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/self_train_noisy.yaml"
OUTPUT="experiments/yolov8_st_noisy_adaptive_v2_42"
LOG="experiments/run_yolov8_noisy_st_adaptive_v2.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE/labels"

echo "=== YOLOv8 noisy adaptive self-training (v2) ===" | tee "$LOG"
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
