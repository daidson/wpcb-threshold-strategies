#!/usr/bin/env bash
# YOLOv10 noisy progressive self-training — v3 (corrected Gen0 teacher)
# Teacher: runs/yolov10/baseline_noisy_v3/weights/best.pt (50 epochs, same budget as students)
# Strategy: progressive threshold schedule (gen1 0.35 → gen2 0.45 → gen3 0.55)
# Training set: data/FICS_PCB_REMAP_NOISE_TRAIN/ (75% annotations corrupted)
# Prerequisite: run_yolov10_noisy_gen0_v3.sh must have completed first.
set -euo pipefail

MODEL="yolov10"
CONFIG="configs/yolov10/self_train_noisy_progressive.yaml"
OUTPUT="experiments/yolov10_st_noisy_progressive_v3_42"
LOG="experiments/run_yolov10_noisy_st_progressive_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"

echo "=== YOLOv10 noisy progressive self-training (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Labeled set: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: self-training loop (progressive: gen1 conf=0.35, gen2 conf=0.45, gen3 conf=0.55) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
