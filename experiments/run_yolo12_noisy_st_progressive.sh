#!/usr/bin/env bash
# YOLO12m noisy progressive self-training
# Teacher: runs/yolo12/baseline_noisy/weights/best.pt
# Strategy: progressive threshold schedule (gen1 0.35 → gen2 0.45 → gen3 0.55)
# Training set: data/FICS_PCB_REMAP_NOISE/ (75% annotations corrupted)
# Prerequisite: run_yolo12_noisy_gen0.sh must have completed first.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_noisy_progressive.yaml"
OUTPUT="experiments/yolo12_st_noisy_progressive_42"
LOG="experiments/run_yolo12_noisy_st_progressive.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE/labels"

echo "=== YOLO12m noisy progressive self-training ===" | tee "$LOG"
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
