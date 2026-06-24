#!/usr/bin/env bash
# Noise experiment — RT-DETR-x static threshold (conf=0.25, seed=42)
# Training set: data/FICS_PCB_REMAP_NOISE/ (75% annotations corrupted — 25% class-swap + 25% removal + 25% box-distortion)
# Prerequisite: configs/rtdetr_x/base_noisy.yaml must have been trained first
set -euo pipefail

MODEL="rtdetr_x"
CONFIG="configs/rtdetr_x/self_train_noisy.yaml"
OUTPUT="experiments/rtdetr_x_st_noisy_static_42"
LOG="experiments/run_rtdetr_x_noisy_st_static.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE/labels"
WEIGHTS="runs/rtdetr_x/baseline_noisy/weights/best.pt"

echo "=== RT-DETR-x noise experiment — static self-training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Labeled set: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: generate gen1 pseudo-labels ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/generate_pseudo_labels.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --output "$OUTPUT/iter1/pseudo/labels" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 2: self-training loop (static conf=0.25) ---" | tee -a "$LOG"
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
