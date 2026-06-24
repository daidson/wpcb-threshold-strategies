#!/usr/bin/env bash
# Noise experiment — RT-DETR-l adaptive threshold (per-class max-F1, seed=42)
# Training set: data/FICS_PCB_REMAP_NOISE/ (75% annotations corrupted — 25% class-swap + 25% removal + 25% box-distortion)
# Prerequisite: configs/rtdetr_l/base_noisy.yaml must have been trained first
set -euo pipefail

MODEL="rtdetr_l"
CONFIG="configs/rtdetr_l/self_train_noisy.yaml"
OUTPUT="experiments/rtdetr_l_st_noisy_adaptive_42"
LOG="experiments/run_rtdetr_l_noisy_st_adaptive.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE/labels"
WEIGHTS="runs/rtdetr_l/baseline_noisy/weights/best.pt"

echo "=== RT-DETR-l noise experiment — adaptive self-training ===" | tee "$LOG"
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
    --images "$LABELED_IMAGES" \
    --output "$OUTPUT/iter1/pseudo/labels" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 2: self-training loop (adaptive per-class thresholds) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
