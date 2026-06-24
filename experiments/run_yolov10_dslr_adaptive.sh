#!/usr/bin/env bash
# PCB-DSLR target domain — YOLOv10 adaptive threshold (seed=42)
# GEN0 teacher: runs/yolov10/baseline_noisy/weights/best.pt (FICS-noisy trained)
# IC GT merged via partial_labels_dir; pseudo-labels for all other classes from teacher
set -euo pipefail

MODEL="yolov10"
CONFIG="configs/yolov10/self_train_dslr.yaml"
OUTPUT="experiments/yolov10_st_dslr_adaptive_42"
LOG="experiments/run_yolov10_dslr_adaptive.log"
LABELED_IMAGES="data/PCB_DSLR_CROPS_512/orig_distribution/images"
LABELED_LABELS="data/PCB_DSLR_CROPS_512/orig_distribution/labels"
WEIGHTS="runs/yolov10/baseline_noisy/weights/best.pt"

echo "=== YOLOv10 PCB-DSLR — adaptive self-training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Labeled set: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: generate GEN1 pseudo-labels ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/generate_pseudo_labels.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --images "$LABELED_IMAGES" \
    --output "$OUTPUT/iter1/pseudo/labels" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 2: self-training loop (adaptive threshold) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
