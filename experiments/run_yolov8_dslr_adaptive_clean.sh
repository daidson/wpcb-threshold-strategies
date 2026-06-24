#!/usr/bin/env bash
# PCB-DSLR target domain — YOLOv8 adaptive threshold, CLEAN teacher (control run)
# GEN0 teacher: runs/yolov8/baseline-2/weights/best.pt (clean FICS, mAP50=0.860)
# Identical to run_yolov8_dslr_adaptive.sh in every parameter except the teacher.
# Purpose: determine whether class collapse is caused by teacher noise or by crop-scale
#   geometry (512x512 tiles vs Honorato's full 4928x3280 images). If non-IC classes
#   survive here but collapse under the noisy teacher, noise propagation is the cause.
# IC GT merged via partial_labels_dir; pseudo-labels for all other classes from teacher.
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/self_train_dslr_clean.yaml"
OUTPUT="experiments/yolov8_st_dslr_adaptive_clean_42"
LOG="experiments/run_yolov8_dslr_adaptive_clean.log"
LABELED_IMAGES="data/PCB_DSLR_CROPS_512/orig_distribution/images"
LABELED_LABELS="data/PCB_DSLR_CROPS_512/orig_distribution/labels"
WEIGHTS="runs/yolov8/baseline-2/weights/best.pt"

echo "=== YOLOv8 PCB-DSLR — adaptive self-training (CLEAN TEACHER CONTROL) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Teacher: $WEIGHTS" | tee -a "$LOG"
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
echo "--- Step 2: self-training loop (adaptive threshold, clean teacher) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
