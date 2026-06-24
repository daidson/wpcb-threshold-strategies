#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolov8 adaptive (adaptive precision-target on DEV), Gen0->Gen3
# Teacher: runs/yolov8/baseline_noisy_v3/weights/best.pt (warm_start in config)
# iter1 seeded inside SelfTrainer from the teacher with the strategy threshold.
# IC GT merged via partial_labels_dir; pseudo-labels for the other 7 classes from the student.
# Prerequisite: run_yolov8_noisy_gen0_v3.sh must have completed first.
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/self_train_dslr.yaml"
OUTPUT="experiments/yolov8_st_dslr_adaptive_v3_42"
LOG="experiments/run_yolov8_dslr_adaptive_v3.log"
LABELED_IMAGES="data/PCB_DSLR_CROPS_512/orig_distribution/images"
LABELED_LABELS="data/PCB_DSLR_CROPS_512/orig_distribution/labels"

echo "=== yolov8 PCB-DSLR adaptive self-training (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (adaptive precision-target on DEV) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
