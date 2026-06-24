#!/usr/bin/env bash
# Rerun YOLOv8 noisy static — iter4 ONLY
# Context: the original iter4/train/weights/best.pt was 198MB (corrupted; 7-epoch crash).
# This script resumes from iter3's best.pt and retrains only iter4.
# SelfTrainer skips iter1-3 (best.pt exists) and renames the partial iter4/train/ dir.
# combined/ is already intact (4193 images/labels) — merge is re-done from pseudo labels.
#
# After completion, run:
#   PYTHONPATH=. uv run python scripts/evaluate.py \
#     --model yolov8 \
#     --weights experiments/yolov8_st_noisy_static_42/iter4/train/weights/best.pt \
#     --config configs/yolov8/self_train_noisy.yaml
# and update docs/experiment-log.md (yolov8_noisy_gen1_static_v2 Gen4 entry).
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/self_train_noisy.yaml"
OUTPUT="experiments/yolov8_st_noisy_static_42"
LOG="experiments/run_yolov8_noisy_st_static_iter4_rerun.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE/labels"

echo "=== YOLOv8 noisy static — iter4 rerun ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

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

echo "" | tee -a "$LOG"
echo "--- Canonical eval (run immediately after) ---" | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. ~/.local/bin/uv run python scripts/evaluate.py \
    --model yolov8 \
    --weights "$OUTPUT/iter4/train/weights/best.pt" \
    --config "$CONFIG" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Done: $(date)" | tee -a "$LOG"
