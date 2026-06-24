#!/usr/bin/env bash
# Gen 1 self-training: YOLOv10 — adaptive threshold (per-class max-F1, seed=42)
set -euo pipefail

MODEL="yolov10"
CONFIG="configs/yolov10/self_train.yaml"
OUTPUT="experiments/yolov10_st_adaptive_42"
LOG="experiments/run_yolov10_gen1_adaptive.log"

echo "=== YOLOv10 Gen 1 adaptive self-training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- Step 1: generate iter1 pseudo-labels (static conf=0.25) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/generate_pseudo_labels.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --weights runs/yolov10/baseline/weights/best.pt \
    --output "$OUTPUT/iter1/pseudo/labels" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 2: self-training loop (adaptive from iter2 onward) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
