#!/usr/bin/env bash
# Noise experiment — YOLOv8 GEN0 baseline on FICS_PCB_REMAP_NOISE
# Must run before any noisy self-training scripts for this model.
# After training, verify checkpoint lands at:
#   runs/yolov8/baseline_noisy/weights/best.pt
# (Ultralytics auto-increments the name if that dir exists — update
#  configs/yolov8/self_train_noisy*.yaml warm_start if it becomes baseline_noisy-2)
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/base_noisy.yaml"
LOG="experiments/run_yolov8_noisy_gen0.log"

echo "=== YOLOv8 noisy GEN0 baseline training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONPATH=. .venv/bin/python scripts/train.py --model "$MODEL" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Checkpoint expected at: runs/yolov8/baseline_noisy/weights/best.pt" | tee -a "$LOG"
