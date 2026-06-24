#!/usr/bin/env bash
# Baseline training run: YOLOv8 — Generation 0
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/base.yaml"
LOG="experiments/run_yolov8_baseline.log"

echo "=== YOLOv8 baseline training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONPATH=. .venv/bin/python scripts/train.py --model "$MODEL" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
