#!/usr/bin/env bash
# Baseline training run: RT-DETR-x — Generation 0
set -euo pipefail

MODEL="rtdetr_x"
CONFIG="configs/rtdetr_x/base.yaml"
LOG="experiments/run_rtdetr_x_baseline.log"

echo "=== RT-DETR-x baseline training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONPATH=. .venv/bin/python scripts/train.py --model "$MODEL" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
