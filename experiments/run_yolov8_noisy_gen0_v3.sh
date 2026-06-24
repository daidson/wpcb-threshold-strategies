#!/usr/bin/env bash
# YOLOv8 noisy GEN0 baseline — corrected hyperparameters (v3)
# Change from v1: epochs 100→50, lr0 0.001→0.0005, patience 20→15, warmup_epochs removed.
# Purpose: align Gen0 training budget with Gen1-4 students so the Gen0→Gen4 comparison
# isolates label source (noisy human vs pseudo-labels), not training effort.
# Checkpoint lands at: runs/yolov8/baseline_noisy_v3/weights/best.pt
set -euo pipefail

MODEL="yolov8"
CONFIG="configs/yolov8/base_noisy.yaml"
LOG="experiments/run_yolov8_noisy_gen0_v3.log"

echo "=== YOLOv8 noisy GEN0 baseline training (v3 — corrected hyperparams) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONPATH=. ~/.local/bin/uv run python scripts/train.py --model "$MODEL" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Checkpoint expected at: runs/yolov8/baseline_noisy_v3/weights/best.pt" | tee -a "$LOG"
