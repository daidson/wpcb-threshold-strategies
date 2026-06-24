#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolo12 PATH A adaptive (adaptive precision-target on DEV), Gen0->Gen3
# PATH A: FICS support base retained every iteration (pseudo_only_mode: false in config),
#         so the 7 non-IC classes keep a labeled anchor and survive across generations.
# Support base (NOISY): data/FICS_PCB_REMAP_NOISE_TRAIN — passed via --labeled-images/--labeled-labels.
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt (warm_start in config).
# iter1 seeded inside SelfTrainer from the teacher with the strategy threshold.
# PCB-DSLR IC GT merged via partial_labels_dir; pseudo-labels for the other 7 classes from the student.
# Prerequisite: run the iter1 probe (run_yolo12_dslr_pathA_probe_v3.sh) and confirm the gate passes first.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_dslr_pathA.yaml"
OUTPUT="experiments/yolo12_st_dslr_pathA_adaptive_v3_42"
LOG="experiments/run_yolo12_dslr_pathA_adaptive_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"

echo "=== yolo12 PCB-DSLR PATH A adaptive self-training (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Support base: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (adaptive precision-target on DEV, support base retained) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
