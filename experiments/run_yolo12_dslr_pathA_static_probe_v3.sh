#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolo12 PATH A STATIC-0.25 PROBE (2 iterations).
# Threshold-strategy ablation of the adaptive probe: same NOISY support base
# (data/FICS_PCB_REMAP_NOISE_TRAIN), Path A (support base retained), but FLAT conf=0.25
# every generation (matches Honorato's PCB-DSLR GEN run) instead of adaptive precision-0.9.
#
# WHY: the adaptive probe collapsed ALL non-IC classes on DSLR (resistor 3679->0,
# ceramic ->31 by iter2), whereas Honorato's same-structure loop with a flat 0.25 keeps
# the abundant classes growing. Hypothesis: the precision-0.9 adaptive threshold starved
# the non-IC pseudo-labels. This run holds noise constant and flips only the threshold.
#
# Runs iter1 + iter2 then computes the detection-growth curve (Gen0 + iter1 + iter2 on
# PCB-DSLR @0.25). READ: do the abundant non-IC classes (resistor/ceramic/inductor)
# PERSIST/grow across generations the way Honorato's do, or still collapse?
#
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt (warm_start in config).
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_dslr_pathA_static_probe.yaml"
OUTPUT="experiments/yolo12_st_dslr_pathA_static_probe_v3_42"
LOG="experiments/run_yolo12_dslr_pathA_static_probe_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"
GEN0_WEIGHTS="runs/yolo12/baseline_noisy_v3/weights/best.pt"

echo "=== yolo12 PCB-DSLR PATH A STATIC-0.25 probe, 2 iters (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Support base (NOISY): $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (static conf=0.25, 2 iters, support base retained) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --no-adaptive \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- detection-growth (Gen0 + iter1 + iter2 on PCB-DSLR, conf=0.25) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/evaluate_pcb_dslr.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --experiment "$OUTPUT" \
    --gen0-weights "$GEN0_WEIGHTS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "READ: $OUTPUT/detection_growth.csv — do abundant non-IC classes persist iter1->iter2?" | tee -a "$LOG"
