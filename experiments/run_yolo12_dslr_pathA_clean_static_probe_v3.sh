#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolo12 PATH A CLEAN-BASE + STATIC-0.25 PROBE (2 iters).
# The decisive noise test. Identical to run_yolo12_dslr_pathA_static_probe_v3.sh EXCEPT
# the support base is the CLEAN (pre-corruption) FICS TRAIN set data/train_clean instead
# of the 75%-corrupted FICS_PCB_REMAP_NOISE_TRAIN. Same 3324 stems, only labels differ.
# Reuses configs/yolo12/self_train_dslr_pathA_static_probe.yaml (static 0.25, iterations:2,
# Path A); only --labeled-images/--labeled-labels and the output dir change.
#
# WHY: both adaptive AND static probes on the NOISY base collapse to IC-only on DSLR (the
# threshold strategy was refuted as the cause). Honorato's same-structure loop with a CLEAN
# FICS-PCB REMAP support base keeps the abundant classes (IC/resistor/ceramic/inductor)
# growing. This run holds threshold (static 0.25) and detector constant and flips ONLY the
# support-base noise.
#   * abundant non-IC persist/grow iter1->iter2 -> NOISE was the cause; Path A viable on clean base.
#   * still collapse to IC-only -> structural (target has GT only for IC); keep Gen0 assessor for YOLO12.
#
# Teacher (warm_start) stays the NOISY-v3 Gen0 (no clean YOLO12 baseline exists); only the
# support-base LABELS change, isolating the noise variable.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_dslr_pathA_static_probe.yaml"
OUTPUT="experiments/yolo12_st_dslr_pathA_clean_static_probe_v3_42"
LOG="experiments/run_yolo12_dslr_pathA_clean_static_probe_v3.log"
LABELED_IMAGES="data/train_clean/images"
LABELED_LABELS="data/train_clean/labels"
GEN0_WEIGHTS="runs/yolo12/baseline_noisy_v3/weights/best.pt"

echo "=== yolo12 PCB-DSLR PATH A CLEAN-BASE + STATIC-0.25 probe, 2 iters (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Support base (CLEAN): $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (static conf=0.25, 2 iters, CLEAN support base retained) ---" | tee -a "$LOG"
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
echo "READ: $OUTPUT/detection_growth.csv — do abundant non-IC persist iter1->iter2 (clean base)?" | tee -a "$LOG"
