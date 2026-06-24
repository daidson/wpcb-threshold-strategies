#!/usr/bin/env bash
# Canonical evaluation pass (v3) — 30 evals on the held-out TEST set, run sequentially.
#
# v3 design: models train on data/FICS_PCB_REMAP_NOISE_TRAIN, monitor best.pt on the
# clean held-out DEV split (data/dev), and adaptive calibrates its precision-target
# thresholds on DEV. TEST (= data/test, a byte-for-byte alias of data/val) is read by
# NOTHING but this final eval — so --val-images data/test/images is passed everywhere.
#
# 30 = 3 models × (Gen0 + 3 strategies × 3 generations).
# Each eval writes results/v3/<tag>/<model>_metrics.json (val_images recorded inside).
#
# Run from repo root:
#   bash experiments/run_canonical_evals_v3.sh 2>&1 | tee experiments/run_canonical_evals_v3.log
set -euo pipefail
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
UV="$HOME/.local/bin/uv"
TEST_IMAGES="data/test/images"

echo "=== Canonical eval batch v3 (TEST = $TEST_IMAGES) — $(date -u +%Y-%m-%dT%H:%MZ) ==="

evaluate () {  # $1=model  $2=config  $3=weights  $4=tag
  local model="$1" config="$2" weights="$3" tag="$4"
  echo "--- $tag ---"
  if [[ ! -f "$weights" ]]; then
    echo "MISSING WEIGHTS: $weights — skipping (run the training first)" >&2
    return 0
  fi
  $UV run python scripts/evaluate.py \
    --model "$model" --config "$config" --weights "$weights" \
    --val-images "$TEST_IMAGES" --output "results/v3/$tag"
}

for model in yolov8 yolov10 yolo12; do
  base_cfg="configs/$model/base_noisy.yaml"
  st_cfg="configs/$model/self_train_noisy.yaml"
  prog_cfg="configs/$model/self_train_noisy_progressive.yaml"

  # Gen0 teacher
  evaluate "$model" "$base_cfg" \
    "runs/$model/baseline_noisy_v3/weights/best.pt" "${model}_gen0"

  # Gen1–3 for each strategy
  for gen in 1 2 3; do
    evaluate "$model" "$st_cfg" \
      "experiments/${model}_st_noisy_adaptive_v3_42/iter${gen}/train/weights/best.pt" \
      "${model}_adaptive_gen${gen}"
    evaluate "$model" "$st_cfg" \
      "experiments/${model}_st_noisy_static_v3_42/iter${gen}/train/weights/best.pt" \
      "${model}_static_gen${gen}"
    evaluate "$model" "$prog_cfg" \
      "experiments/${model}_st_noisy_progressive_v3_42/iter${gen}/train/weights/best.pt" \
      "${model}_progressive_gen${gen}"
  done
done

echo "=== Canonical eval batch v3 complete — $(date -u +%Y-%m-%dT%H:%MZ) ==="
