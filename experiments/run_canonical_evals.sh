#!/usr/bin/env bash
# Canonical evaluation pass — 31 evals, run sequentially.
# RT-DETR-l progressive and static: iter3 accepted as final generation (iter4 failed/dropped).
# Run from repo root: bash experiments/run_canonical_evals.sh 2>&1 | tee experiments/run_canonical_evals.log
set -euo pipefail
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
UV="$HOME/.local/bin/uv"

echo "=== Canonical eval batch — $(date -u +%Y-%m-%dT%H:%MZ) ==="

# ── YOLOv8 noisy adaptive ── Gen1–3 (Gen4 already canonical)
echo "--- yolov8 adaptive gen1 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_adaptive_42/iter1/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

echo "--- yolov8 adaptive gen2 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_adaptive_42/iter2/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

echo "--- yolov8 adaptive gen3 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_adaptive_42/iter3/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

# ── YOLOv8 noisy static ── Gen1–3 (Gen4 already canonical)
echo "--- yolov8 static gen1 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_static_42/iter1/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

echo "--- yolov8 static gen2 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_static_42/iter2/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

echo "--- yolov8 static gen3 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_static_42/iter3/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy.yaml

# ── YOLOv8 noisy progressive ── Gen1–4 (Gen4 provisional)
echo "--- yolov8 progressive gen1 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_progressive_42/iter1/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy_progressive.yaml

echo "--- yolov8 progressive gen2 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_progressive_42/iter2/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy_progressive.yaml

echo "--- yolov8 progressive gen3 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_progressive_42/iter3/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy_progressive.yaml

echo "--- yolov8 progressive gen4 ---"
$UV run python scripts/evaluate.py --model yolov8 \
  --weights experiments/yolov8_st_noisy_progressive_42/iter4/train/weights/best.pt \
  --config configs/yolov8/self_train_noisy_progressive.yaml

# ── YOLOv10 noisy adaptive ── Gen1–4 (all provisional)
echo "--- yolov10 adaptive gen1 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_adaptive_42/iter1/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 adaptive gen2 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_adaptive_42/iter2/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 adaptive gen3 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_adaptive_42/iter3/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 adaptive gen4 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_adaptive_42/iter4/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

# ── YOLOv10 noisy static ── Gen1–4 (all provisional)
echo "--- yolov10 static gen1 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_static_42/iter1/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 static gen2 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_static_42/iter2/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 static gen3 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_static_42/iter3/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

echo "--- yolov10 static gen4 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_static_42/iter4/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy.yaml

# ── YOLOv10 noisy progressive ── Gen1–4 (all provisional)
echo "--- yolov10 progressive gen1 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_progressive_42/iter1/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy_progressive.yaml

echo "--- yolov10 progressive gen2 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_progressive_42/iter2/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy_progressive.yaml

echo "--- yolov10 progressive gen3 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_progressive_42/iter3/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy_progressive.yaml

echo "--- yolov10 progressive gen4 ---"
$UV run python scripts/evaluate.py --model yolov10 \
  --weights experiments/yolov10_st_noisy_progressive_42/iter4/train/weights/best.pt \
  --config configs/yolov10/self_train_noisy_progressive.yaml

# ── RT-DETR-l noisy adaptive ── Gen1–3 (Gen4 already canonical)
echo "--- rtdetr_l adaptive gen1 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_adaptive_42/iter1/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

echo "--- rtdetr_l adaptive gen2 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_adaptive_42/iter2/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

echo "--- rtdetr_l adaptive gen3 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_adaptive_42/iter3/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

# ── RT-DETR-l noisy progressive ── Gen1–3 (iter3 accepted as final gen)
echo "--- rtdetr_l progressive gen1 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_progressive_42/iter1/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy_progressive.yaml

echo "--- rtdetr_l progressive gen2 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_progressive_42/iter2/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy_progressive.yaml

echo "--- rtdetr_l progressive gen3 (final) ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_progressive_42/iter3/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy_progressive.yaml

# ── RT-DETR-l noisy static ── Gen1–3 (iter3 accepted as final gen)
echo "--- rtdetr_l static gen1 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_static_42/iter1/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

echo "--- rtdetr_l static gen2 ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_static_42/iter2/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

echo "--- rtdetr_l static gen3 (final) ---"
$UV run python scripts/evaluate.py --model rtdetr_l \
  --weights experiments/rtdetr_l_st_noisy_static_42/iter3/train/weights/best.pt \
  --config configs/rtdetr_l/self_train_noisy.yaml

echo "=== All 31 evals complete — $(date -u +%Y-%m-%dT%H:%MZ) ==="
