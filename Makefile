PYTHON := PYTHONPATH=. uv run python
MODELS  := yolov8 yolov10 yolo12

.PHONY: help smoke train-% pseudo-% self-train-% eval-% baseline-all

help:
	@echo "Models: yolov8, yolov10, yolo12 (RT-DETR dropped from the v3 design)."
	@echo ""
	@echo "Baseline training (clean base.yaml):"
	@echo "  make train-yolov8          train YOLOv8 baseline"
	@echo "  make train-yolov10         train YOLOv10 baseline"
	@echo "  make baseline-all          run the clean baselines sequentially"
	@echo "  NOTE: the canonical v3 Gen0 baselines train on base_noisy.yaml, not base.yaml."
	@echo "        YOLO12 ships only the noisy configs (no configs/yolo12/base.yaml or"
	@echo "        self_train.yaml). Point the generic targets at a real config with CONFIG=,"
	@echo "        e.g.  make train-yolo12 CONFIG=configs/yolo12/base_noisy.yaml"
	@echo ""
	@echo "Smoke test (set epochs: 1 in config first):"
	@echo "  make smoke-yolov8"
	@echo ""
	@echo "Pseudo-label generation (requires WEIGHTS=<path/to/best.pt>):"
	@echo "  make pseudo-yolov8 WEIGHTS=runs/yolov8/baseline_noisy_v3/weights/best.pt \\"
	@echo "       CONFIG=configs/yolov8/self_train_noisy.yaml"
	@echo ""
	@echo "Self-training (full noisy v3 loop — use the launch scripts):"
	@echo "  bash experiments/run_yolov8_noisy_st_adaptive_v3.sh"
	@echo "  bash experiments/run_yolov10_noisy_st_static_v3.sh"
	@echo "  bash experiments/run_yolo12_noisy_st_progressive_v3.sh"
	@echo "  (pattern: experiments/run_<model>_noisy_st_{adaptive,static,progressive}_v3.sh)"
	@echo "  The make self-train-<model> target below runs one ad-hoc loop from a config."
	@echo ""
	@echo "Evaluation (requires WEIGHTS=<path/to/best.pt>):"
	@echo "  make eval-yolov8 WEIGHTS=runs/yolov8/baseline_noisy_v3/weights/best.pt"
	@echo "  Reports on the config's val split — base.yaml = TEST (data/val). Never report DEV;"
	@echo "  evaluate.py also accepts --val-images data/test/images to force TEST explicitly."

# ── Baseline training ────────────────────────────────────────────────────────
# Override CONFIG to point any generic target at a different config, e.g. a noisy
# Gen0 or a yolo12 config:  make train-yolo12 CONFIG=configs/yolo12/base_noisy.yaml

CONFIG ?=

train-%:
	$(PYTHON) scripts/train.py --model $* --config $(if $(CONFIG),$(CONFIG),configs/$*/base.yaml)

baseline-all:
	$(MAKE) train-yolov8
	$(MAKE) train-yolov10

# ── Smoke test ───────────────────────────────────────────────────────────────

smoke-%:
	$(PYTHON) scripts/train.py --model $* --config $(if $(CONFIG),$(CONFIG),configs/$*/base.yaml)

# ── Pseudo-label generation ──────────────────────────────────────────────────

WEIGHTS ?= ""

pseudo-%:
	$(PYTHON) scripts/generate_pseudo_labels.py \
		--model $* \
		--config $(if $(CONFIG),$(CONFIG),configs/$*/self_train.yaml) \
		--weights $(WEIGHTS)

# ── Self-training ────────────────────────────────────────────────────────────
# For the canonical v3 loop prefer experiments/run_<model>_noisy_st_*_v3.sh;
# this target runs one ad-hoc loop from a config.

self-train-%:
	$(PYTHON) scripts/self_train.py \
		--model $* \
		--config $(if $(CONFIG),$(CONFIG),configs/$*/self_train.yaml) \
		--output experiments/$*_st_$(shell date +%Y%m%d)

# ── Evaluation ───────────────────────────────────────────────────────────────
# Default base.yaml evaluates on TEST (data/val). Reporting rule: never on DEV.

eval-%:
	$(PYTHON) scripts/evaluate.py \
		--model $* \
		--weights $(WEIGHTS) \
		--config $(if $(CONFIG),$(CONFIG),configs/$*/base.yaml)
