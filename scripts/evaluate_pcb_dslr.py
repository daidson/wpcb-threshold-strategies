"""Detection-growth curve for PCB-DSLR target domain evaluation.

Runs inference at a fixed confidence threshold across Gen0 (teacher) and all
self-training iterations for a given experiment, counting detections per class.
Outputs a long-format CSV suitable for plotting the detection-growth curve.

PCB-DSLR has GT only for IC (class 2), so full mAP across classes is not
possible — detection count growth across generations is the standard eval for
this domain. For IC specifically, the GT *is* available, so this script also
scores the IC detections against that GT (precision/recall at IoU >= 0.5),
showing whether the extra detections are correct, not merely more numerous.

Outputs (per experiment):
  * ``<experiment>/detection_growth.csv`` — per-class detection counts per generation
  * ``<experiment>/ic_quality.csv``       — IC TP/FP/FN/precision/recall per generation

Usage:
    PYTHONPATH=. uv run python scripts/evaluate_pcb_dslr.py \\
        --model yolov8 \\
        --config configs/yolov8/self_train_dslr.yaml \\
        --experiment experiments/yolov8_st_dslr_adaptive_v3_42 \\
        --gen0-weights runs/yolov8/baseline_noisy_v3/weights/best.pt
"""
import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.models import MODEL_REGISTRY, build_model
from src.utils.io import IMAGE_EXTS, load_config
from src.utils.labels import box_iou, greedy_match, read_yolo_label, yolo_to_xyxy

IC_IOU_THR = 0.5

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_inference(
    model, images_dir: Path, conf: float, nms_iou: float, ic_class: int, limit: int | None = None
) -> tuple[dict[int, int], dict[str, list]]:
    """Run inference over all images.

    Returns ``(class_counts, ic_preds_by_stem)`` where ``ic_preds_by_stem`` maps
    image stem -> list of IC predictions as ``(cls, cx, cy, w, h)`` normalized
    tuples (for matching against the IC GT in normalized space). ``limit`` caps
    the number of images (for smoke tests).
    """
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        image_paths = image_paths[:limit]
    counts: dict[int, int] = defaultdict(int)
    ic_preds: dict[str, list] = {}
    for img_path in tqdm(image_paths, desc="  inference", leave=False):
        results = model.predict(str(img_path), conf=conf, iou=nms_iou)
        preds_here: list = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                c = int(box.cls)
                counts[c] += 1
                if c == ic_class:
                    cx, cy, w, h = box.xywhn[0].tolist()
                    preds_here.append((c, cx, cy, w, h))
        ic_preds[img_path.stem] = preds_here
    return dict(counts), ic_preds


def score_ic(ic_preds: dict[str, list], ic_labels_dir: Path, ic_class: int) -> dict[str, int]:
    """Score IC predictions against the IC GT at IoU >= 0.5 (class-constrained).

    Driven by the inference image list (every image was seen); an image with no
    GT file contributes only FP. Returns ``{'tp','fp','fn','gt'}`` totals.
    """
    tp = fp = fn = gt_total = 0
    for stem, preds in ic_preds.items():
        gt = [b for b in read_yolo_label(ic_labels_dir / f"{stem}.txt") if b[0] == ic_class]
        gt_total += len(gt)
        iou = box_iou(yolo_to_xyxy(gt), yolo_to_xyxy(preds))
        matches = greedy_match(
            iou, [b[0] for b in gt], [b[0] for b in preds],
            iou_thr=IC_IOU_THR, class_constrained=True,
        )
        tp += len(matches)
        fn += len(gt) - len(matches)
        fp += len(preds) - len(matches)
    return {"tp": tp, "fp": fp, "fn": fn, "gt": gt_total}


def checkpoints_for_experiment(experiment_dir: Path, gen0_weights: Path) -> list[tuple[str, Path]]:
    """Return [(generation_label, checkpoint_path), ...] in order."""
    entries: list[tuple[str, Path]] = [("gen0", gen0_weights)]
    for i in range(1, 5):
        pt = experiment_dir / f"iter{i}" / "train" / "weights" / "best.pt"
        if pt.exists():
            entries.append((f"iter{i}", pt))
        else:
            logger.warning("Checkpoint missing, stopping at iter%d: %s", i, pt)
            break
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCB-DSLR detection-growth curve evaluation")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", required=True,
                   help="YAML config (data.names, pseudo_label.conf_threshold + nms_iou are read)")
    p.add_argument("--experiment", required=True, help="Experiment dir, e.g. experiments/yolov8_st_dslr_adaptive_42")
    p.add_argument("--gen0-weights", required=True, help="Gen0 teacher best.pt path")
    p.add_argument("--images", default="data/PCB_DSLR_CROPS_512/orig_distribution/images",
                   help="PCB-DSLR images directory")
    p.add_argument("--ic-labels", default="data/PCB_DSLR_CROPS_512/orig_distribution/labels",
                   help="IC GT labels directory (class 2 only)")
    p.add_argument("--conf", type=float, default=None,
                   help="Fixed inference conf for all generations "
                        "(default: pseudo_label.conf_threshold from config)")
    p.add_argument("--output", default=None,
                   help="Detection-count CSV path (default: <experiment>/detection_growth.csv)")
    p.add_argument("--ic-output", default=None,
                   help="IC quality CSV path (default: <experiment>/ic_quality.csv)")
    p.add_argument("--no-ic-quality", action="store_true",
                   help="Skip IC GT scoring (detection counts only)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of images (smoke test only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    nms_iou: float = cfg["pseudo_label"]["nms_iou"]
    # Honorato-validated C_th; applied uniformly so the strategy schedule doesn't
    # leak into the detection-growth comparison. Sourced from config, not hardcoded.
    conf: float = args.conf if args.conf is not None else cfg["pseudo_label"]["conf_threshold"]
    class_names: list[str] = cfg["data"]["names"]

    experiment_dir = Path(args.experiment)
    gen0_weights = Path(args.gen0_weights)
    images_dir = Path(args.images)
    ic_labels_dir = Path(args.ic_labels)
    output_csv = Path(args.output) if args.output else experiment_dir / "detection_growth.csv"
    ic_output_csv = Path(args.ic_output) if args.ic_output else experiment_dir / "ic_quality.csv"
    ic_id = class_names.index("ic")
    do_ic = not args.no_ic_quality and ic_labels_dir.exists()
    if not args.no_ic_quality and not ic_labels_dir.exists():
        logger.warning("IC labels dir not found (%s) — skipping IC quality", ic_labels_dir)

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)
    if not gen0_weights.exists():
        logger.error("Gen0 weights not found: %s", gen0_weights)
        sys.exit(1)

    checkpoints = checkpoints_for_experiment(experiment_dir, gen0_weights)
    logger.info("Evaluating %d generations on %s", len(checkpoints), images_dir)
    logger.info("Fixed conf=%.2f, NMS IoU=%.2f, IC quality=%s", conf, nms_iou, do_ic)

    rows: list[dict] = []
    ic_rows: list[dict] = []
    for gen_label, ckpt_path in checkpoints:
        logger.info("[%s] Loading %s", gen_label, ckpt_path)
        model = build_model(args.model, cfg)
        model.load_weights(str(ckpt_path))

        counts, ic_preds = run_inference(model, images_dir, conf, nms_iou, ic_id, args.limit)
        total = sum(counts.values())
        logger.info("[%s] total detections: %d (IC: %d)", gen_label, total, counts.get(ic_id, 0))

        for cls_id, name in enumerate(class_names):
            rows.append({
                "experiment": experiment_dir.name,
                "generation": gen_label,
                "checkpoint": str(ckpt_path),
                "conf": conf,
                "class_id": cls_id,
                "class_name": name,
                "count": counts.get(cls_id, 0),
            })

        if do_ic:
            s = score_ic(ic_preds, ic_labels_dir, ic_id)
            precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
            recall = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
            logger.info(
                "[%s] IC quality: TP=%d FP=%d FN=%d  P=%.3f R=%.3f",
                gen_label, s["tp"], s["fp"], s["fn"], precision, recall,
            )
            ic_rows.append({
                "experiment": experiment_dir.name,
                "generation": gen_label,
                "checkpoint": str(ckpt_path),
                "conf": conf,
                "ic_gt": s["gt"],
                "ic_tp": s["tp"],
                "ic_fp": s["fp"],
                "ic_fn": s["fn"],
                "ic_precision": f"{precision:.4f}",
                "ic_recall": f"{recall:.4f}",
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "generation", "checkpoint", "conf", "class_id", "class_name", "count"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved detection-growth CSV: %s", output_csv)

    if do_ic:
        ic_fields = ["experiment", "generation", "checkpoint", "conf", "ic_gt",
                     "ic_tp", "ic_fp", "ic_fn", "ic_precision", "ic_recall"]
        with open(ic_output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ic_fields)
            writer.writeheader()
            writer.writerows(ic_rows)
        logger.info("Saved IC quality CSV: %s", ic_output_csv)

    _print_summary(rows, ic_id=ic_id)


def _print_summary(rows: list[dict], ic_id: int) -> None:
    from itertools import groupby

    def _gen_key(r: dict) -> str:
        return r["generation"]

    sorted_rows = sorted(rows, key=_gen_key)
    print("\nDetection-growth summary:")
    print(f"{'Gen':<8}  {'IC':>6}  {'Total':>8}")
    print("-" * 28)
    for gen, group in groupby(sorted_rows, key=_gen_key):
        group_list = list(group)
        ic_count = next((r["count"] for r in group_list if r["class_id"] == ic_id), 0)
        total = sum(r["count"] for r in group_list)
        print(f"{gen:<8}  {ic_count:>6}  {total:>8}")


if __name__ == "__main__":
    main()
