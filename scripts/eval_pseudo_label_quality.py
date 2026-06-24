"""Pseudo-label quality vs clean GT, per class, per generation, per strategy.

Pure label-file operation (no model, no GPU). For each completed *noisy v3*
self-training run, treats the pseudo-labels of every generation as predictions
and scores them against the clean pre-corruption GT
(``data/train_clean/labels``) at a single operating point:
**IoU >= 0.5 with class match** (greedy one-to-one matching).

The loop is driven by the **clean-GT image list**, not the pseudo files: a clean
image with no matching (or empty) pseudo file contributes all-FN. This avoids
under-counting FN (e.g. iter1 has 3323 pseudo files vs 3324 clean images — that
one image is entirely missed detections).

Generation attribution: ``producer_gen = iter - 1``.

Output: ``results/analysis/pseudo_label_quality.csv``.

Usage (from repo root):
    PYTHONPATH=. uv run python scripts/eval_pseudo_label_quality.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

from src.utils.io import load_config
from src.utils.labels import (
    box_iou,
    greedy_match,
    parse_run_name,
    read_yolo_label,
    yolo_to_xyxy,
)

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
CLEAN_GT_DIR = Path("data/train_clean/labels")
NAMES_CONFIG = Path("configs/yolov8/base_noisy.yaml")
OUTPUT_CSV = Path("results/analysis/pseudo_label_quality.csv")

IOU_THR = 0.5

FIELDS = [
    "run",
    "model",
    "strategy",
    "iter",
    "producer_gen",
    "class_id",
    "class_name",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
]


def load_clean_gt() -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Read all clean GT label files once, keyed by file stem."""
    return {p.stem: read_yolo_label(p) for p in CLEAN_GT_DIR.glob("*.txt")}


def score_iteration(
    clean_gt: dict[str, list],
    pseudo_dir: Path,
) -> dict[int, dict[str, int]]:
    """Score one generation's pseudo-labels against clean GT.

    Returns ``{class_id: {'tp','fp','fn'}}``. Driven by the clean-GT image list,
    so every clean box is counted as exactly one of TP/FN per its class.
    """
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for stem, gt_boxes in clean_gt.items():
        pred_boxes = read_yolo_label(pseudo_dir / f"{stem}.txt")
        gt_cls = [b[0] for b in gt_boxes]
        pred_cls = [b[0] for b in pred_boxes]
        iou = box_iou(yolo_to_xyxy(gt_boxes), yolo_to_xyxy(pred_boxes))
        matches = greedy_match(
            iou, gt_cls, pred_cls, iou_thr=IOU_THR, class_constrained=True
        )
        matched_gt = {i for i, _ in matches}
        matched_pred = {j for _, j in matches}
        # TP: a matched clean box (class agrees by construction).
        for i in matched_gt:
            stats[gt_cls[i]]["tp"] += 1
        # FN: clean box with no class-matching pseudo box.
        for i, c in enumerate(gt_cls):
            if i not in matched_gt:
                stats[c]["fn"] += 1
        # FP: pseudo box that matched no clean box of its class.
        for j, c in enumerate(pred_cls):
            if j not in matched_pred:
                stats[c]["fp"] += 1
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = ap.parse_args()

    cfg = load_config(NAMES_CONFIG)
    names = cfg["data"]["names"]
    name_of = {i: n for i, n in enumerate(names)}

    clean_gt = load_clean_gt()
    clean_per_class = defaultdict(int)
    for boxes in clean_gt.values():
        for c, *_ in boxes:
            clean_per_class[c] += 1
    logger.info("Clean GT: %d images, %d annotations", len(clean_gt), sum(clean_per_class.values()))

    runs = sorted(
        d for d in EXPERIMENTS_DIR.glob("*_st_noisy_*_v3_42") if d.is_dir()
    )
    rows: list[dict] = []
    for run in runs:
        meta = parse_run_name(run.name)
        if meta is None or meta["track"] != "noisy" or meta["version"] != "v3":
            continue
        iter_dirs = sorted(
            run.glob("iter*/pseudo/labels"),
            key=lambda p: int(re.search(r"iter(\d+)", str(p)).group(1)),
        )
        for pseudo_dir in iter_dirs:
            it = int(re.search(r"iter(\d+)", str(pseudo_dir)).group(1))
            stats = score_iteration(clean_gt, pseudo_dir)
            # Exact reconciliation: per class, TP + FN == clean count.
            for c, n_clean in clean_per_class.items():
                tp_fn = stats[c]["tp"] + stats[c]["fn"]
                assert tp_fn == n_clean, (
                    f"{run.name} iter{it} class {c}: TP+FN={tp_fn} != clean {n_clean}"
                )
            for c in sorted(name_of):
                tp = stats[c]["tp"]
                fp = stats[c]["fp"]
                fn = stats[c]["fn"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else ""
                recall = tp / (tp + fn) if (tp + fn) > 0 else ""
                rows.append(
                    {
                        "run": run.name,
                        "model": meta["model"],
                        "strategy": meta["strategy"],
                        "iter": it,
                        "producer_gen": it - 1,
                        "class_id": c,
                        "class_name": name_of[c],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": f"{precision:.4f}" if precision != "" else "",
                        "recall": f"{recall:.4f}" if recall != "" else "",
                    }
                )
            tot_tp = sum(stats[c]["tp"] for c in name_of)
            tot_fp = sum(stats[c]["fp"] for c in name_of)
            tot_fn = sum(stats[c]["fn"] for c in name_of)
            prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0.0
            rec = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0.0
            logger.info(
                "%s iter%d (Gen%d): TP=%d FP=%d FN=%d  P=%.3f R=%.3f",
                run.name, it, it - 1, tot_tp, tot_fp, tot_fn, prec, rec,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
