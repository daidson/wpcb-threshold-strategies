"""WPCB-EFAv2+ — multi-class economic-feasibility composition (MVP tier).

Honorato's WPCB-EFAv2 (Silva et al. 2025, §V) reports, per component type,
quantity / total area / % of board area, but its quantitative recovery chain
covers only ICs and electrolytic capacitors; every other class is merely
categorised by recycling value and environmental risk. This script reproduces
the *composition* layer for all eight detector classes and joins the recycling-
priority categorisation, on real WPCB images (PCB-DSLR).

Key design fact (verified from the detection-growth CSVs): on the PCB-DSLR
target domain only the **Generation-0 teacher** detects all eight classes; the
self-trained students specialise to IC and the other seven classes collapse to
zero from iter1 onward (`pseudo_only_mode` with IC-only GT merged). The
multi-class assessor is therefore the Gen0 teacher — pass its ``best.pt`` via
``--weights``. The script is checkpoint-agnostic so this stays an explicit,
inspectable choice rather than a hidden default.

This is the MVP tier: composition (count, area-density, relative class share) +
qualitative priority. No physical scale, no grams, no dollars — the absolute
mass/profit tier is gated on data the repo does not have
(docs/wpcb_efa_feasibility.md).

Crops, not tiles: PCB-DSLR ``pcbNNN_recM_crop_KK`` files are **content-centric**
512x512 crops, not a disjoint grid partition of the board. Inspecting the images
shows they include off-board background and overlap each other (adjacent crops
share components), and the crop count varies across recordings of the same board.
Two consequences the metrics must respect: (1) crop area is **not** board area, so
no quantity here is a "% of board area"; (2) summed counts across overlapping
crops would double-count, so a per-board count is a relative composition, not an
exhaustive inventory.

Board de-duplication: a single physical board (``pcbNNN``) may be photographed in
several recordings (``recM``). ``--rec-policy`` controls the per-board reduction:

  * ``average`` (default) — treat each recording as one observation of the board
    and take the mean over recordings, so a board with more recordings does not
    dominate. Counts become fractional, which is correct for a *mean per board*.
  * ``representative`` — keep one recording (most crops; ties broken by lowest
    recording id) and discard the rest; integer counts.

``area_density`` for a class is the mean over a recording's crops of the summed
normalised box area (``w * h``) of that class — i.e. the fraction of a crop the
class occupies, averaged over crops. It is a relative size proxy (an IC occupies
more pixels than a resistor), **not** a share of board area.

Outputs (``--output-dir``, default ``results/efa/``):
  * ``efa_per_capture.csv``    — per (board, recording, class): count + area
  * ``efa_per_board.csv``      — per (board, class), rec-reduced, with
                                  area-density + priority categorisation
  * ``efa_composition.csv``    — dataset-level per-class composition + priority

Usage:
    PYTHONPATH=. uv run python scripts/wpcb_efa.py \\
        --model yolov8 \\
        --config configs/yolov8/self_train_dslr.yaml \\
        --weights runs/yolov8/baseline_noisy_v3/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from src.models import MODEL_REGISTRY, build_model
from src.utils.io import IMAGE_EXTS, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# pcbNNN_recM_crop_KK  ->  board="pcbNNN", rec="recM"
_TILE_RE = re.compile(r"^(pcb\d+)_(rec\d+)_")


def parse_tile_stem(stem: str) -> tuple[str, str] | None:
    """Return ``(board_id, rec_id)`` for a PCB-DSLR tile stem, or ``None``.

    ``None`` signals a filename that does not follow the ``pcbNNN_recM_crop_KK``
    convention so the caller can warn instead of silently mis-grouping.
    """
    m = _TILE_RE.match(stem)
    if m is None:
        return None
    return m.group(1), m.group(2)


def run_inference(
    model, image_paths: list[Path], conf: float, nms_iou: float
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str, int], dict[str, float]]]:
    """Run inference over the tiles, aggregating to (board, rec) granularity.

    Returns ``(tile_counts, agg)`` where ``tile_counts`` maps ``(board, rec)``
    to its number of tiles, and ``agg`` maps ``(board, rec, class_id)`` to
    ``{'count', 'area'}`` with ``area`` the summed normalized box area
    (``w * h`` in ``[0, 1]``, i.e. fraction of one tile).
    """
    tile_counts: dict[tuple[str, str], int] = defaultdict(int)
    agg: dict[tuple[str, str, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "area": 0.0}
    )
    skipped = 0
    for img_path in tqdm(image_paths, desc="inference", leave=False):
        parsed = parse_tile_stem(img_path.stem)
        if parsed is None:
            skipped += 1
            continue
        board, rec = parsed
        tile_counts[(board, rec)] += 1
        results = model.predict(str(img_path), conf=conf, iou=nms_iou)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                c = int(box.cls)
                _, _, w, h = box.xywhn[0].tolist()
                cell = agg[(board, rec, c)]
                cell["count"] += 1
                cell["area"] += w * h
    if skipped:
        logger.warning("Skipped %d tiles not matching pcbNNN_recM_crop_KK", skipped)
    return dict(tile_counts), dict(agg)


def recs_by_board(tile_counts: dict[tuple[str, str], int]) -> dict[str, list[str]]:
    """Group recording ids under each board, sorted by recording number."""
    by_board: dict[str, list[str]] = defaultdict(list)
    for (board, rec) in tile_counts:
        by_board[board].append(rec)
    for board in by_board:
        by_board[board].sort(key=_rec_num)
    return dict(by_board)


def reduce_to_board(
    board: str,
    recs: list[str],
    tile_counts: dict[tuple[str, str], int],
    agg: dict[tuple[str, str, int], dict[str, float]],
    n_classes: int,
    policy: str,
) -> tuple[list[str], list[dict[str, float]]]:
    """Reduce a board's recordings to one per-class composition vector.

    Returns ``(used_recs, per_class)`` where ``per_class[cid]`` has
    ``{'count', 'area', 'pct'}``. Under ``average`` every recording (including
    those with zero detections of a class) contributes to the mean, so a sparse
    recording correctly pulls the estimate down rather than being dropped.
    """
    if policy == "representative":
        rep = max(recs, key=lambda r: (tile_counts[(board, r)], -_rec_num(r)))
        recs = [rep]
    per_class: list[dict[str, float]] = []
    for cid in range(n_classes):
        count = area = pct = 0.0
        for rec in recs:
            cell = agg.get((board, rec, cid))
            n_tiles = tile_counts[(board, rec)]
            c = cell["count"] if cell else 0.0
            a = cell["area"] if cell else 0.0
            count += c
            area += a
            pct += a / n_tiles if n_tiles else 0.0
        k = len(recs)
        per_class.append({"count": count / k, "area": area / k, "pct": pct / k})
    return recs, per_class


def _rec_num(rec: str) -> int:
    m = re.search(r"\d+", rec)
    return int(m.group()) if m else 0


def load_priority(path: Path) -> dict[str, dict]:
    """Load the recycling-priority categorisation keyed by class name."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["classes"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WPCB-EFAv2+ multi-class composition (MVP)")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", required=True,
                   help="YAML config (data.names + pseudo_label.nms_iou are read)")
    p.add_argument("--weights", required=True,
                   help="Detector checkpoint. For multi-class composition this is the "
                        "Gen0 teacher best.pt (students collapse to IC on PCB-DSLR).")
    p.add_argument("--images", default="data/PCB_DSLR_CROPS_512/orig_distribution/images",
                   help="PCB-DSLR tile directory")
    p.add_argument("--rec-policy", choices=["average", "representative"], default="average",
                   help="Per-board reduction over recordings (default: average)")
    p.add_argument("--priority-config", default="configs/wpcb_efa_priority.yaml",
                   help="Recycling-priority categorisation table")
    p.add_argument("--conf", type=float, default=None,
                   help="Inference conf (default: pseudo_label.conf_threshold from config)")
    p.add_argument("--output-dir", default="results/efa",
                   help="Directory for the three EFA CSVs")
    p.add_argument("--detection-growth-csv", default=None,
                   help="Optional detection_growth.csv to cross-check the Gen0 IC total "
                        "(same weights/conf/images must yield the same IC count)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of tiles (smoke test only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    nms_iou: float = cfg["pseudo_label"]["nms_iou"]
    conf: float = args.conf if args.conf is not None else cfg["pseudo_label"]["conf_threshold"]
    class_names: list[str] = cfg["data"]["names"]
    priority = load_priority(Path(args.priority_config))

    images_dir = Path(args.images)
    weights = Path(args.weights)
    out_dir = Path(args.output_dir)
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        sys.exit(1)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    logger.info("EFA over %d tiles | conf=%.2f NMS=%.2f | weights=%s",
                len(image_paths), conf, nms_iou, weights)

    model = build_model(args.model, cfg)
    model.load_weights(str(weights))
    tile_counts, agg = run_inference(model, image_paths, conf, nms_iou)

    boards = recs_by_board(tile_counts)
    n_boards = len(boards)
    logger.info("Boards: %d | captures: %d | tiles: %d | rec-policy=%s",
                n_boards, len(tile_counts), len(image_paths), args.rec_policy)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_per_capture(out_dir / "efa_per_capture.csv", tile_counts, agg, class_names)
    per_board = _write_per_board(
        out_dir / "efa_per_board.csv", boards, tile_counts, agg,
        class_names, priority, args.rec_policy
    )
    _write_composition(
        out_dir / "efa_composition.csv", per_board, n_boards, class_names, priority
    )

    _ic_cross_check(agg, class_names, conf, args.detection_growth_csv)
    _print_summary(per_board, n_boards, class_names, priority)


def _write_per_capture(path, tile_counts, agg, class_names) -> None:
    fields = ["board", "rec", "n_tiles", "class_id", "class_name", "count", "sum_area_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (board, rec) in sorted(tile_counts):
            for cid, name in enumerate(class_names):
                cell = agg.get((board, rec, cid))
                w.writerow({
                    "board": board, "rec": rec, "n_tiles": tile_counts[(board, rec)],
                    "class_id": cid, "class_name": name,
                    "count": int(cell["count"]) if cell else 0,
                    "sum_area_frac": f"{cell['area']:.6f}" if cell else "0.000000",
                })
    logger.info("Saved %s", path)


def _write_per_board(path, boards, tile_counts, agg, class_names, priority, policy) -> list[dict]:
    """Write per-board rows (reduced over recordings); return them for aggregation."""
    fields = ["board", "n_recs", "used_recs", "class_id", "class_name", "count",
              "sum_area_frac", "area_density", "recycling_value", "env_risk", "key_metals"]
    rows: list[dict] = []
    for board in sorted(boards):
        recs = boards[board]
        used, per_class = reduce_to_board(
            board, recs, tile_counts, agg, len(class_names), policy
        )
        for cid, name in enumerate(class_names):
            pc = per_class[cid]
            meta = priority.get(name, {})
            rows.append({
                "board": board, "n_recs": len(recs), "used_recs": "/".join(used),
                "class_id": cid, "class_name": name,
                "count": pc["count"], "sum_area_frac": pc["area"],
                "area_density": pc["pct"],
                "recycling_value": meta.get("recycling_value", ""),
                "env_risk": meta.get("env_risk", ""),
                "key_metals": "/".join(meta.get("key_metals", [])),
            })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "count": f"{r['count']:.4f}",
                        "sum_area_frac": f"{r['sum_area_frac']:.6f}",
                        "area_density": f"{r['area_density']:.6f}"})
    logger.info("Saved %s", path)
    return rows


def _write_composition(path, per_board, n_boards, class_names, priority) -> None:
    """Dataset-level per-class composition, ranked by recycling priority."""
    totals: dict[str, dict[str, float]] = {
        n: {"count": 0.0, "pct_sum": 0.0} for n in class_names
    }
    for r in per_board:
        t = totals[r["class_name"]]
        t["count"] += r["count"]
        t["pct_sum"] += r["area_density"]

    fields = ["class_id", "class_name", "total_count", "mean_count_per_board",
              "mean_area_density", "recycling_value", "env_risk", "key_metals",
              "value_rank", "risk_rank", "priority_rank"]
    rows = []
    for cid, name in enumerate(class_names):
        meta = priority.get(name, {})
        rows.append({
            "class_id": cid, "class_name": name,
            # board-deduplicated total = sum of per-board (rec-reduced) means;
            # rounded so a small fractional sum doesn't read as 0 while the mean is > 0
            "total_count": round(totals[name]["count"]),
            "mean_count_per_board": totals[name]["count"] / n_boards if n_boards else 0.0,
            "mean_area_density": totals[name]["pct_sum"] / n_boards if n_boards else 0.0,
            "recycling_value": meta.get("recycling_value", ""),
            "env_risk": meta.get("env_risk", ""),
            "key_metals": "/".join(meta.get("key_metals", [])),
            "value_rank": meta.get("value_rank", 99),
            "risk_rank": meta.get("risk_rank", 99),
        })
    # priority: recycling value first (value_rank asc), then env risk (risk_rank asc)
    order = sorted(rows, key=lambda r: (r["value_rank"], r["risk_rank"]))
    for rank, r in enumerate(order, start=1):
        r["priority_rank"] = rank
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "mean_count_per_board": f"{r['mean_count_per_board']:.4f}",
                        "mean_area_density": f"{r['mean_area_density']:.6f}"})
    logger.info("Saved %s", path)


def _ic_cross_check(agg, class_names, conf, dg_csv) -> None:
    """Cross-check the all-tile IC total against a detection_growth.csv Gen0 row."""
    if "ic" not in class_names:
        return
    ic_id = class_names.index("ic")
    ic_total = int(sum(c["count"] for (_, _, cid), c in agg.items() if cid == ic_id))
    logger.info("IC cross-check: EFA all-tile IC detections = %d", ic_total)
    if dg_csv is None:
        return
    p = Path(dg_csv)
    if not p.exists():
        logger.warning("detection_growth.csv not found for cross-check: %s", p)
        return
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["generation"] == "gen0" and row["class_name"] == "ic":
                dg_ic = int(row["count"])
                status = "MATCH" if dg_ic == ic_total else "MISMATCH"
                logger.info("IC cross-check vs %s gen0: %d (%s)", p.name, dg_ic, status)
                if status == "MISMATCH":
                    logger.warning("IC totals differ — check weights/conf/images alignment")
                return
    logger.warning("No gen0/ic row found in %s", p)


def _print_summary(per_board, n_boards, class_names, priority) -> None:
    counts = {n: 0 for n in class_names}
    pct = {n: 0.0 for n in class_names}
    for r in per_board:
        counts[r["class_name"]] += r["count"]
        pct[r["class_name"]] += r["area_density"]
    print(f"\nWPCB-EFAv2+ composition over {n_boards} boards (rec-reduced):")
    print(f"{'class':<24} {'total':>8} {'/board':>8} {'a-dens%':>8}  {'value':>6}/{'risk':<6}")
    print("-" * 70)
    order = sorted(class_names,
                   key=lambda n: priority.get(n, {}).get("value_rank", 99))
    for n in order:
        m = priority.get(n, {})
        per = counts[n] / n_boards if n_boards else 0.0
        pa = 100 * pct[n] / n_boards if n_boards else 0.0
        print(f"{n:<24} {counts[n]:>8.1f} {per:>8.2f} {pa:>7.2f}%  "
              f"{m.get('recycling_value', '?'):>6}/{m.get('env_risk', '?'):<6}")


if __name__ == "__main__":
    main()
