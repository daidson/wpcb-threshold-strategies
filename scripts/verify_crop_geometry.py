"""Recover PCB-DSLR crop positions in their full-resolution source images and
test the crop-selection geometry (grid vs overlap vs content-centric on ICs).

Each 512x512 crop in data/PCB_DSLR_CROPS_512 is template-matched (coarse scale)
back into its parent pcbN/recM.jpg from the full CVL PCB-DSLR set, recovering the
crop's top-left pixel position. From the recovered positions we measure:
  * match quality (are crops exact native-resolution windows?),
  * how many of the ~60 possible non-overlapping windows per image are used,
  * pairwise overlap between crops of the same capture,
  * whether crop centres sit on IC annotations (content-centric) vs uniform.

The full images are NOT redistributed: pass --full-dir pointing at a local
extraction of the CVL zips. Run from repo root.

    PYTHONPATH=. uv run python scripts/verify_crop_geometry.py \
        --full-dir "/mnt/f/Mestrado Data/data/PCB_DSLR_FULL/extracted" \
        --out results/analysis/crop_geometry.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CROP_IMG = Path("data/PCB_DSLR_CROPS_512/orig_distribution/images")
CROP_LBL = Path("data/PCB_DSLR_CROPS_512/orig_distribution/labels")
CROP_PX = 512
SCALE = 0.25  # coarse-match downscale; geometry tolerance ~1/SCALE px
STEM_RE = re.compile(r"(pcb\d+)_(rec\d+)_crop_\d+$")


def _gray(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def match_crop(full_gray_small: np.ndarray, crop_path: Path) -> tuple[float, int, int]:
    """Return (score, x, y) of the crop's top-left in full-image pixel coords."""
    crop = np.asarray(Image.open(crop_path).convert("RGB"))
    cs = cv2.resize(_gray(crop), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(full_gray_small, cs, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return float(score), int(round(loc[0] / SCALE)), int(round(loc[1] / SCALE))


def read_ic_annots(annot: Path) -> list[tuple[float, float]]:
    """IC centres (x, y) in full-image pixels from a CVL recM-annot.txt."""
    if not annot.exists():
        return []
    out = []
    for line in annot.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 4:
            out.append((float(p[0]), float(p[1])))  # cx, cy
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", required=True, help="dir with extracted pcbN/recM.jpg + recM-annot.txt")
    ap.add_argument("--out", default="results/analysis/crop_geometry.csv")
    ap.add_argument("--fig", default="results/analysis/crop_geometry_pcb103.png")
    args = ap.parse_args()
    full_dir = Path(args.full_dir)
    boards = sorted(d.name for d in full_dir.glob("pcb*") if d.is_dir())
    logger.info("boards extracted: %d", len(boards))

    rows: list[dict] = []
    cache: dict[str, np.ndarray] = {}
    for crop_path in sorted(CROP_IMG.glob("*.png")):
        m = STEM_RE.match(crop_path.stem)
        if not m:
            continue
        board, rec = m.groups()
        if board not in boards:
            continue
        full_jpg = full_dir / board / f"{rec}.jpg"
        if not full_jpg.exists():
            continue
        key = f"{board}/{rec}"
        if key not in cache:
            full = np.asarray(Image.open(full_jpg).convert("RGB"))
            cache[key] = (
                cv2.resize(_gray(full), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA),
                full.shape[1], full.shape[0],
            )
        small, W, H = cache[key]
        score, x, y = match_crop(small, crop_path)
        rows.append(dict(board=board, rec=rec, crop=crop_path.stem, score=round(score, 3),
                         x=x, y=y, full_w=W, full_h=H))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d crops matched)", out, len(rows))

    # ---------------- analysis ----------------
    sc = np.array([r["score"] for r in rows])
    logger.info("match score: median=%.3f  p10=%.3f  frac>=0.8=%.2f",
                np.median(sc), np.percentile(sc, 10), np.mean(sc >= 0.8))

    by_cap: dict[str, list[dict]] = {}
    for r in rows:
        by_cap.setdefault(f"{r['board']}/{r['rec']}", []).append(r)

    crops_per_cap, windows_per_img, overlap_frac = [], [], []
    for cap, rs in by_cap.items():
        crops_per_cap.append(len(rs))
        W, H = rs[0]["full_w"], rs[0]["full_h"]
        windows_per_img.append((W // CROP_PX) * (H // CROP_PX))
        boxes = [(r["x"], r["y"], r["x"] + CROP_PX, r["y"] + CROP_PX) for r in rs]
        pairs = ov = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                pairs += 1
                ax0, ay0, ax1, ay1 = boxes[i]
                bx0, by0, bx1, by1 = boxes[j]
                iw = max(0, min(ax1, bx1) - max(ax0, bx0))
                ih = max(0, min(ay1, by1) - max(ay0, by0))
                if iw * ih > 0:
                    ov += 1
        if pairs:
            overlap_frac.append(ov / pairs)
    logger.info("crops/capture: mean=%.1f  | possible non-overlap windows/img: mean=%.0f  -> use=%.0f%%",
                np.mean(crops_per_cap), np.mean(windows_per_img),
                100 * np.mean(crops_per_cap) / np.mean(windows_per_img))
    logger.info("fraction of same-capture crop PAIRS that overlap: %.2f", np.mean(overlap_frac))

    # content-centric: distance from crop centre to nearest IC annot, normalised by crop half-size (256px)
    near_ic, has_ic_at_centre = [], []
    for r in rows:
        annot = full_dir / r["board"] / f"{r['rec']}-annot.txt"
        ics = read_ic_annots(annot)
        if not ics:
            continue
        ccx, ccy = r["x"] + CROP_PX / 2, r["y"] + CROP_PX / 2
        d = min(((ccx - ix) ** 2 + (ccy - iy) ** 2) ** 0.5 for ix, iy in ics)
        near_ic.append(d / (CROP_PX / 2))
        # is an IC centre inside the central quarter of the crop?
        has_ic_at_centre.append(any(abs(ccx - ix) <= CROP_PX / 4 and abs(ccy - iy) <= CROP_PX / 4
                                    for ix, iy in ics))
    near_ic = np.array(near_ic)
    logger.info("crop-centre -> nearest IC (in crop-half-widths): median=%.2f  (0=on an IC, content-centric)",
                np.median(near_ic))
    logger.info("crops with an IC centre in their central quarter: %.0f%%  (uniform expectation ~25%%)",
                100 * np.mean(has_ic_at_centre))

    _figure(full_dir, by_cap, args.fig)


def _figure(full_dir: Path, by_cap: dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cap = next((c for c in by_cap if c.startswith("pcb103/")), next(iter(by_cap)))
    board, rec = cap.split("/")
    rs = by_cap[cap]
    full = np.asarray(Image.open(full_dir / board / f"{rec}.jpg").convert("RGB"))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(full)
    for r in rs:
        ax.add_patch(Rectangle((r["x"], r["y"]), CROP_PX, CROP_PX, fill=False,
                               edgecolor="#00d5ff", linewidth=2))
    for ix, iy in read_ic_annots(full_dir / board / f"{rec}-annot.txt"):
        ax.plot(ix, iy, "x", color="#e6194B", markersize=9, markeredgewidth=2)
    ax.set_title(f"{cap}: recovered crop windows (cyan) vs IC annotations (red x)")
    ax.axis("off")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)


if __name__ == "__main__":
    main()
