"""Pure label-file helpers for pseudo-label analysis.

These operate entirely on YOLO ``.txt`` label files in *normalized* coordinate
space (``cx cy w h`` in ``[0, 1]``). All matching is always within a single
image, so IoU computed in normalized space is exact — image dimensions are
never read. No model inference, no GPU.

The functions here back the pseudo-label volume/quality/recovery analysis
scripts (``scripts/count_pseudo_labels.py``,
``scripts/eval_pseudo_label_quality.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# (gt_index, pred_index) pairs produced by greedy_match.
Match = tuple[int, int]

# Longest model keys first so rtdetr_l/rtdetr_x match before bare prefixes.
KNOWN_MODELS = ["rtdetr_l", "rtdetr_x", "yolov10", "yolov8", "yolo12"]


def parse_run_name(name: str) -> dict | None:
    """Parse a self-training experiment dir name into its components.

    Handles the full set of names on disk, e.g. ``yolov8_st_noisy_adaptive_v3_42``,
    ``yolov8_st_dslr_adaptive_clean_42``, ``rtdetr_l_st_noisy_static_42``,
    ``yolov10_st_adaptive_42`` (no track/version tokens). Missing tokens default
    to ``track='clean'`` and ``version='v1'``.

    Returns:
        Dict with keys ``run, model, track, strategy, version, seed`` (seed may
        be ``""``), or ``None`` if ``name`` is not a ``<model>_st_...`` run dir.
    """
    model = next((m for m in KNOWN_MODELS if name.startswith(m + "_")), None)
    if model is None:
        return None
    rest = name[len(model) + 1 :]
    if not rest.startswith("st_"):
        return None
    toks = rest[3:].split("_")
    seed = ""
    if toks and toks[-1].isdigit():
        seed = toks[-1]
        toks = toks[:-1]
    version = "v1"
    if toks and re.fullmatch(r"v\d+", toks[-1]):
        version = toks[-1]
        toks = toks[:-1]
    track = "clean"
    if toks and toks[0] in ("noisy", "dslr"):
        track = toks[0]
        toks = toks[1:]
    strategy = "_".join(toks)
    return {
        "run": name,
        "model": model,
        "track": track,
        "strategy": strategy,
        "version": version,
        "seed": seed,
    }


def read_yolo_label(path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file into ``(cls, cx, cy, w, h)`` tuples.

    Tolerant of a missing or empty file (returns ``[]``) and of blank lines.
    Confidence columns, if present, are ignored — only the first five fields of
    each line are read.

    Args:
        path: Path to a YOLO ``.txt`` label file.

    Returns:
        One tuple per annotation; empty list if the file is absent or empty.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out


def yolo_to_xyxy(
    boxes: list[tuple[int, float, float, float, float]],
) -> np.ndarray:
    """Convert ``(cls, cx, cy, w, h)`` tuples to ``(N, 4)`` xyxy corner form.

    Coordinates stay in normalized space. The class column is dropped; pass the
    classes separately to the matcher. Returns an empty ``(0, 4)`` array for an
    empty input so downstream numpy code stays branch-free.
    """
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    arr = np.asarray([b[1:5] for b in boxes], dtype=np.float64)
    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def box_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix between two sets of xyxy boxes.

    Args:
        a_xyxy: ``(M, 4)`` boxes.
        b_xyxy: ``(N, 4)`` boxes.

    Returns:
        ``(M, N)`` IoU matrix; ``(M, 0)`` / ``(0, N)`` when either side is empty.
    """
    m, n = a_xyxy.shape[0], b_xyxy.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)
    ax1, ay1, ax2, ay2 = (a_xyxy[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b_xyxy[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = ((ax2 - ax1) * (ay2 - ay1))
    area_b = ((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(
    iou: np.ndarray,
    classes_a: list[int],
    classes_b: list[int],
    iou_thr: float = 0.5,
    class_constrained: bool = True,
) -> list[Match]:
    """Greedily match boxes A↔B by descending IoU, each box used once.

    Candidate pairs with ``iou >= iou_thr`` are considered in order of
    descending IoU; ties are broken deterministically by ``(a_index, b_index)``
    ascending, so the result does not depend on numpy's sort stability. When
    ``class_constrained`` is True, only pairs with ``classes_a[i] ==
    classes_b[j]`` are eligible (quality pass B); when False, class is ignored
    so a class swap can still match by location (recovery pass C).

    Args:
        iou: ``(M, N)`` IoU matrix (A rows, B cols).
        classes_a: class id per A box.
        classes_b: class id per B box.
        iou_thr: minimum IoU for a pair to be eligible.
        class_constrained: require matching class ids.

    Returns:
        List of ``(a_index, b_index)`` matches, each index appearing at most once.
    """
    m, n = iou.shape
    if m == 0 or n == 0:
        return []
    cand: list[tuple[float, int, int]] = []
    for i in range(m):
        for j in range(n):
            v = iou[i, j]
            if v < iou_thr:
                continue
            if class_constrained and classes_a[i] != classes_b[j]:
                continue
            cand.append((v, i, j))
    # descending IoU; deterministic tie-break on (i, j) ascending
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[Match] = []
    for _v, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j))
    return matches
