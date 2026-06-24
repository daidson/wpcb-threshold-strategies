import json
from pathlib import Path


def compute_adaptive_thresholds(
    metrics, nc: int, fallback: float, target_precision: float = 0.9
) -> dict[int, float]:
    """Compute per-class confidence thresholds from the val PR curve (precision target).

    For each class, selects the *lowest* confidence at which the precision curve
    reaches ``target_precision`` — maximising recall (pseudo-label quantity)
    subject to an estimated-precision floor on the calibration set. This matches
    Honorato's *precisão estimada* wording and gives an explicit pseudo-label
    noise knob via ``target_precision``. Falls back to ``fallback`` for classes
    that never reach the target (mirrors the old no-detections fallback).

    Args:
        metrics: Ultralytics val metrics object (has .box.p_curve and .box.px).
        nc: Number of classes.
        fallback: Threshold used when a class never reaches ``target_precision``.
        target_precision: Per-class precision floor on the calibration set.

    Returns:
        Dict mapping class index to confidence threshold.
    """
    p_curve = metrics.box.p_curve  # (nc, 1000) precision at each confidence
    px = metrics.box.px            # (1000,) confidence axis [0, 1]

    thresholds: dict[int, float] = {}
    for cls_idx in range(nc):
        meets = p_curve[cls_idx] >= target_precision
        if not bool(meets.any()):
            thresholds[cls_idx] = fallback
        else:
            # argmax on a boolean array returns the index of the first True,
            # i.e. the lowest confidence meeting the precision target.
            thresholds[cls_idx] = float(px[int(meets.argmax())])
    return thresholds


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_runs(results_dir: str | Path) -> list[dict]:
    """Collect all metrics.json files under results_dir and return sorted by mAP50."""
    results_dir = Path(results_dir)
    runs = []
    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        data = load_metrics(metrics_file)
        data["run"] = str(metrics_file.parent)
        runs.append(data)
    return sorted(runs, key=lambda x: x.get("map50", 0), reverse=True)
