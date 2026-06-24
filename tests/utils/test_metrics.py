import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

from src.utils.metrics import compute_adaptive_thresholds, compare_runs, save_metrics


def _make_metrics(p_curves: list[list[float]], px: list[float] | None = None):
    """Build a minimal mock of an Ultralytics val metrics object (precision curves)."""
    n = len(p_curves[0])
    mock = MagicMock()
    mock.box.p_curve = np.array(p_curves, dtype=float)
    mock.box.px = np.array(px if px is not None else np.linspace(0, 1, n))
    return mock


def test_selects_lowest_conf_reaching_precision_target():
    # Precision first reaches 0.9 at index 2 (conf=0.4); lower confs stay below target.
    px = [0.0, 0.2, 0.4, 0.6, 0.8]
    p = [[0.1, 0.3, 0.9, 0.95, 0.99]]
    metrics = _make_metrics(p, px)
    result = compute_adaptive_thresholds(metrics, nc=1, fallback=0.5, target_precision=0.9)
    assert result[0] == pytest.approx(0.4)


def test_fallback_when_target_never_reached():
    p = [[0.1, 0.4, 0.6]]
    metrics = _make_metrics(p)
    result = compute_adaptive_thresholds(metrics, nc=1, fallback=0.6, target_precision=0.9)
    assert result[0] == pytest.approx(0.6)


def test_multiple_classes_independent():
    px = [0.1, 0.5, 0.9]
    # cls 0 reaches target at index 0 (conf=0.1); cls 1 only at index 2 (conf=0.9)
    p = [[0.92, 0.95, 0.99], [0.1, 0.4, 0.93]]
    metrics = _make_metrics(p, px)
    result = compute_adaptive_thresholds(metrics, nc=2, fallback=0.5, target_precision=0.9)
    assert result[0] == pytest.approx(0.1)
    assert result[1] == pytest.approx(0.9)


def test_fallback_only_for_no_target_class(tmp_path):
    px = [0.2, 0.6]
    # cls 0 reaches target at conf=0.6; cls 1 never reaches it
    p = [[0.3, 0.95], [0.2, 0.5]]
    metrics = _make_metrics(p, px)
    result = compute_adaptive_thresholds(metrics, nc=2, fallback=0.55, target_precision=0.9)
    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(0.55)


def test_compare_runs_sorted_by_map50(tmp_path):
    for name, val in [("run_a", 0.7), ("run_b", 0.9), ("run_c", 0.5)]:
        d = tmp_path / name
        d.mkdir()
        save_metrics({"map50": val}, d / "metrics.json")
    runs = compare_runs(tmp_path)
    assert [r["map50"] for r in runs] == [0.9, 0.7, 0.5]


def test_compare_runs_empty_dir(tmp_path):
    assert compare_runs(tmp_path) == []
