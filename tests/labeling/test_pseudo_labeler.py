import pytest
import numpy as np
from unittest.mock import MagicMock

from src.labeling.pseudo_labeler import PseudoLabeler


def _make_model():
    return MagicMock()


def _make_box(cls: int, conf: float, xywhn=(0.5, 0.5, 0.2, 0.2)):
    box = MagicMock()
    box.cls = np.array(cls)       # 0-d so int(box.cls) works
    box.conf = np.array(conf)     # 0-d so float(box.conf) works
    box.xywhn = [np.array(xywhn)]
    return box


def _make_result(boxes):
    r = MagicMock()
    r.boxes = boxes
    return r


class TestResultsToYolo:
    def setup_method(self):
        self.labeler = PseudoLabeler(
            model=_make_model(),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )

    def test_box_above_threshold_included(self):
        boxes = [_make_box(cls=2, conf=0.8)]
        lines = self.labeler._results_to_yolo([_make_result(boxes)], None)
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "2"
        assert len(parts) == 5

    def test_box_below_threshold_excluded(self):
        boxes = [_make_box(cls=0, conf=0.3)]
        lines = self.labeler._results_to_yolo([_make_result(boxes)], None)
        assert lines == []

    def test_box_exactly_at_threshold_included(self):
        # Filtering is `conf < threshold`, so conf==threshold is kept.
        boxes = [_make_box(cls=0, conf=0.5)]
        lines = self.labeler._results_to_yolo([_make_result(boxes)], None)
        assert len(lines) == 1

    def test_no_boxes_returns_empty(self):
        r = MagicMock()
        r.boxes = None
        lines = self.labeler._results_to_yolo([r], None)
        assert lines == []

    def test_per_class_threshold_filters_correctly(self):
        # cls 0: threshold=0.7 → conf=0.6 should be excluded
        # cls 1: threshold=0.4 → conf=0.6 should be included
        boxes = [_make_box(cls=0, conf=0.6), _make_box(cls=1, conf=0.6)]
        per_class = {0: 0.7, 1: 0.4}
        lines = self.labeler._results_to_yolo([_make_result(boxes)], per_class)
        assert len(lines) == 1
        assert lines[0].startswith("1 ")

    def test_per_class_falls_back_to_global_for_unknown_class(self):
        # cls 5 not in per_class dict → falls back to conf_threshold=0.5
        boxes = [_make_box(cls=5, conf=0.6)]
        per_class = {0: 0.7}
        lines = self.labeler._results_to_yolo([_make_result(boxes)], per_class)
        assert len(lines) == 1
        assert lines[0].startswith("5 ")

    def test_coordinates_written_in_yolo_format(self):
        boxes = [_make_box(cls=3, conf=0.9, xywhn=(0.1, 0.2, 0.3, 0.4))]
        lines = self.labeler._results_to_yolo([_make_result(boxes)], None)
        parts = lines[0].split()
        assert parts[0] == "3"
        assert float(parts[1]) == pytest.approx(0.1, abs=1e-5)
        assert float(parts[2]) == pytest.approx(0.2, abs=1e-5)
        assert float(parts[3]) == pytest.approx(0.3, abs=1e-5)
        assert float(parts[4]) == pytest.approx(0.4, abs=1e-5)

    def test_multiple_results_aggregated(self):
        results = [
            _make_result([_make_box(cls=0, conf=0.9)]),
            _make_result([_make_box(cls=1, conf=0.8)]),
        ]
        lines = self.labeler._results_to_yolo(results, None)
        assert len(lines) == 2
