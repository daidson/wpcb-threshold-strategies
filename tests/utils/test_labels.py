import numpy as np

from src.utils.labels import (
    box_iou,
    greedy_match,
    read_yolo_label,
    yolo_to_xyxy,
)


def test_read_yolo_label_parses_and_tolerates(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n\n1 0.1 0.1 0.05 0.05 0.9\n")
    boxes = read_yolo_label(p)
    assert boxes == [
        (0, 0.5, 0.5, 0.2, 0.2),
        (1, 0.1, 0.1, 0.05, 0.05),  # trailing confidence column ignored
    ]


def test_read_yolo_label_missing_file(tmp_path):
    assert read_yolo_label(tmp_path / "nope.txt") == []


def test_box_iou_known_values():
    # Two identical boxes -> IoU 1.0
    a = yolo_to_xyxy([(0, 0.5, 0.5, 0.2, 0.2)])
    assert box_iou(a, a)[0, 0] == 1.0

    # Half-overlapping equal-size boxes: shift one by half its width along x.
    # box1 x in [0.4,0.6], box2 x in [0.5,0.7]; y identical [0.4,0.6].
    b1 = yolo_to_xyxy([(0, 0.5, 0.5, 0.2, 0.2)])
    b2 = yolo_to_xyxy([(0, 0.6, 0.5, 0.2, 0.2)])
    # intersection = 0.1*0.2 = 0.02; union = 2*0.04 - 0.02 = 0.06; IoU = 1/3
    assert np.isclose(box_iou(b1, b2)[0, 0], 1.0 / 3.0)

    # Disjoint boxes -> 0
    c = yolo_to_xyxy([(0, 0.9, 0.9, 0.05, 0.05)])
    assert box_iou(b1, c)[0, 0] == 0.0


def test_iou_normalization_invariant():
    # IoU is scale-free: multiplying all coords by a constant leaves it unchanged.
    a = yolo_to_xyxy([(0, 0.5, 0.5, 0.2, 0.2)])
    b = yolo_to_xyxy([(0, 0.6, 0.5, 0.2, 0.2)])
    iou_norm = box_iou(a, b)[0, 0]
    iou_scaled = box_iou(a * 640, b * 640)[0, 0]
    assert np.isclose(iou_norm, iou_scaled)


def test_box_iou_empty():
    a = yolo_to_xyxy([(0, 0.5, 0.5, 0.2, 0.2)])
    empty = yolo_to_xyxy([])
    assert box_iou(a, empty).shape == (1, 0)
    assert box_iou(empty, a).shape == (0, 1)


def _build(gt, pred):
    gt_xyxy = yolo_to_xyxy(gt)
    pred_xyxy = yolo_to_xyxy(pred)
    iou = box_iou(gt_xyxy, pred_xyxy)
    return iou, [b[0] for b in gt], [b[0] for b in pred]


def test_greedy_match_three_vs_three_class_constrained():
    # GT: class0@0.2, class1@0.5, class2@0.8
    # Pred: class0@0.2 (TP), class9@0.5 (wrong class -> no match), class2@0.8 (TP)
    gt = [(0, 0.2, 0.5, 0.1, 0.1), (1, 0.5, 0.5, 0.1, 0.1), (2, 0.8, 0.5, 0.1, 0.1)]
    pred = [(0, 0.2, 0.5, 0.1, 0.1), (9, 0.5, 0.5, 0.1, 0.1), (2, 0.8, 0.5, 0.1, 0.1)]
    iou, ca, cb = _build(gt, pred)

    m = greedy_match(iou, ca, cb, iou_thr=0.5, class_constrained=True)
    matched_gt = {i for i, _ in m}
    assert matched_gt == {0, 2}  # class1 GT unmatched (class mismatch)
    tp = len(m)
    fp = len(pred) - tp
    fn = len(gt) - tp
    assert (tp, fp, fn) == (2, 1, 1)


def test_greedy_match_class_agnostic_recovers_swap():
    # Same setup; class-agnostic matching pairs the swapped box by location.
    gt = [(0, 0.2, 0.5, 0.1, 0.1), (1, 0.5, 0.5, 0.1, 0.1), (2, 0.8, 0.5, 0.1, 0.1)]
    pred = [(0, 0.2, 0.5, 0.1, 0.1), (9, 0.5, 0.5, 0.1, 0.1), (2, 0.8, 0.5, 0.1, 0.1)]
    iou, ca, cb = _build(gt, pred)
    m = greedy_match(iou, ca, cb, iou_thr=0.5, class_constrained=False)
    assert len(m) == 3  # all three matched by location


def test_greedy_match_empty_pseudo_all_fn():
    gt = [(0, 0.2, 0.5, 0.1, 0.1), (1, 0.5, 0.5, 0.1, 0.1)]
    iou, ca, cb = _build(gt, [])
    m = greedy_match(iou, ca, cb, iou_thr=0.5, class_constrained=True)
    assert m == []
    assert len(gt) - len(m) == 2  # all FN


def test_greedy_match_each_box_used_once():
    # One GT, two overlapping preds of same class: only the higher-IoU pred matches.
    gt = [(0, 0.5, 0.5, 0.2, 0.2)]
    pred = [(0, 0.55, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.2, 0.2)]  # 2nd is exact match
    iou, ca, cb = _build(gt, pred)
    m = greedy_match(iou, ca, cb, iou_thr=0.5, class_constrained=True)
    assert m == [(0, 1)]  # exact-match pred (index 1, IoU 1.0) wins
