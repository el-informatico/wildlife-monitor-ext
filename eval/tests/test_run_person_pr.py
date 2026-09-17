"""Offline tests for the person/vehicle P/R runner (eval/run_person_pr.py).

Pure-logic coverage of the two accumulators and the class-restricted
matching glue: image-level flagging (Seattle tier), box-level greedy
matching against COCO GT (COCO tier), and the definition-level traps —
cross-class detections must never match, per-image denominators must
keep the other class's positives, and thresholds must be ≥, not >.
Everything runs on hand-computed synthetic fixtures: no cv2, no images,
no datasets. The detector path itself is exercised by the runner's
``--max-per-set`` smoke mode, not here.
"""

from __future__ import annotations

import pytest

from eval.run_person_pr import (
    PERSON_CLS, VEHICLE_CLS, box_level_pr, flag_image, image_level_pr,
)


def _det(cls, conf, bbox=(0, 0, 10, 10)):
    return {"bbox": list(bbox), "conf": conf, "cls": cls}


# ---------------------------------------------------------------------------
# flag_image — threshold semantics
# ---------------------------------------------------------------------------

def test_flag_image_threshold_is_inclusive():
    dets = [_det(PERSON_CLS, 0.25)]
    assert flag_image(dets, PERSON_CLS, 0.25) is True   # >=, not >
    assert flag_image(dets, PERSON_CLS, 0.2501) is False


def test_flag_image_class_restricted():
    dets = [_det(VEHICLE_CLS, 0.9)]
    assert flag_image(dets, PERSON_CLS, 0.10) is False  # vehicle ≠ person
    assert flag_image(dets, VEHICLE_CLS, 0.10) is True


# ---------------------------------------------------------------------------
# image_level_pr — Seattle tier definitions
# ---------------------------------------------------------------------------

def test_image_level_pr_hand_computed():
    rows = [
        # TP at every threshold: dan image, strong person det
        {"gt": True, "stratum": "dan", "dets": [_det(PERSON_CLS, 0.8)]},
        # TP only below 0.5: dan image, weak det (FN at 0.5)
        {"gt": True, "stratum": "dan", "dets": [_det(PERSON_CLS, 0.3)]},
        # FN at every threshold: dan image, no person det
        {"gt": True, "stratum": "dan", "dets": [_det(VEHICLE_CLS, 0.9)]},
        # FP at 0.10 only: empty image, floor-level person det
        {"gt": False, "stratum": "empty", "dets": [_det(PERSON_CLS, 0.12)]},
        # clean negative
        {"gt": False, "stratum": "animal", "dets": [_det(0, 0.9)]},
    ]
    out = image_level_pr(rows, PERSON_CLS)
    lo = out["0.10"]
    assert (lo["tp"], lo["fp"], lo["fn"]) == (2, 1, 1)
    assert lo["precision"] == round(2 / 3, 4)
    assert lo["recall"] == round(2 / 3, 4)
    assert lo["fp_by_stratum"] == {"empty": 1}
    hi = out["0.50"]
    assert (hi["tp"], hi["fp"], hi["fn"]) == (1, 0, 2)
    assert hi["precision"] == 1.0
    assert hi["recall"] == round(1 / 3, 4)


def test_image_level_pr_keeps_other_class_positives_in_denominator():
    """A person det on a vehicle-only image is an honest person FP."""
    rows = [{"gt": False, "stratum": "vehicle",
             "dets": [_det(PERSON_CLS, 0.9)]}]
    out = image_level_pr(rows, PERSON_CLS)["0.10"]
    assert (out["tp"], out["fp"], out["fn"]) == (0, 1, 0)
    assert out["precision"] == 0.0


def test_image_level_pr_no_predictions_gives_null_precision():
    rows = [{"gt": False, "stratum": "empty", "dets": []}]
    out = image_level_pr(rows, PERSON_CLS)["0.10"]
    assert out["precision"] is None   # 0/0 — undefined, not 0
    assert out["recall"] is None


# ---------------------------------------------------------------------------
# box_level_pr — COCO tier, class-restricted greedy matching
# ---------------------------------------------------------------------------

def test_box_level_pr_hand_computed_with_iou50():
    # GT is COCO-native xywh: (100,100,10,10) → xyxy (100,100,110,110)
    gt_xywh = [(0, 0, 10, 10), (100, 100, 10, 10)]
    rows = [{
        "gt": [list(b) for b in gt_xywh],
        "name": "person_gt",
        "dets": [
            _det(PERSON_CLS, 0.9, (0, 0, 10, 10)),          # matches box 0
            _det(PERSON_CLS, 0.4, (200, 200, 210, 210)),    # no GT there: FP
        ],
    }]
    out = box_level_pr(rows, PERSON_CLS)
    lo = out["0.10"]
    assert (lo["tp"], lo["fp"], lo["fn"]) == (1, 1, 1)
    assert lo["n_gt_boxes"] == 2
    hi = out["0.50"]                       # 0.4-det dropped by threshold
    assert (hi["tp"], hi["fp"], hi["fn"]) == (1, 0, 1)


def test_box_level_pr_wrong_class_never_matches():
    """A vehicle det on a person-GT image at IoU 1.0 is a person FN —
    class-restricted pre-filtering means it can never be a TP."""
    rows = [{
        "gt": [[0, 0, 10, 10]],
        "name": "person_gt",
        "dets": [_det(VEHICLE_CLS, 0.9, (0, 0, 10, 10))],
    }]
    out = box_level_pr(rows, PERSON_CLS)["0.10"]
    assert (out["tp"], out["fp"], out["fn"]) == (0, 0, 1)


def test_box_level_pr_high_score_greedy_wins():
    """Greedy by score: the 0.9 det consumes the GT first; the 0.8 det
    overlapping the same GT has nothing left to match → FP."""
    rows = [{
        "gt": [[0, 0, 10, 10]],
        "name": "person_gt",
        "dets": [
            _det(PERSON_CLS, 0.8, (0, 0, 9, 10)),   # IoU 0.9 vs GT
            _det(PERSON_CLS, 0.9, (0, 0, 10, 10)),  # IoU 1.0 vs GT
        ],
    }]
    out = box_level_pr(rows, PERSON_CLS)["0.10"]
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)


def test_box_level_pr_fp_attribution_by_gt_group():
    """FPs are attributed per threshold by the image's GT group: on-class
    images are localization errors, off-class images are confusions."""
    rows = [{
        "gt": [[0, 0, 10, 10]],
        "name": "person_gt",
        "dets": [_det(PERSON_CLS, 0.4, (50, 50, 60, 60))],  # wrong place
    }]
    lo = box_level_pr(rows, PERSON_CLS)["0.10"]
    assert lo["fp_on"] == {"person_gt": 1}          # localization error
    hi = box_level_pr(rows, PERSON_CLS)["0.50"]     # det below threshold
    assert hi["fp"] == 0 and hi["fp_on"] == {}
