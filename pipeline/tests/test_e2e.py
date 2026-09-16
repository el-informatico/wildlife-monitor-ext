"""Offline tests for the E2E union glue (no cv2, models, or datasets).

Heavy imports (cv2 for extract_crop, wildlife_monitor for the matcher's
IoU) are either avoided by construction or skipped when the sprints-1-2
environment is absent. The training-recipe test copies the reference
arithmetic from scripts/train_classifier_v0.py verbatim, so any drift
between the crop recipe used here and the one the classifier was
trained on fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import e2e


# ---------------------------------------------------------------------------
# Crop windows
# ---------------------------------------------------------------------------


def _reference_training_window(xc, yc, bw, bh, w, h, margin=0.10):
    """Verbatim arithmetic of scripts/train_classifier_v0.py (MARGIN=0.10)."""
    x1 = (xc * w - bw * w / 2) - margin * bw * w
    y1 = (yc * h - bh * h / 2) - margin * bh * h
    x2 = (xc * w + bw * w / 2) + margin * bw * w
    y2 = (yc * h + bh * h / 2) + margin * bh * h
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    return x1, y1, x2, y2


def test_crop_window_yolo_matches_training_recipe_exactly():
    """GT-crop windows must be byte-identical to the trainer's arithmetic."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        w, h = int(rng.integers(200, 1400)), int(rng.integers(150, 1000))
        xc, yc = rng.uniform(0.05, 0.95, size=2)
        bw, bh = rng.uniform(0.03, 0.9, size=2)
        got = e2e.crop_window_yolo(xc, yc, bw, bh, w, h)
        want = _reference_training_window(xc, yc, bw, bh, w, h)
        small = (want[2] - want[0] < 8) or (want[3] - want[1] < 8)
        if small:
            assert got is None  # trainer skips these crops entirely
        else:
            assert got == want


def test_crop_window_margin_is_ten_percent_per_side():
    # 100x100 box centered in a 400x400 frame -> +10 px per side.
    win = e2e.crop_window(150, 150, 250, 250, 400, 400)
    assert win == (140, 140, 260, 260)


def test_crop_window_clamps_to_frame():
    win = e2e.crop_window(0, 0, 60, 40, 100, 80)
    assert win == (0, 0, 66, 44)          # rounds outward, clamped at 0
    win = e2e.crop_window(90, 70, 100, 80, 100, 80)
    assert win == (89, 69, 100, 80)       # clamped at the far edges


def test_crop_window_drops_tiny_boxes():
    # a 3x3 box expands to well under 8 px and must be dropped;
    # a 20x20 box expands to 22x22 (+10% per side) and survives.
    assert e2e.crop_window(10, 10, 13, 13, 30, 30) is None
    assert e2e.crop_window(0, 0, 20, 20, 30, 30) == (0, 0, 22, 22)


def test_crop_window_xywh_matches_xyxy_semantics():
    # xywh path (absolute GT form) agrees with the xyxy path.
    assert e2e.crop_window_xywh([100, 50, 40, 60], 400, 300) == \
        e2e.crop_window(100, 50, 140, 110, 400, 300)


def test_extract_crop_resizes():
    cv2 = pytest.importorskip("cv2")
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[40:60, 40:60] = 255
    crop = e2e.extract_crop(bgr, (40, 40, 60, 60), size=224)
    assert crop.shape == (224, 224, 3)
    assert crop[0, 0, 0] == 255  # content preserved through the resize


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------


def _fake_sample(n_per_stratum=40):
    images = []
    for stratum in ("aw_with_gt", "mo_with_gt", "mo_empty"):
        for i in range(n_per_stratum):
            images.append({
                "image_id": f"{stratum}-{i:03d}", "stratum": stratum,
                "source_dataset": "x", "rel_path": "x", "width": 10, "height": 10,
                "gt_boxes": [],
            })
    return images


def test_draw_sample_deterministic_and_sorted():
    images = _fake_sample()
    a = e2e.draw_sample(images, seed=42, sizes={"aw_with_gt": 5, "mo_with_gt": 3, "mo_empty": 2})
    b = e2e.draw_sample(images, seed=42, sizes={"aw_with_gt": 5, "mo_with_gt": 3, "mo_empty": 2})
    assert [e["image_id"] for e in a] == [e["image_id"] for e in b]
    ids = [e["image_id"] for e in a]
    assert ids == sorted(ids)                      # output order is stable
    assert len(a) == 10
    assert sum(1 for e in a if e["stratum"] == "aw_with_gt") == 5


def test_draw_sample_seed_changes_selection():
    images = _fake_sample()
    sizes = {"aw_with_gt": 5, "mo_with_gt": 3, "mo_empty": 2}
    a = e2e.draw_sample(images, seed=42, sizes=sizes)
    c = e2e.draw_sample(images, seed=43, sizes=sizes)
    assert [e["image_id"] for e in a] != [e["image_id"] for e in c]


def test_draw_sample_oversized_stratum_takes_everything():
    images = _fake_sample(n_per_stratum=3)
    out = e2e.draw_sample(images, seed=42, sizes={"aw_with_gt": 10, "mo_with_gt": 1, "mo_empty": 0})
    counts = {}
    for e in out:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    assert counts == {"aw_with_gt": 3, "mo_with_gt": 1}


def test_parse_sizes():
    assert e2e.parse_sizes("aw_with_gt=100, mo_with_gt=75,mo_empty=25") == {
        "aw_with_gt": 100, "mo_with_gt": 75, "mo_empty": 25}
    with pytest.raises(ValueError):
        e2e.parse_sizes("   ")


# ---------------------------------------------------------------------------
# Record assembly against the infra ingest contract
# ---------------------------------------------------------------------------


def _fake_image_meta(**over):
    meta = {
        "image_id": "aw-00007", "source_dataset": "african_wildlife",
        "width": 550, "height": 412, "is_night": False,
    }
    meta.update(over)
    return meta


def _animal_det():
    return {"bbox": [132.99, 96.0, 417.0, 314.0], "conf": 0.5689, "cls": 0}


def test_batch_validates_against_infra_schema():
    records = [
        e2e.detection_record(_animal_det(), 550, 412, species="zebra"),
        e2e.detection_record({"bbox": [10, 10, 40, 60], "conf": 0.9, "cls": 1}, 550, 412),
    ]
    batch = e2e.build_batch(_fake_image_meta(), records, "raw/cam-01/aw-00007.jpg",
                            camera_id="cam-01", runtime="cv2.dnn (OpenCV 5.0.0)")
    n = e2e.validate_batch(batch)
    assert n == 2
    schema = e2e.import_schema()
    det = schema.validate_detection(batch["detections"][0])
    assert det.species == "zebra"
    assert det.label == "animal"
    # normalized bbox = pixel bbox / frame size, within tolerance
    assert det.bbox == pytest.approx([132.99 / 550, 96.0 / 412, 417.0 / 550, 314.0 / 412], abs=1e-3)
    assert batch["detections"][1].get("species") is None  # person never labelled


def test_animal_without_species_is_valid():
    """Small animal crops can be unclassifiable; species must be optional."""
    rec = e2e.detection_record(_animal_det(), 550, 412, species=None)
    batch = e2e.build_batch(_fake_image_meta(), [rec], "raw/cam/aw-00007.jpg")
    assert e2e.validate_batch(batch) == 1
    assert "species" not in batch["detections"][0]


def test_schema_rejects_species_on_non_animal():
    """The contract itself must guard the producer against mislabelling."""
    schema = e2e.import_schema()
    with pytest.raises(schema.SchemaError):
        schema.validate_detection({"label": "person", "species": "zebra",
                                   "confidence": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2]})


def test_detection_record_confidence_and_bbox_bounds():
    det = {"bbox": [0.0, 0.0, 550.0, 412.0], "conf": 0.31, "cls": 0}
    rec = e2e.detection_record(det, 550, 412, species="buffalo")
    assert 0.0 <= rec["confidence"] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in rec["bbox"])
    assert rec["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_batch_source_fields():
    batch = e2e.build_batch(_fake_image_meta(), [], "raw/c1/i.jpg", camera_id="c1")
    schema = e2e.import_schema()
    parsed = schema.validate_batch(batch)
    assert parsed.schema_version == schema.SCHEMA_VERSION == 1
    assert parsed.source["dataset"] == "african_wildlife"
    assert parsed.source["camera_id"] == "c1"
    assert parsed.source["is_night"] is False
    assert parsed.detections == ()


# ---------------------------------------------------------------------------
# Detection ↔ GT matching
# ---------------------------------------------------------------------------


def test_match_dets_to_gt_greedy_and_one_to_one():
    pytest.importorskip("wildlife_monitor")
    gt = [[0, 0, 10, 10], [20, 0, 10, 10]]           # two adjacent GT boxes
    dets = [[0, 0, 10, 10], [1, 1, 11, 11], [21, 0, 31, 10]]
    scores = [0.9, 0.8, 0.7]
    mapping = e2e.match_dets_to_gt(gt, dets, scores, iou_threshold=0.5)
    # best det takes GT0; the shifted box still overlaps GT0 heavily -> GT0
    # is consumed, so it goes unmatched; det3 takes GT1.
    assert mapping[0] == 0
    assert mapping[1] == -1
    assert mapping[2] == 1


def test_match_dets_to_gt_agrees_with_parity_counts():
    pytest.importorskip("wildlife_monitor")
    from wildlife_monitor.detect import parity
    rng = np.random.default_rng(11)
    for _ in range(50):
        n_gt = int(rng.integers(0, 5))
        n_det = int(rng.integers(0, 6))
        gt = rng.uniform(0, 80, size=(n_gt, 4)).tolist()   # xywh
        det_boxes = []
        for x, y in rng.uniform(0, 100, size=(n_det, 2)):
            w, h = rng.uniform(5, 50, size=2)
            det_boxes.append([float(x), float(y), float(x + w), float(y + h)])
        scores = rng.uniform(0.1, 1.0, size=n_det).tolist()
        mapping = e2e.match_dets_to_gt(gt, det_boxes, scores, iou_threshold=0.5)
        counts = parity.match_image(gt, det_boxes, scores, iou_threshold=0.5)
        assert sum(1 for m in mapping if m >= 0) == counts["tp"]
        assert sum(1 for m in mapping if m < 0) == counts["fp"]
        assert n_gt - sum(1 for m in mapping if m >= 0) == counts["fn"]


# ---------------------------------------------------------------------------
# AW species labels
# ---------------------------------------------------------------------------


def test_aw_species_boxes_keeps_class_and_order(tmp_path: Path):
    label = tmp_path / "1.txt"
    label.write_text("0 0.5 0.5 0.4 0.6\n3 0.2 0.2 0.1 0.1\n")
    classes = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
    boxes = e2e.aw_species_boxes(label, 100, 200, classes)
    assert [b["species"] for b in boxes] == ["buffalo", "zebra"]
    assert boxes[0]["bbox"] == pytest.approx([30.0, 40.0, 40.0, 120.0])
    assert boxes[0]["yolo"] == (0.5, 0.5, 0.4, 0.6)
    assert e2e.aw_species_boxes(tmp_path / "missing.txt", 100, 200, classes) == []


def test_aw_label_path_layout():
    p = e2e.aw_label_path(Path("data"), "african-wildlife/images/train/p (1).jpg")
    assert p == Path("data/african-wildlife/labels/train/p (1).txt")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def test_confidence_histogram_buckets():
    hist = e2e.confidence_histogram([0.25, 0.3, 0.95, 1.0, 0.55], 0.25, 1.0, step=0.25)
    assert [b["n"] for b in hist] == [2, 1, 2]     # 1.0 lands in the last bucket
    assert hist[0]["bucket"] == "[0.25,0.50]"
    assert hist[-1]["bucket"] == "[0.75,1.00]"


def test_summarize_species_counts_and_histogram():
    per_image = [
        [
            {"label": "animal", "species": "zebra", "confidence": 0.6},
            {"label": "animal", "species": "zebra", "confidence": 0.9},
            {"label": "person", "confidence": 0.8},
        ],
        [{"label": "animal", "species": "buffalo", "confidence": 0.3}],
        [],                                             # zero-detection image
        [{"label": "animal", "confidence": 0.4}],       # unclassifiable crop
    ]
    out = e2e.summarize_species(per_image)
    assert out["counts"] == {"buffalo": 1, "unclassified": 1, "zebra": 2}
    assert sum(b["n"] for b in out["detector_conf_histogram"]) == 4


def test_percentiles_shape():
    p = e2e._percentiles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5)
    assert p["p50"] == 5.5 and p["p95"] == 10.0 and p["mean"] == 5.5
    assert e2e._percentiles([]) == {}


def test_json_roundtrip_of_batch_shape():
    records = [e2e.detection_record(_animal_det(), 550, 412, species="rhino")]
    batch = e2e.build_batch(_fake_image_meta(is_night=True), records, "raw/c/i.jpg")
    loaded = json.loads(json.dumps(batch))
    assert e2e.validate_batch(loaded) == 1    # survives the jsonl round trip
