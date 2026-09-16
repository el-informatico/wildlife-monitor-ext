"""Offline tests for the S5 final evaluation stage (eval/run_final.py).

Pure-logic coverage: the Wellington/person-GT probe, the matching
protocol reuse, FP/camera/day grouping, the person-load table, and the
markdown renderer. Everything runs on synthetic fixtures — no pixels,
no parquet, no cv2 (numpy via eval.metrics only, as in test_metrics).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import run_final  # noqa: E402


# ---------------------------------------------------------------------------
# Wellington / person-GT probe
#


def _write_manifest_cats(path: Path, cats: list[str]) -> None:
    path.write_text(json.dumps(
        {"categories": [{"id": i, "name": n} for i, n in enumerate(cats)]}))


def test_wellington_probe_absent(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "african-wildlife").mkdir()
    (data / "missouri").mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "african-wildlife.zip").write_bytes(b"z")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest_cats(manifests / "mo.json", ["empty", "roe_deer"])

    sample = {"images": [
        {"image_id": "x", "gt_boxes": [{"bbox": [1, 2, 3, 4],
                                        "category": "animal"}]}]}
    sample_path = tmp_path / "eval_sample_s2.json"
    sample_path.write_text(json.dumps(sample))
    monkeypatch.setattr(run_final, "load_sample", lambda p=None: sample)
    # default arg points at the real processed/ — redirect explicitly
    probe = run_final.wellington_probe(data, downloads, manifests)

    assert probe["wellington_present"] is False
    assert probe["person_gt_in_any_vocabulary"] == []
    assert probe["person_p_r_status"] == "NOT_COMPUTABLE"
    assert "mo.json" in probe["gt_vocabularies"]


def test_wellington_probe_present_with_person_gt(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "wellington_camera_traps").mkdir()
    sample = {"images": []}
    monkeypatch.setattr(run_final, "load_sample", lambda p=None: sample)
    probe = run_final.wellington_probe(data, tmp_path / "downloads",
                                       tmp_path / "manifests")
    assert probe["wellington_present"] is True
    assert len(probe["wellington_hits"]) == 1
    # presence alone makes person P/R computable in principle
    assert probe["person_p_r_status"] == "computable"


def test_wellington_probe_person_category_detected(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest_cats(manifests / "x.json", ["empty", "human", "fox"])
    sample = {"images": []}
    monkeypatch.setattr(run_final, "load_sample", lambda p=None: sample)
    probe = run_final.wellington_probe(tmp_path / "data",
                                       tmp_path / "downloads", manifests)
    assert probe["person_gt_in_any_vocabulary"] == ["x.json"]


# ---------------------------------------------------------------------------
# Matching protocol reuse
#


def _img(iid, boxes_xywh, ds="missouri"):
    return {"image_id": iid, "source_dataset": ds, "gt_boxes": [
        {"bbox": b, "category": "animal"} for b in boxes_xywh]}


def _det(bbox, conf, cls=0):
    return {"bbox": bbox, "conf": conf, "cls": cls}


def test_match_image_protocol():
    # GT is xywh (as stored by the S2 sample), detections xyxy (as stored
    # by the detector) — greedy_match's exact convention.
    img = _img("a", [[10, 10, 20, 20], [100, 100, 30, 30]])
    dets = [_det([10, 10, 30, 30], 0.9),         # exact match with GT 0
            _det([500, 500, 540, 540], 0.4),     # disjoint → FP
            _det([101, 101, 129, 129], 0.3),     # IoU ~0.93 with GT 1
            _det([12, 12, 24, 24], 0.2)]         # below deploy floor
    res = run_final.match_image(img, dets, conf=0.25)
    assert res["n_gt"] == 2
    assert res["n_deployed"] == 3
    assert sum(res["matched"]) == 2
    assert len(res["fp_dets"]) == 1
    assert res["fp_dets"][0]["conf"] == 0.4
    assert res["person_dets"] == []
    # the sub-threshold detection never reaches the deployed set
    assert all(d["conf"] >= 0.25 for d in res["fp_dets"])


def test_match_image_person_label_routed():
    img = _img("a", [])
    dets = [_det([10, 10, 20, 20], 0.8, cls=1),
            _det([50, 50, 20, 20], 0.7, cls=2)]
    res = run_final.match_image(img, dets, conf=0.25)
    # no GT at all: both are FPs, but person/vehicle routing is by label
    assert len(res["fp_dets"]) == 2
    assert len(res["person_dets"]) == 1
    assert len(res["vehicle_dets"]) == 1
    assert res["person_dets"][0]["cls"] == 1


# ---------------------------------------------------------------------------
# FP/camera/day grouping
#


def _fixtures():
    sample = {"images": [
        _img("i1", [[10, 10, 20, 20]], ds="missouri"),
        _img("i2", [], ds="missouri"),
        _img("i3", [[10, 10, 20, 20]], ds="african_wildlife"),
    ]}
    detections = {
        "i1": [_det([10, 10, 30, 30], 0.9),      # TP
               _det([300, 300, 340, 340], 0.5)],  # FP
        "i2": [_det([300, 300, 340, 340], 0.6)],  # FP (empty frame)
        "i3": [_det([10, 10, 30, 30], 0.9)],      # TP
    }
    camera_days = {
        "i1": {"camera_id": "missouri-L1", "day": "2026-03-01"},
        "i2": {"camera_id": "missouri-L1", "day": "2026-03-01"},
        "i3": {"camera_id": "african_wildlife-L2", "day": "2026-03-02"},
    }
    ir_flags = {"i1": False, "i2": True, "i3": False}
    return sample, detections, camera_days, ir_flags


def test_fp_camera_day_cells_and_scopes():
    sample, dets, cam_days, ir = _fixtures()
    out = run_final.fp_camera_day(sample, dets, cam_days, ir, conf=0.25)

    assert out["parity_crosscheck"]["verdict"] == "MISMATCH"  # tiny fixture
    scopes = out["pr_by_scope"]
    assert scopes["all|all"] == {"tp": 2, "fp": 2, "fn": 0,
                                 "recall": 1.0, "precision": 0.5}
    assert scopes["missouri|day"]["fp"] == 1
    assert scopes["missouri|night"]["fp"] == 1
    assert scopes["african_wildlife|day"]["tp"] == 1

    cells = out["cells"]
    assert cells["n_cells"] == 2
    assert cells["fp_total"] == 2
    assert cells["fp_per_active_cell"]["mean"] == 1.0
    assert cells["fp_per_active_cell"]["max"] == 2
    # missouri-L1 has both images on one day → one cell, 2 FP
    worst = cells["worst_10"][0]
    assert worst["camera_id"] == "missouri-L1"
    assert worst["fp"] == 2
    assert worst["images"] == 2

    by_ds = {b["dataset"]: b for b in out["by_dataset"]}
    assert by_ds["missouri"]["fp"] == 2
    assert by_ds["missouri"]["fp_per_camera_day"] == 2.0
    assert by_ds["african_wildlife"]["fp"] == 0

    cam = {r["camera_id"]: r for r in out["per_camera"]}
    assert cam["missouri-L1"]["images"] == 2
    assert cam["missouri-L1"]["days_active"] == 1
    assert cam["missouri-L1"]["fp_per_image"] == 1.0


def test_fp_camera_day_parity_crosscheck_match():
    """The cross-check flips to 'match' when the fixture IS the target."""
    sample, dets, cam_days, ir = _fixtures()
    run_final.G3_COMMITTED_ORIG = run_final.G3_COMMITTED
    # inject a target equal to the fixture's overall counts
    run_final.G3_COMMITTED = {"tp": 2, "fp": 2, "fn": 0}
    try:
        out = run_final.fp_camera_day(sample, dets, cam_days, ir, conf=0.25)
        assert out["parity_crosscheck"]["verdict"] == "match"
    finally:
        run_final.G3_COMMITTED = run_final.G3_COMMITTED_ORIG


def test_person_load_grouping():
    sample = {"images": [
        _img("i1", []),
        _img("i2", []),
        _img("i3", []),
    ]}
    detections = {
        "i1": [_det([10, 10, 20, 20], 0.8, cls=1), _det([40, 40, 10, 10], 0.6, cls=1)],
        "i2": [_det([10, 10, 20, 20], 0.2, cls=1)],   # below deploy floor
        "i3": [_det([10, 10, 20, 20], 0.9, cls=0)],
    }
    cam_days = {
        "i1": {"camera_id": "african_wildlife-L9", "day": "2026-03-01"},
        "i2": {"camera_id": "african_wildlife-L9", "day": "2026-03-02"},
        "i3": {"camera_id": "african_wildlife-L9", "day": "2026-03-01"},
    }
    out = run_final.person_load(sample, detections, cam_days, conf=0.25)

    assert out["images_with_person_det"] == 1
    assert out["person_dets"] == 2
    assert out["conf_stats"]["max"] == 0.8
    assert out["conf_stats"]["p50"] == pytest.approx(0.7)
    assert out["per_camera"][0]["camera_id"] == "african_wildlife-L9"
    assert out["per_camera"][0]["dets"] == 2
    assert out["per_camera"][0]["max_conf"] == 0.8
    assert out["per_day"][0] == {"day": "2026-03-01", "dets": 2, "images": 1}


# ---------------------------------------------------------------------------
# Renderer
#


def test_render_tables_smoke():
    sample, dets, cam_days, ir = _fixtures()
    report = {
        "fp_camera_day": run_final.fp_camera_day(sample, dets, cam_days, ir),
        "person_load": run_final.person_load(sample, dets, cam_days),
        "wellington": {
            "wellington_present": False,
            "wellington_hits": [],
            "person_gt_in_any_vocabulary": [],
            "gt_vocabularies": {"mo.json": ["empty", "roe_deer"]},
            "person_p_r_status": "NOT_COMPUTABLE",
        },
        "agent_queue_crossref": {
            "person_events_total": 7,
            "escalation_load": {"agent_escalates": 0,
                                "agent_person_events_queued": 7,
                                "baseline_escalates": 5},
        },
    }
    md = run_final.render_tables(report)
    assert "FP per camera per day" in md
    assert "missouri-L1" in md
    assert "NOT_COMPUTABLE" in md
    assert "agent escalates 0 / queues 7" in md
    # no raw python dict rendering leaks into the markdown
    assert "{'" not in md and "{\"" not in md
    assert "TP 2 / FP 2 / FN 0" in md


def test_render_tables_g3_crosscheck_format():
    report = {
        "fp_camera_day": {
            "conf": 0.25,
            "parity_crosscheck": {
                "computed": {"tp": 812, "fp": 676, "fn": 210},
                "g3_committed": {"tp": 812, "fp": 676, "fn": 210},
                "verdict": "match"},
            "pr_by_scope": {"all|all": {"tp": 812, "fp": 676, "fn": 210,
                                        "recall": 0.7945,
                                        "precision": 0.5457}},
            "cells": {"n_cells": 1, "n_images": 1, "fp_total": 1,
                      "fp_per_active_cell": {"mean": 1.0, "p50": 1.0,
                                             "p95": 1.0, "max": 1},
                      "zero_fp_cells": 0,
                      "worst_10": [{"camera_id": "c", "day": "d",
                                    "fp": 1, "images": 1}]},
            "per_camera": [{"camera_id": "c", "images": 1, "days_active": 1,
                            "fp": 1, "fp_per_day": 1.0, "fp_per_image": 1.0}],
            "by_dataset": [{"dataset": "d", "cameras": 1, "cells": 1,
                            "images": 1, "fp": 1, "fp_per_camera_day": 1.0}],
            "clock": "synthetic",
        },
        "person_load": {
            "conf": 0.25, "images_with_person_det": 0, "person_dets": 0,
            "conf_stats": {"min": None, "p50": None, "max": None},
            "per_camera": [], "per_day": [], "gt_status": "none",
        },
        "wellington": {
            "wellington_present": False, "wellington_hits": [],
            "person_gt_in_any_vocabulary": [],
            "gt_vocabularies": {}, "person_p_r_status": "NOT_COMPUTABLE",
        },
    }
    md = run_final.render_tables(report)
    assert "TP 812 / FP 676 / FN 210" in md
    assert "MISMATCH" not in md
