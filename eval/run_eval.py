"""Run the G3 perception evaluation (Sprint 3).

Four stages over the S1/S2 datasets and artifacts (see eval/metrics.py
for the pure glue this runner drives):

- ``motion`` — full-manifest MOG2 gate pass (per camera, capture order,
  warmup 2 frames/camera) with CLAHE driven by the CONTENT-based IR
  flag, caching per-frame gate state to ``processed/g3_motion.parquet``;
- ``clf`` — held-out split integrity FIRST, then classifier v0 metrics
  (top-1/top-5, micro and macro) on the trainer's own 375-crop test
  cache — byte-identical to the inputs v0 was trained on — plus a
  torch-prediction cross-check;
- ``det`` — detection recall/precision/FP-per-image at the deploy
  threshold (0.20/0.25/0.30 sweep) from the stored S2 cv2.dnn
  detections, per dataset × day/night, with an own-matcher vs
  parity-module cross-check;
- ``seq`` — the simple sequence logic over the e2e pipeline: MOG2
  trigger gate + max-confidence-per-trigger selection, measured on
  Missouri's real GT sequences and african_wildlife test-split bursts,
  with species accuracy on selected frames vs all frames (test split).

Day/night is defined from pixels (mean HSV saturation < 25 on the
≤1024px working decode ⇒ IR/night); the manifest's ``is_night`` is a
synthetic random assignment and is reported only as an agreement
statistic.

Outputs (under ``processed/``, prefix ``g3``): ``g3_report.json`` (full
report incl. failure-mining records), ``g3_metrics.csv`` (tidy long
format), ``g3_metrics.md`` (the human table). Usage — needs the
sprints-1-2 venv, see pipeline/README.md:

    python -m eval.run_eval [--stages motion,clf,det,seq] [--engine new]

Stage caches: ``g3_motion.parquet`` and the two seq detection JSONs are
reused across runs; pass ``--refresh`` to recompute them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from pipeline import e2e
from . import metrics, runtime

DEPLOY_CONF = e2e.DEPLOY_CONF
CONF_SWEEP = (0.20, 0.25, 0.30)
V0_TEST_REF = 0.9867  # classifier v0 held-out test accuracy (torch, 375 crops)
OUT_PREFIX = "g3"
MOTION_NAME = "g3_motion.parquet"
MO_SEQ_DETS = "g3_seq_dets_missouri.json"
AW_TEST_DETS = "g3_seq_dets_aw_test.json"
MINE_CAP = 40  # max embedded mining records per failure family

DATASETS = ("african_wildlife", "missouri")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


# ---------------------------------------------------------------------------
# Stage: motion — full-manifest MOG2 gate pass
# ---------------------------------------------------------------------------


def stage_motion(
    manifest: list[dict], ir_cache: runtime.IRFlagCache, refresh: bool = False
) -> tuple[list[dict], dict[str, Any]]:
    """Per-frame MOG2 gate state for the whole manifest.

    Mirrors the S1 motion bench (per camera, frames in capture order —
    camera_ts sort, which reproduces the ETL's exact sequence/frame
    order; warmup 2 frames/camera; working resolution ≤1024px) with one
    deliberate G3 change: CLAHE is applied on the CONTENT-based IR
    flag, not the manifest's synthetic random ``is_night``.
    """
    out_path = e2e.ext_root() / "processed" / MOTION_NAME
    if out_path.exists() and not refresh:
        import pyarrow.parquet as pq

        records = pq.read_table(out_path).to_pylist()
        return records, _motion_summary(records, cached=True)

    bench = e2e.import_wildlife("wildlife_monitor.motion.bench")
    mog2 = e2e.import_wildlife("wildlife_monitor.motion.mog2")
    night = e2e.import_wildlife("wildlife_monitor.motion.night")
    # Mirrored constants — refuse to run silently diverged.
    assert metrics.TRIGGER_FG_THRESHOLD == bench.TRIGGER_FG_THRESHOLD
    assert runtime.WORK_RESOLUTION == bench.MAX_LONG_SIDE

    tracker = mog2.CameraMotionTracker(history=50, var_threshold=24.0)
    enhancer = night.NightEnhancer()
    data_dir = e2e.ext_root() / "data"
    frame_counter: dict[str, int] = defaultdict(int)
    decode_failures: list[str] = []

    by_camera: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        by_camera[row["location_id"]].append(row)

    records: list[dict] = []
    wall0 = time.perf_counter()
    for location_id in sorted(by_camera):
        seq = sorted(by_camera[location_id], key=lambda r: r["camera_ts"])
        for row in seq:
            t0 = time.perf_counter()
            bgr = runtime.decode_capped(data_dir / row["rel_path"])
            t1 = time.perf_counter()
            if bgr is None:
                decode_failures.append(row["image_id"])
                continue
            ir = ir_cache.of(row["image_id"], bgr)
            gray = night.to_gray(bgr)
            ms_clahe = 0.0
            if ir:
                tc0 = time.perf_counter()
                gray = enhancer.enhance(gray)
                ms_clahe = (time.perf_counter() - tc0) * 1000
            tm0 = time.perf_counter()
            fg_ratio = round(tracker.process(location_id, gray).fg_ratio, 6)
            tm1 = time.perf_counter()
            warmup = frame_counter[location_id] < bench.WARMUP_FRAMES_PER_CAMERA
            frame_counter[location_id] += 1
            records.append(
                {
                    "image_id": row["image_id"],
                    "source_dataset": row["source_dataset"],
                    "location_id": location_id,
                    "sequence_id": row["sequence_id"],
                    "camera_ts": row["camera_ts"],
                    "split": row["split"],
                    "species": row["species"],
                    "fg_ratio": fg_ratio,
                    "warmup": warmup,
                    "gate": bool(metrics.gate_pass(fg_ratio, warmup)),
                    "ir": bool(ir),
                    "is_night_synthetic": bool(row["is_night"]),
                    "ms_decode": round((t1 - t0) * 1000, 3),
                    "ms_clahe": round(ms_clahe, 3),
                    "ms_mog2": round((tm1 - tm0) * 1000, 3),
                }
            )
    wall_s = time.perf_counter() - wall0

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records), out_path)
    summary = _motion_summary(records)
    summary["wall_seconds"] = round(wall_s, 2)
    summary["decode_failures"] = decode_failures
    return records, summary


def _motion_summary(records: list[dict], cached: bool = False) -> dict[str, Any]:
    def _block(rows: list[dict]) -> dict[str, Any]:
        active = [r for r in rows if not r["warmup"]]
        return {
            "n_images": len(rows),
            "n_cameras": len({r["location_id"] for r in rows}),
            "n_warmup": len(rows) - len(active),
            "n_gated": sum(1 for r in rows if r["gate"]),
            "frame_gate_rate": _rate(sum(1 for r in rows if r["gate"]), len(active)),
            "n_ir": sum(1 for r in rows if r["ir"]),
            "ir_rate": _rate(sum(1 for r in rows if r["ir"]), len(rows)),
            "fg_ratio": _percentiles([r["fg_ratio"] for r in active]),
            "fg_ratio_ir": _percentiles(
                [r["fg_ratio"] for r in active if r["ir"]]),
            "fg_ratio_day": _percentiles(
                [r["fg_ratio"] for r in active if not r["ir"]]),
            "ms_decode": _percentiles([r["ms_decode"] for r in rows]),
            "ms_mog2": _percentiles([r["ms_mog2"] for r in rows]),
        }

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ds[r["source_dataset"]].append(r)
    summary: dict[str, Any] = {
        "overall": _block(records),
        "by_dataset": {ds: _block(rows) for ds, rows in sorted(by_ds.items())},
        "by_dataset_ir": {
            f"{ds}|{'night' if ir else 'day'}": _block(
                [r for r in by_ds[ds] if r["ir"] == ir]
            )
            for ds in sorted(by_ds)
            for ir in (False, True)
        },
        # Transparency stat: the manifest is_night is synthetic random.
        "synthetic_is_night_agreement": {
            ds: {
                "n": len(rows),
                "agreement": round(metrics.agreement_rate(
                    [r["ir"] for r in rows], [r["is_night_synthetic"] for r in rows]
                ), 4),
                "ir_rate": _rate(sum(1 for r in rows if r["ir"]), len(rows)),
                "synthetic_night_rate": _rate(
                    sum(1 for r in rows if r["is_night_synthetic"]), len(rows)
                ),
            }
            for ds, rows in sorted(by_ds.items())
        },
    }
    if cached:
        summary["cached"] = True
    return summary


# ---------------------------------------------------------------------------
# Stage: clf — held-out classifier metrics (split integrity first)
# ---------------------------------------------------------------------------


def stage_clf(
    manifest: list[dict], ir_cache: runtime.IRFlagCache, engine: str
) -> dict[str, Any]:
    import cv2

    ext = e2e.ext_root()
    clf = e2e.import_wildlife("wildlife_monitor.classify.model")
    assert e2e.CLASSIFY_INPUT == clf.INPUT_SIZE, "classifier input size drifted"

    meta = json.loads((ext / "models" / "classifier_v0_meta.json").read_text())
    index = json.loads((ext / "processed" / "classifier_crops_v0.json").read_text())
    classes = {int(k): v for k, v in index["classes"].items()}
    assert classes == clf.CLASSES, "classifier vocabulary drifted from the crop index"

    test = [c for c in index["crops"] if c["split"] == "test"]
    aw = [r for r in manifest if r["source_dataset"] == "african_wildlife"]
    aw_by_rel = {r["rel_path"]: r for r in aw}

    # --- 1. held-out split integrity (fails loudly on any leak) --------
    integrity = metrics.check_split_integrity(
        [
            {
                "id": c["crop_id"],
                "split": c["split"],
                "class_name": c["class_name"],
                "source_image": c["source_image"],
            }
            for c in test
        ],
        allowed_splits={"test"},
        expected_counts={"test": meta["train"]["crops_by_split"]["test"]},
        vocab=set(classes.values()),
        heldout_sources={r["rel_path"] for r in aw if r["split"] == "test"},
        other_split_sources={r["rel_path"] for r in aw if r["split"] != "test"},
    )

    # --- 2. independent recompute: crop windows from the YOLO labels ---
    # Proves the cached JPEGs are the training-recipe crops of the
    # manifest's test images (not just any crops with matching names).
    data_dir = ext / "data"
    mismatches: list[dict] = []
    n_checked = 0
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for c in test:
        by_rel[c["source_image"]].append(c)
    for rel in sorted(by_rel):
        row = aw_by_rel[rel]
        boxes = e2e.aw_species_boxes(
            e2e.aw_label_path(data_dir, rel), row["width"], row["height"], classes
        )
        for c in sorted(by_rel[rel], key=lambda c: c["crop_id"]):
            idx = int(c["crop_id"].rsplit("__", 1)[1])
            problem = None
            if idx >= len(boxes):
                problem = "crop index beyond label-file boxes"
            else:
                b = boxes[idx]
                win = e2e.crop_window_yolo(*b["yolo"], row["width"], row["height"])
                want = (
                    [win[0], win[1], win[2] - win[0], win[3] - win[1]]
                    if win
                    else None
                )
                if b["species"] != c["class_name"]:
                    problem = "species mismatch vs label file"
                elif want != c["box_xywh"]:
                    problem = f"window {want} != index {c['box_xywh']}"
            if problem:
                mismatches.append({"crop_id": c["crop_id"], "problem": problem})
            else:
                n_checked += 1
    window_check = {
        "n_checked": n_checked,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:5],
        "recipe": "crop_window_yolo (margin 0.1, min edge 8px) == crop index box_xywh",
    }

    # --- 3. classify the cached crops (byte-identical to training) -----
    classifier = clf.ClassifierOpenCV(
        ext / "models" / "classifier_v0_resnet18.onnx", engine=engine
    )
    results: list[dict] = []
    for c in test:
        bgr = cv2.imread(str(ext / "processed" / c["path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cached crop missing: {c['path']}")
        pred, _ = classifier.classify(bgr)
        ranked = [int(i) for i in np.argsort(-np.asarray(pred["probs"]), kind="stable")]
        ir = ir_cache.get(aw_by_rel[c["source_image"]]["image_id"],
                          data_dir / c["source_image"])
        results.append({**c, "probs": pred["probs"], "ranked": ranked, "ir": ir})

    def _scoped(rows: list[dict]) -> dict[str, Any]:
        return metrics.topk_metrics(
            [r["class_id"] for r in rows], [r["ranked"] for r in rows],
            n_classes=len(classes), k=5, class_names=classes,
        )

    day = [r for r in results if r["ir"] is False]
    night = [r for r in results if r["ir"] is True]
    tk = {
        "all": _scoped(results),
        "day": _scoped(day),
        "night": _scoped(night),
    }
    conf = metrics.confusion_matrix(
        [r["class_id"] for r in results], [r["ranked"][0] for r in results], classes
    )

    # --- 4. torch cross-check (the trainer's own predictions) ----------
    torch_preds = json.loads(
        (ext / "processed" / "classifier_v0_torch_preds.json").read_text()
    )["test"]
    tp_by_id = {p["crop_id"]: p for p in torch_preds}
    agree = 0
    max_diff = 0.0
    for r in results:
        p = tp_by_id[r["crop_id"]]
        agree += int(r["ranked"][0] == p["top1"])
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(r["probs"], p["probs"])))
    crosscheck = {
        "n": len(results),
        "top1_agree": agree,
        "max_abs_prob_diff": round(max_diff, 4),
    }

    names = classes
    misclassified = [
        {
            "crop_id": r["crop_id"],
            "true": names[r["class_id"]],
            "pred": names[r["ranked"][0]],
            "probs": r["probs"],
            "crop_path": r["path"],
            "source_image": r["source_image"],
            "box_xywh": r["box_xywh"],
            "ir": r["ir"],
        }
        for r in results
        if r["ranked"][0] != r["class_id"]
    ]

    return {
        "split": "test (trainer's own 375-crop held-out index)",
        "n_crops": len(results),
        "integrity": integrity,
        "window_check": window_check,
        "topk": tk,
        "confusion": conf,
        "torch_crosscheck": crosscheck,
        "v0_test_ref_torch": V0_TEST_REF,
        "misclassified": misclassified,
    }


# ---------------------------------------------------------------------------
# Stage: det — detection metrics per dataset × day/night
# ---------------------------------------------------------------------------


def stage_det(ir_cache: runtime.IRFlagCache) -> dict[str, Any]:
    ext = e2e.ext_root()
    data_dir = ext / "data"
    sample = json.loads((ext / "processed" / "eval_sample_s2.json").read_text())
    images = sample["images"]
    blob = json.loads((ext / "processed" / "cv2_detections_s2_new.json").read_text())
    floor = blob["detections"]
    parity = e2e.import_wildlife("wildlife_monitor.detect.parity")

    irs = {
        img["image_id"]: ir_cache.get(img["image_id"], data_dir / img["rel_path"])
        for img in images
    }
    if any(v is None for v in irs.values()):
        missing = [k for k, v in irs.items() if v is None][:5]
        raise RuntimeError(f"IR flag unavailable for sample images: {missing}")

    def scope_key(img: dict) -> tuple[str, str]:
        dn = "night" if irs[img["image_id"]] else "day"
        return img["source_dataset"], dn

    # --- per-image deployment matching at each sweep threshold ---------
    per_conf: dict[float, dict[tuple[str, str], dict[str, int]]] = {}
    fn_cases: list[dict] = []
    fp_cases: list[dict] = []
    for conf in CONF_SWEEP:
        totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_images": 0}
        )
        for img in images:
            gt = img["gt_boxes"]  # xywh absolute, as stored by the S2 sample
            deployed = [
                d for d in floor.get(img["image_id"], []) if d["conf"] >= conf
            ]
            mapping = metrics.greedy_match(
                [b["bbox"] for b in gt],
                [d["bbox"] for d in deployed],
                [d["conf"] for d in deployed],
            )
            matched = [m >= 0 for m in mapping]
            for key in (
                ("all", "all"),
                (img["source_dataset"], "all"),
                ("all", scope_key(img)[1]),
                scope_key(img),
            ):
                t = totals[key]
                t["tp"] += sum(matched)
                t["fp"] += len(matched) - sum(matched)
                t["fn"] += len(gt) - sum(matched)
                t["n_gt"] += len(gt)
                t["n_images"] += 1
            if conf == DEPLOY_CONF:
                if len(gt) - sum(matched) > 0 and len(fn_cases) < MINE_CAP:
                    fn_cases.append(
                        {
                            "image_id": img["image_id"],
                            "rel_path": img["rel_path"],
                            "source_dataset": img["source_dataset"],
                            "day_night": scope_key(img)[1],
                            "n_gt": len(gt),
                            "n_deployed": len(deployed),
                            "n_matched": sum(matched),
                            "gt_boxes_xywh": gt,
                            "dets": deployed,
                            "matched": matched,
                        }
                    )
                fp = len(matched) - sum(matched)
                if fp >= 2 and len(fp_cases) < MINE_CAP:
                    fp_cases.append(
                        {
                            "image_id": img["image_id"],
                            "rel_path": img["rel_path"],
                            "source_dataset": img["source_dataset"],
                            "day_night": scope_key(img)[1],
                            "fp": fp,
                            "n_gt": len(gt),
                            "dets": deployed,
                            "matched": matched,
                            "gt_boxes_xywh": gt,
                        }
                    )
        per_conf[conf] = dict(totals)

    def pr_block(t: dict[str, int]) -> dict[str, Any]:
        return {
            **t,
            "recall": _rate(t["tp"], t["tp"] + t["fn"]),
            "precision": _rate(t["tp"], t["tp"] + t["fp"]),
            "fp_per_image": _rate(t["fp"], t["n_images"]),
        }

    table: dict[str, Any] = {}
    for conf in CONF_SWEEP:
        table[str(conf)] = {
            f"{ds}|{dn}": pr_block(t)
            for (ds, dn), t in sorted(per_conf[conf].items())
        }

    # --- own-matcher vs parity-module cross-check ----------------------
    par = parity.engine_metrics(images, floor, DEPLOY_CONF)["scopes"]["overall"]
    own = pr_block(per_conf[DEPLOY_CONF][("all", "all")])
    crosscheck = {
        "own_matcher": {k: own[k] for k in ("tp", "fp", "fn", "recall", "precision")},
        "parity_module": {
            k: par[k] for k in ("tp", "fp", "fn", "recall", "precision")
        },
        "match": all(own[k] == par[k] for k in ("tp", "fp", "fn")),
    }

    # --- synthetic is_night transparency stat on this sample -----------
    agreement = {
        ds: {
            "n": n,
            "agreement": round(metrics.agreement_rate(
                [irs[i["image_id"]] for i in images if i["source_dataset"] == ds],
                [i["is_night"] for i in images if i["source_dataset"] == ds],
            ), 4),
        }
        for ds, n in sorted(
            (ds, sum(1 for i in images if i["source_dataset"] == ds))
            for ds in {i["source_dataset"] for i in images}
        )
    }

    return {
        "split": "S2 800-image deterministic evaluation sample "
                 "(seed 42; aw_with_gt=400/mo_with_gt=300/mo_empty=100)",
        "source": "stored cv2.dnn detections (floor 0.10), engine new",
        "deploy_conf": DEPLOY_CONF,
        "table": table,
        "matcher_crosscheck": crosscheck,
        "synthetic_is_night_agreement": agreement,
        "fn_cases": fn_cases,
        "fp_cases": fp_cases,
    }


# ---------------------------------------------------------------------------
# Stage: seq — MOG2 trigger gate + max-per-trigger selection
# ---------------------------------------------------------------------------


def _detect_frames(
    rows: list[dict], detector, cache_name: str, refresh: bool
) -> dict[str, list[dict]]:
    """MDv6 (floor 0.10) over the given manifest rows, JSON-cached."""
    import cv2

    ext = e2e.ext_root()
    cache = ext / "processed" / cache_name
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    data_dir = ext / "data"
    out: dict[str, list[dict]] = {}
    failed: list[str] = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        bgr = cv2.imread(str(data_dir / row["rel_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            failed.append(row["image_id"])
            continue
        out[row["image_id"]] = runtime.detect_image(detector, bgr, e2e.CONF_FLOOR)
    cache.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if failed:
        raise RuntimeError(f"decode failures during seq detection: {failed[:5]}")
    return out


def _frame_record(
    row: dict,
    dets: list[dict],
    gt: list[dict],
    motion: dict | None,
) -> dict[str, Any]:
    """Per-frame record for the sequence roll-up (metrics.sequence_rollup)."""
    gate = bool(motion["gate"]) if motion else False
    deployed = [d for d in dets if d["conf"] >= DEPLOY_CONF]
    mapping = metrics.greedy_match(
        [b["bbox"] for b in gt],
        [d["bbox"] for d in deployed],
        [d["conf"] for d in deployed],
    )
    matched = [m >= 0 for m in mapping]
    animals = [
        {"conf": d["conf"], "matched": matched[di]}
        for di, d in enumerate(deployed)
        if e2e.DET_LABELS[d["cls"]] == "animal"
    ]
    return {
        "image_id": row["image_id"],
        "rel_path": row["rel_path"],
        "gate": gate,
        "warmup": bool(motion["warmup"]) if motion else False,
        "fg_ratio": motion["fg_ratio"] if motion else None,
        "ir": bool(motion["ir"]) if motion else None,
        "has_gt": bool(gt),
        "matched": any(matched),
        "animal_dets": animals,
        "animal_fp": [a["conf"] for a in animals if not a["matched"]],
        "bbox_scale": (row["width"], row["height"]),
    }


def _rollup_sequence(
    frames: list[dict], dataset: str
) -> tuple[dict[str, Any], list[dict]]:
    """select_max_per_trigger + rollup + a day/night label (majority IR)."""
    sel_frames = [{"gate": f["gate"], "dets": f["animal_dets"]} for f in frames]
    sel_idx, reason = metrics.select_max_per_trigger(sel_frames)
    if sel_idx is not None:
        frames[sel_idx]["selected"] = True
        frames[sel_idx]["winner"] = max(
            frames[sel_idx]["animal_dets"], key=lambda a: a["conf"]
        )
    rollup = metrics.sequence_rollup(
        [
            {k: f.get(k) for k in (
                "image_id", "warmup", "gate", "has_gt", "matched",
                "animal_dets", "animal_fp", "selected",
            )}
            for f in frames
        ],
        reason,
    )
    irs = [f["ir"] for f in frames if f["ir"] is not None]
    rollup["dataset"] = dataset
    rollup["day_night"] = (
        "night" if irs and sum(irs) * 2 > len(irs) else "day"
    )  # ties (and unknown) count as day; documented
    return rollup, frames


def stage_seq(
    manifest: list[dict],
    motion_records: list[dict],
    engine: str,
    refresh: bool = False,
) -> dict[str, Any]:
    ext = e2e.ext_root()
    data_dir = ext / "data"
    clf = e2e.import_wildlife("wildlife_monitor.classify.model")
    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    mo_gt = runtime.missouri_gt()
    detector = mdv6.MDv6OpenCV(ext / "models" / "MDV6-yolov10-c.onnx", engine=engine)
    classifier = clf.ClassifierOpenCV(
        ext / "models" / "classifier_v0_resnet18.onnx", engine=engine
    )
    aw_load = runtime.gt_loader(data_dir, clf.CLASSES, mo_gt)

    motion_by_id = {m["image_id"]: m for m in motion_records}
    row_by_id = {r["image_id"]: r for r in manifest}

    def seq_frames(rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda r: r["camera_ts"])

    # --- Missouri: every sequence carrying ≥1 boxed GT frame ----------
    mo_by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        if r["source_dataset"] == "missouri":
            mo_by_seq[r["sequence_id"]].append(r)
    gt_seqs = {
        sid: seq_frames(rows)
        for sid, rows in sorted(mo_by_seq.items())
        if any(mo_gt.boxes_for(r["rel_path"]) for r in rows)
    }
    mo_rows = [row for rows in gt_seqs.values() for row in rows]
    mo_dets = _detect_frames(mo_rows, detector, MO_SEQ_DETS, refresh)

    mo_rollups: list[dict] = []
    mo_frames: dict[str, list[dict]] = {}
    for sid, rows in gt_seqs.items():
        frames = [
            _frame_record(
                row, mo_dets[row["image_id"]],
                mo_gt.boxes_for(row["rel_path"]),
                motion_by_id.get(row["image_id"]),
            )
            for row in rows
        ]
        rollup, frames = _rollup_sequence(frames, "missouri")
        rollup["sequence_id"] = sid
        mo_rollups.append(rollup)
        mo_frames[sid] = frames

    # --- african_wildlife: test-split bursts ---------------------------
    aw_test = [r for r in manifest
               if r["source_dataset"] == "african_wildlife" and r["split"] == "test"]
    aw_by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in aw_test:
        aw_by_seq[r["sequence_id"]].append(r)
    aw_dets = _detect_frames(aw_test, detector, AW_TEST_DETS, refresh)

    aw_rollups: list[dict] = []
    aw_frames: dict[str, list[dict]] = {}
    for sid, rows in sorted(aw_by_seq.items()):
        frames = [
            _frame_record(
                row, aw_dets[row["image_id"]],
                aw_load(row),
                motion_by_id.get(row["image_id"]),
            )
            for row in seq_frames(rows)
        ]
        rollup, frames = _rollup_sequence(frames, "african_wildlife")
        rollup["sequence_id"] = sid
        aw_rollups.append(rollup)
        aw_frames[sid] = frames

    # Burst purity: AW bursts were formed from file order and may mix
    # splits. Species metrics stay on pure-test bursts (train-split crops
    # were seen in training); gate/detection metrics use all test frames.
    aw_all_by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        if r["source_dataset"] == "african_wildlife":
            aw_all_by_seq[r["sequence_id"]].append(r)
    pure_test = {
        sid for sid, rows in aw_all_by_seq.items()
        if all(r["split"] == "test" for r in rows)
    }

    # --- species accuracy: all test frames vs selected frames ---------
    import cv2

    def species_call(row: dict, det: dict) -> tuple[str, float] | None:
        bgr = cv2.imread(str(data_dir / row["rel_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        window = e2e.crop_window(*det["bbox"], row["width"], row["height"])
        if window is None:
            return None
        pred, _ = classifier.classify(e2e.extract_crop(bgr, window))
        return pred["class_name"], pred["prob"]

    # all-frames: every deployed animal det on a test frame that matched
    # species-labelled GT.
    all_pairs: list[dict] = []
    for row in sorted(aw_test, key=lambda r: r["image_id"]):
        gt = aw_load(row)
        dets = aw_dets[row["image_id"]]
        deployed = [d for d in dets if d["conf"] >= DEPLOY_CONF]
        mapping = metrics.greedy_match(
            [b["bbox"] for b in gt], [d["bbox"] for d in deployed],
            [d["conf"] for d in deployed],
        )
        for di, det in enumerate(deployed):
            if e2e.DET_LABELS[det["cls"]] != "animal" or mapping[di] < 0:
                continue
            gt_species = gt[mapping[di]]["species"]
            if not gt_species:
                continue
            call = species_call(row, det)
            if call is None:
                continue
            all_pairs.append(
                {
                    "image_id": row["image_id"],
                    "gt": gt_species,
                    "pred": call[0],
                    "prob": call[1],
                }
            )

    # selected-frames: the winning detection of each selected pure-test burst.
    sel_pairs: list[dict] = []
    sel_missing_gt = 0
    for sid in sorted(aw_frames):
        if sid not in pure_test:
            continue
        frames = aw_frames[sid]
        sel = next((f for f in frames if f.get("selected")), None)
        if sel is None or "winner" not in sel:
            continue
        row = row_by_id[sel["image_id"]]
        # winner det: recompute from cached detections (frames store flags only)
        dets = aw_dets[sel["image_id"]]
        gt = aw_load(row)
        deployed = [d for d in dets if d["conf"] >= DEPLOY_CONF
                    and e2e.DET_LABELS[d["cls"]] == "animal"]
        mapping = metrics.greedy_match(
            [b["bbox"] for b in gt], [d["bbox"] for d in deployed],
            [d["conf"] for d in deployed],
        )
        winner = max(range(len(deployed)), key=lambda i: deployed[i]["conf"])
        if mapping[winner] < 0 or not gt[mapping[winner]]["species"]:
            sel_missing_gt += 1
            continue
        call = species_call(row, deployed[winner])
        if call is None:
            continue
        sel_pairs.append(
            {
                "sequence_id": sid,
                "image_id": sel["image_id"],
                "gt": gt[mapping[winner]]["species"],
                "pred": call[0],
                "prob": call[1],
            }
        )

    def _acc(pairs: list[dict]) -> dict[str, Any]:
        correct = sum(1 for p in pairs if p["pred"] == p["gt"])
        return {
            "n": len(pairs),
            "correct": correct,
            "accuracy": _rate(correct, len(pairs)),
            "errors": [
                {k: p[k] for k in ("image_id", "gt", "pred", "prob")}
                if "image_id" in p else p
                for p in pairs if p["pred"] != p["gt"]
            ],
        }

    species = {
        "scope": "african_wildlife test split",
        "all_frames": _acc(all_pairs),
        "selected_frames_pure_test_bursts": _acc(sel_pairs),
        "selected_winner_without_species_gt": sel_missing_gt,
        "pure_test_bursts": len(pure_test & set(aw_frames)),
        "test_bursts_total": len(aw_frames),
    }

    # --- aggregate + trigger coverage ----------------------------------
    rollups = mo_rollups + aw_rollups
    aggregate = metrics.aggregate_sequences(
        rollups,
        scopes={
            "dataset": list(DATASETS),
            "day_night": ["day", "night"],
        },
    )

    def _coverage(rows: list[dict]) -> dict[str, Any]:
        with_gate = sum(1 for r in rows if r["frames_gated"] > 0)
        observed = sum(
            1 for r in rows if r["animal_dets_analyzed"] > 0
        )
        return {
            "sequences": len(rows),
            "with_gate_pass": with_gate,
            "with_animal_detection_on_gated": observed,
            "trigger_observed_rate": _rate(observed, len(rows)),
            "no_gate_pass": sum(1 for r in rows if r["selected_reason"] == "no_gate_pass"),
            "no_detection": sum(1 for r in rows if r["selected_reason"] == "no_detection"),
            "selected": sum(1 for r in rows if r["selected_reason"] == "selected"),
        }

    coverage = {
        "all": _coverage(rollups),
        **{
            f"{ds}|{dn}": _coverage(
                [r for r in rollups if r["dataset"] == ds and r["day_night"] == dn]
            )
            for ds in DATASETS
            for dn in ("day", "night")
        },
    }

    # --- failure-mining records ----------------------------------------
    selection_loss = [
        {
            "sequence_id": r["sequence_id"],
            "dataset": r["dataset"],
            "day_night": r["day_night"],
            "reason": r["selected_reason"],
            # NB: only the winning frame carries "selected" (set in
            # _rollup_sequence); siblings report False here.
            "frames": [
                {**{k: f[k] for k in (
                    "image_id", "rel_path", "gate", "fg_ratio", "ir",
                    "has_gt", "matched", "animal_dets",
                )}, "selected": bool(f.get("selected"))}
                for f in mo_frames.get(r["sequence_id"], [])
                + aw_frames.get(r["sequence_id"], [])
            ],
        }
        for r in rollups
        if r["has_gt"] and r["gated_union_matched"] and not r["selected_matched"]
    ][:MINE_CAP]
    gate_lost_gt = [
        {
            "sequence_id": r["sequence_id"],
            "dataset": r["dataset"],
            "day_night": r["day_night"],
            "gt_frames": r["gt_frames"],
            "gt_frames_gated": r["gt_frames_gated"],
            "frames_gated": r["frames_gated"],
        }
        for r in rollups
        if r["has_gt"] and not r["gated_union_matched"]
    ][:MINE_CAP]

    return {
        "definition": {
            "trigger": f"MOG2 fg_ratio > {metrics.TRIGGER_FG_THRESHOLD} "
                       "(per camera, capture order, warmup 2 frames/camera, "
                       "CLAHE on content-IR frames)",
            "selection": "among gated frames of a sequence, the frame with "
                         "the highest-confidence deployed animal detection; "
                         "its winning detection is the sequence representative",
        },
        "missouri_gt_sequences": {
            "sequences": len(gt_seqs),
            "frames": len(mo_rows),
            "frames_with_boxes": sum(1 for r in mo_rows if mo_gt.boxes_for(r["rel_path"])),
        },
        "aw_test": {
            "images": len(aw_test),
            "bursts": len(aw_frames),
            "pure_test_bursts": species["pure_test_bursts"],
        },
        "species": species,
        "aggregate": aggregate,
        "coverage": coverage,
        "mining": {
            "selection_loss": selection_loss,
            "gate_lost_gt": gate_lost_gt,
        },
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    parts: dict[str, Any],
    stages: list[str],
    engine: str,
    wall_s: float,
) -> tuple[dict[str, Any], str, str]:
    ext = e2e.ext_root()
    import cv2

    day_night = {
        "definition": (
            "content-based IR monochrome flag: mean HSV saturation < "
            f"{metrics.IR_SAT_THRESHOLD} on the ≤{runtime.WORK_RESOLUTION}px "
            "working decode (night camera-trap frames are infrared grayscale; "
            "Missouri's distribution is sharply bimodal, p50 0.8 vs p95 122)"
        ),
        "manifest_is_night": (
            "the manifest's is_night is a SYNTHETIC random assignment "
            "(etl.synthetic, night_weight 0.45) — reported only as an "
            "agreement statistic, never used to split G3 numbers"
        ),
    }
    for part in ("motion", "det"):
        if part in parts:
            key = "synthetic_is_night_agreement"
            if key in parts[part]:
                day_night[f"agreement_{part}"] = parts[part][key]

    meta = {
        "stages_run": stages,
        "engine": engine,
        "seed": e2e.SEED,
        "sampling": "none — full populations (no random draws in this harness)",
        "wall_seconds": round(wall_s, 2),
        "versions": {
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "params": {
            "deploy_conf": DEPLOY_CONF,
            "conf_floor": e2e.CONF_FLOOR,
            "conf_sweep": list(CONF_SWEEP),
            "match_iou": metrics.MATCH_IOU,
            "ir_sat_threshold": metrics.IR_SAT_THRESHOLD,
            "trigger_fg_threshold": metrics.TRIGGER_FG_THRESHOLD,
            "warmup_frames_per_camera": 2,
            "work_resolution": runtime.WORK_RESOLUTION,
        },
        "models": {
            name: runtime.sha256(ext / "models" / name)
            for name in ("MDV6-yolov10-c.onnx", "classifier_v0_resnet18.onnx")
        },
        "inputs": {
            name: runtime.sha256(ext / "processed" / name)
            for name in (
                "manifest.parquet", "eval_sample_s2.json",
                "cv2_detections_s2_new.json", "classifier_crops_v0.json",
                "classifier_v0_torch_preds.json",
            )
        },
    }

    clf_part = parts.get("clf")
    caveats = [
        "CPU-only environment (no nvidia-smi); single-image inference, no batching to degrade.",
        "Day/night is a content-based IR flag; the manifest's is_night is synthetic random "
        "(agreement ≈ chance) and is never used to split metrics.",
    ]
    if clf_part:
        night_n = clf_part["topk"]["night"]["n"]
        caveats += [
            f"classifier v0 has 4 classes, so top-5 is degenerate (top-4 always "
            f"contains the truth → 1.0 by construction); top-1 is the real signal",
            f"AW night subsample is n={night_n} crops — too small for a macro "
            f"reading; reported for completeness, not as a night-performance claim",
            "Missouri carries no classifier metrics: its species are disjoint from "
            "v0's 4-species AW vocabulary (closed-set OOV — see the failure catalog)",
            "CCT20 is metadata-only in this repo (images never downloaded) — excluded",
        ]
    if "det" in parts:
        caveats.append(
            "detection scopes use the S2 800-image sample (seed 42): the AW stratum "
            "spans train/val/test images — no leak concern for MDv6 (pretrained, not "
            "trained here), but detector n is the sample, not the full dataset"
        )
    if "seq" in parts:
        caveats += [
            "Missouri sequence recall is box-level: a selected frame without a GT box "
            "counts as not-matched even when the animal is present (sequence-level GT)",
            "species selected-vs-all uses pure-test AW bursts only; mixed-split bursts "
            "are excluded from both sides of that comparison",
            "the motion pass applies CLAHE on the content-IR flag — S1's bench used "
            "the synthetic is_night, so trigger rates are not directly comparable",
        ]

    report = {
        "meta": meta,
        "day_night": day_night,
        **parts,
        "caveats": caveats,
    }
    csv_text, md_text = _tables(report)
    return report, csv_text, md_text


def _tables(report: dict[str, Any]) -> tuple[str, str]:
    from . import metrics as m

    rows: list[dict] = []
    md: list[str] = ["# G3 metrics", ""]

    if "clf" in report:
        clf = report["clf"]
        md.append("## Classifier (african_wildlife test split, trainer's 375-crop cache)")
        md.append("")
        header = ["scope", "n", "top1 micro", "top1 macro", "top5 micro", "top5 macro"]
        trows = []
        for scope in ("all", "day", "night"):
            t = clf["topk"][scope]
            trows.append([
                scope, t["n"], t["top1_micro"], t["top1_macro"],
                t["top5_micro"], t["top5_macro"],
            ])
            for metric in ("top1_micro", "top1_macro", "top5_micro", "top5_macro"):
                rows.append({
                    "family": "clf", "dataset": "african_wildlife",
                    "split": "test", "day_night": scope,
                    "metric": metric, "value": t[metric], "n": t["n"],
                })
        md.append(m.markdown_table(header, trows))
        md.append("")
        md.append(f"top-5 is degenerate (4 classes; top-4 ⊇ truth). Torch reference: "
                  f"{clf['v0_test_ref_torch']} (375 crops). Per-class: "
                  + ", ".join(
                      f"{k} n={v['n']} top1={v['top1_acc']}"
                      for k, v in clf["topk"]["all"]["per_class"].items()
                  ))
        md.append("")

    if "det" in report:
        det = report["det"]
        md.append("## Detection (S2 800-image sample, parity protocol IoU ≥ 0.5)")
        md.append("")
        for conf in CONF_SWEEP:
            table = det["table"][str(conf)]
            md.append(f"### conf ≥ {conf}" + (" (deploy)" if conf == DEPLOY_CONF else ""))
            md.append("")
            header = ["dataset", "day/night", "images", "GT", "TP", "FP", "FN",
                      "recall", "precision", "FP/img"]
            trows = []
            for key in ("all|all", "african_wildlife|all", "missouri|all",
                        "all|day", "all|night",
                        "african_wildlife|day", "african_wildlife|night",
                        "missouri|day", "missouri|night"):
                if key not in table:
                    continue
                b = table[key]
                ds, dn = key.split("|")
                trows.append([
                    ds, dn, b["n_images"], b["n_gt"], b["tp"], b["fp"], b["fn"],
                    b["recall"], b["precision"], b["fp_per_image"],
                ])
                for metric in ("recall", "precision", "fp_per_image"):
                    rows.append({
                        "family": f"det@{conf}", "dataset": ds, "split": "s2_sample",
                        "day_night": dn, "metric": metric,
                        "value": b[metric], "n": b["n_images"],
                    })
            md.append(m.markdown_table(header, trows))
            md.append("")

    if "seq" in report:
        seq = report["seq"]
        md.append("## Sequence logic (MOG2 trigger + max-per-trigger selection)")
        md.append("")
        md.append("### Trigger coverage")
        md.append("")
        header = ["scope", "sequences", "with gate pass", "trigger observed",
                  "rate", "selected", "no gate", "no det"]
        trows = []
        for key in ("all", "african_wildlife|day", "african_wildlife|night",
                    "missouri|day", "missouri|night"):
            c = seq["coverage"][key]
            trows.append([
                key, c["sequences"], c["with_gate_pass"],
                c["with_animal_detection_on_gated"], c["trigger_observed_rate"],
                c["selected"], c["no_gate_pass"], c["no_detection"],
            ])
            rows.append({
                "family": "seq_trigger", "dataset": key, "split": "-",
                "day_night": "-",
                "metric": "trigger_observed_rate",
                "value": c["trigger_observed_rate"], "n": c["sequences"],
            })
        md.append(m.markdown_table(header, trows))
        md.append("")

        md.append("### GT-sequence recall ladder (union → gated → selected)")
        md.append("")
        header = ["scope", "GT seqs", "union recall", "gated recall",
                  "selected recall", "selection loss", "frame gate rate",
                  "inference saving"]
        trows = []
        for key, block in seq["aggregate"].items():
            if block["gt_sequences"] == 0:
                continue
            trows.append([
                key, block["gt_sequences"], block["seq_union_recall"],
                block["seq_gated_recall"], block["seq_selected_recall"],
                block["selection_loss"], block["frame_gate_rate"],
                block["inference_saving"],
            ])
            for metric in ("seq_union_recall", "seq_gated_recall",
                           "seq_selected_recall", "selection_loss",
                           "frame_gate_rate", "inference_saving"):
                rows.append({
                    "family": "seq_recall", "dataset": key, "split": "-",
                    "day_night": "-", "metric": metric,
                    "value": block[metric], "n": block["gt_sequences"],
                })
        md.append(m.markdown_table(header, trows))
        md.append("")

        sp = seq["species"]
        md.append("### Species accuracy — selected vs all (AW test split)")
        md.append("")
        header = ["scope", "n", "correct", "accuracy"]
        trows = [
            ["all test frames", sp["all_frames"]["n"], sp["all_frames"]["correct"],
             sp["all_frames"]["accuracy"]],
            ["selected (pure-test bursts)", sp["selected_frames_pure_test_bursts"]["n"],
             sp["selected_frames_pure_test_bursts"]["correct"],
             sp["selected_frames_pure_test_bursts"]["accuracy"]],
        ]
        for scope, part in (("all_frames", sp["all_frames"]),
                            ("selected", sp["selected_frames_pure_test_bursts"])):
            rows.append({
                "family": "seq_species", "dataset": "african_wildlife",
                "split": "test", "day_night": "-", "metric": "accuracy",
                "value": part["accuracy"], "n": part["n"],
            })
        md.append(m.markdown_table(header, trows))
        md.append("")

    if "motion" in report:
        mo = report["motion"]
        md.append("## Motion gate (full manifest, content-IR CLAHE policy)")
        md.append("")
        header = ["scope", "images", "cameras", "warmup", "gated",
                  "gate rate", "IR rate", "fg p50", "fg p95"]
        trows = []
        blocks = [("overall", mo["overall"])] + [
            (f"{ds}|{'night' if ir else 'day'}", mo["by_dataset_ir"][f"{ds}|{'night' if ir else 'day'}"])
            for ds in DATASETS for ir in (False, True)
        ]
        for key, b in blocks:
            trows.append([
                key, b["n_images"], b["n_cameras"], b["n_warmup"], b["n_gated"],
                b["frame_gate_rate"], b["ir_rate"],
                b["fg_ratio"].get("p50"), b["fg_ratio"].get("p95"),
            ])
            rows.append({
                "family": "motion", "dataset": key, "split": "-", "day_night": "-",
                "metric": "frame_gate_rate", "value": b["frame_gate_rate"],
                "n": b["n_images"],
            })
        md.append(m.markdown_table(header, trows))
        md.append("")

    return m.rows_to_csv(rows), "\n".join(md)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    stages: tuple[str, ...] = ("motion", "clf", "det", "seq"),
    engine: str = "new",
    refresh: bool = False,
    out_prefix: str = OUT_PREFIX,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = runtime.load_manifest()
    ir_cache = runtime.IRFlagCache()

    parts: dict[str, Any] = {}
    motion_records: list[dict] | None = None

    if "motion" in stages:
        motion_records, parts["motion"] = stage_motion(manifest, ir_cache, refresh)
        ir_cache.flush()

    if "seq" in stages and motion_records is None:
        cache = e2e.ext_root() / "processed" / MOTION_NAME
        if not cache.exists():
            raise RuntimeError(
                "seq stage needs the motion pass — run with --stages motion,seq "
                "(or all) first"
            )
        import pyarrow.parquet as pq

        motion_records = pq.read_table(cache).to_pylist()

    if "clf" in stages:
        parts["clf"] = stage_clf(manifest, ir_cache, engine)
    if "det" in stages:
        parts["det"] = stage_det(ir_cache)
    if "seq" in stages:
        assert motion_records is not None
        parts["seq"] = stage_seq(manifest, motion_records, engine, refresh)

    ir_cache.flush()
    report, csv_text, md_text = build_report(
        parts, list(stages), engine, time.perf_counter() - t0
    )
    names = runtime.write_outputs(out_prefix, report, csv_text, md_text)
    report["outputs"] = names
    (e2e.ext_root() / "processed" / names["report_json"]).write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stages", default="motion,clf,det,seq",
        help="comma-separated subset of motion,clf,det,seq",
    )
    parser.add_argument("--engine", default="new", help="cv2.dnn graph engine")
    parser.add_argument(
        "--refresh", action="store_true",
        help="recompute cached stage outputs (motion parquet, seq detections)",
    )
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    args = parser.parse_args(argv)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in ("motion", "clf", "det", "seq")]
    if unknown:
        parser.error(f"unknown stages: {unknown}")
    report = run(stages, engine=args.engine, refresh=args.refresh,
                 out_prefix=args.out_prefix)
    print(json.dumps({k: report[k] for k in ("meta", "outputs") if k in report},
                     indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
