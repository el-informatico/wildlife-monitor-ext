"""Run the E2E union stage over the S2 evaluation sample.

For a deterministic sub-sample of ETL media: MDv6 detection (cv2.dnn) →
10%-margin 224 px crops → classifier v0 species labels → schema-valid
detection batches. Sanity metrics measured on the way: detection recall
vs GT boxes (S2 parity protocol), classifier accuracy on GT-box crops
by split (train-split crops were seen in training — the test-split
number is the honest comparison against classifier v0's 0.9867), and
species accuracy on detections matched to species-labelled GT.

Outputs (all under <ext>/processed/, prefix e2e_union_v0 by default):

- ``<prefix>_batches.jsonl`` — one schema-valid detection batch per line
  (the ingest contract's upload shape; every line is validated against
  infra.schema before writing);
- ``<prefix>_records.parquet`` — flat per-detection rows incl. species
  probability, GT match and inherited GT species (analytics view);
- ``<prefix>_report.json`` — timings, recall sweep, accuracies, counts;
- ``<prefix>_summary.md`` — human-readable summary.

Usage (needs the sprints-1-2 venv: OpenCV 5, numpy, pyarrow and the
wildlife_monitor package importable — see pipeline/README.md):

    python -m pipeline.run_e2e [--sizes aw_with_gt=100,...] [--engine new]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import e2e
from .e2e import CONF_FLOOR, DEPLOY_CONF, DEFAULT_SIZES, MATCH_IOU, SEED

WARMUP_IMAGES = 3   # processed but excluded from latency stats (S2 convention)
RECALL_SWEEP = (0.20, 0.25, 0.30)
V0_TEST_ACCURACY = 0.9867  # classifier v0 held-out test accuracy (375 crops)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(
    engine: str = "new",
    sample_path: Path | None = None,
    sizes: dict[str, int] | None = None,
    seed: int = SEED,
    deploy_conf: float = DEPLOY_CONF,
    out_prefix: str = "e2e_union_v0",
    conf_sweep: tuple[float, ...] = RECALL_SWEEP,
) -> dict[str, Any]:
    import cv2

    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    parity = e2e.import_wildlife("wildlife_monitor.detect.parity")
    clf = e2e.import_wildlife("wildlife_monitor.classify.model")

    # The glue mirrors upstream constants; refuse to run silently diverged.
    assert e2e.DET_LABELS == mdv6.CLASS_NAMES, "detector label map drifted"
    assert e2e.CLASSIFY_INPUT == clf.INPUT_SIZE, "classifier input size drifted"

    ext = e2e.ext_root()
    sample_path = sample_path or (ext / "processed" / "eval_sample_s2.json")
    manifest_path = ext / "processed" / "manifest.parquet"
    det_onnx = ext / "models" / "MDV6-yolov10-c.onnx"
    clf_onnx = ext / "models" / "classifier_v0_resnet18.onnx"
    data_dir = ext / "data"
    sizes = sizes or DEFAULT_SIZES

    sample = json.loads(sample_path.read_text())
    images = e2e.draw_sample(sample["images"], seed, sizes)
    manifest_join = e2e.load_manifest_join(manifest_path)

    detector = mdv6.MDv6OpenCV(det_onnx, engine=engine)
    classifier = clf.ClassifierOpenCV(clf_onnx, engine=engine)
    runtime = f"cv2.dnn (OpenCV {cv2.__version__}, engine {engine})"

    ms: dict[str, list[float]] = {
        "decode": [], "pre": [], "forward": [], "post": [],
        "crop_extract": [], "classify": [],
    }
    floor_dets: dict[str, list[dict]] = {}     # image_id -> dets ≥ CONF_FLOOR (raw)
    batches: list[dict] = []                   # schema-shaped, deploy conf only
    rows: list[dict] = []                      # flat parquet rows, deploy conf only
    gt_crop_eval: list[dict] = []              # classifier on GT-box crops (AW)
    decode_failures: list[str] = []
    n_gt_crop_skipped = 0
    wall0 = time.perf_counter()

    for i, img in enumerate(images):
        timed = i >= WARMUP_IMAGES
        t0 = time.perf_counter()
        bgr = cv2.imread(str(data_dir / img["rel_path"]), cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        floor_dets[img["image_id"]] = []
        if bgr is None:
            decode_failures.append(img["image_id"])
            batches.append(e2e.build_batch(img, [], _source_key(img, manifest_join),
                                           camera_id=_camera_id(img, manifest_join),
                                           runtime=runtime))
            continue
        h, w = bgr.shape[:2]

        # --- detect (S2 stage, per-stage timings like detect.run_cv2) -----
        params = mdv6.letterbox_params(w, h, detector.input_size)
        padded = mdv6.letterbox_image(bgr, params, detector.input_size)
        blob = mdv6.to_blob(padded)  # swapRB=False — pinned BGR semantics
        t2 = time.perf_counter()
        detector.net.setInput(blob)
        output = np.asarray(detector.net.forward())
        t3 = time.perf_counter()
        dets = mdv6.decode_prediction(output, params, CONF_FLOOR,
                                      mdv6.IOU_THRESHOLD, mdv6.MAX_DETECTIONS)
        t4 = time.perf_counter()
        dets_json = dets.to_json()
        floor_dets[img["image_id"]] = dets_json

        # --- GT for this image (species-annotated where available) --------
        gt = _gt_for_image(img, data_dir, clf.CLASSES)
        det_gt = e2e.match_dets_to_gt(
            [b["bbox"] for b in gt], [d["bbox"] for d in dets_json],
            [d["conf"] for d in dets_json], MATCH_IOU,
        )

        # --- crop + classify deployed animal detections -------------------
        det_records: list[dict] = []
        for di, det in enumerate(dets_json):
            label = e2e.DET_LABELS[det["cls"]]
            row: dict[str, Any] = {
                "image_id": img["image_id"], "source_dataset": img["source_dataset"],
                "stratum": img["stratum"], "split": img.get("split"),
                "rel_path": img["rel_path"], "det_idx": di, "label": label,
                "det_conf": det["conf"],
                "bbox_x1": det["bbox"][0], "bbox_y1": det["bbox"][1],
                "bbox_x2": det["bbox"][2], "bbox_y2": det["bbox"][3],
                "matched_gt": bool(det_gt[di] >= 0),
                "gt_species": gt[det_gt[di]]["species"] if det_gt[di] >= 0 else None,
                "species": None, "species_prob": None,
            }
            if det["conf"] < deploy_conf:
                continue  # decoded for the sweep, not deployed
            if label == "animal":
                tc0 = time.perf_counter()
                window = e2e.crop_window(*det["bbox"], w, h)
                crop = e2e.extract_crop(bgr, window) if window else None
                tc1 = time.perf_counter()
                if crop is not None:
                    pred, _ = classifier.classify(crop)
                    tc2 = time.perf_counter()
                    row["species"] = pred["class_name"]
                    row["species_prob"] = pred["prob"]
                    if timed:
                        ms["crop_extract"].append((tc1 - tc0) * 1000)
                        ms["classify"].append((tc2 - tc1) * 1000)
                det_records.append(e2e.detection_record(det, w, h, row["species"]))
            else:
                det_records.append(e2e.detection_record(det, w, h))
            rows.append(row)

        key = _source_key(img, manifest_join)
        batch = e2e.build_batch(img, det_records, key,
                                camera_id=_camera_id(img, manifest_join),
                                runtime=runtime)
        n_valid = e2e.validate_batch(batch)  # raises on contract violation
        assert n_valid == len(det_records)
        batches.append(batch)

        # --- classifier sanity on GT-box crops (AW only, by split) --------
        if img["source_dataset"] == "african_wildlife":
            for gi, gt_box in enumerate(gt):
                window = e2e.crop_window_yolo(*gt_box["yolo"], w, h)
                if window is None:
                    n_gt_crop_skipped += 1
                    continue
                crop = e2e.extract_crop(bgr, window)
                pred, _ = classifier.classify(crop)
                gt_crop_eval.append({
                    "image_id": img["image_id"], "split": img.get("split"),
                    "gt_index": gi, "gt_species": gt_box["species"],
                    "pred_species": pred["class_name"], "prob": pred["prob"],
                })

        if timed:
            ms["decode"].append((t1 - t0) * 1000)
            ms["pre"].append((t2 - t1) * 1000)
            ms["forward"].append((t3 - t2) * 1000)
            ms["post"].append((t4 - t3) * 1000)

    wall_s = time.perf_counter() - wall0

    report = _build_report(
        images=images, batches=batches, rows=rows, floor_dets=floor_dets,
        gt_crop_eval=gt_crop_eval, n_gt_crop_skipped=n_gt_crop_skipped,
        decode_failures=decode_failures, ms=ms, wall_s=wall_s,
        engine=engine, runtime=runtime, seed=seed, sizes=sizes,
        deploy_conf=deploy_conf, conf_sweep=conf_sweep, cv2=cv2,
        det_onnx=det_onnx, clf_onnx=clf_onnx, clf=clf, parity=parity,
    )
    _write_outputs(out_prefix, report, batches, rows, ext)
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camera_id(img: dict, manifest_join: dict[str, dict]) -> str | None:
    entry = manifest_join.get(img["image_id"])
    return entry["location_id"] if entry else None


def _source_key(img: dict, manifest_join: dict[str, dict]) -> str:
    """Relative object key in the documented raw-media bucket layout."""
    camera = _camera_id(img, manifest_join) or "unknown-cam"
    suffix = Path(img["rel_path"]).suffix or ".jpg"
    return f"raw/{camera}/{img['image_id']}{suffix}"


def _gt_for_image(img: dict, data_dir: Path, classes: dict[int, str]) -> list[dict]:
    """GT boxes for one sample image; AW boxes keep their YOLO species."""
    if img["source_dataset"] == "african_wildlife":
        return e2e.aw_species_boxes(
            e2e.aw_label_path(data_dir, img["rel_path"]),
            img["width"], img["height"], classes,
        )
    return [{"bbox": b["bbox"], "category": b["category"], "species": None}
            for b in img["gt_boxes"]]


def _recall_scopes(images, floor_dets, conf, parity) -> dict:
    return parity.engine_metrics(images, floor_dets, conf)["scopes"]


def _own_recall(images, floor_dets, conf, gt_loader) -> dict:
    """Recall from this module's matcher — cross-checks parity.engine_metrics."""
    tp = fp = fn = 0
    for img in images:
        dets = [d for d in floor_dets.get(img["image_id"], []) if d["conf"] >= conf]
        mapping = e2e.match_dets_to_gt(
            [b["bbox"] for b in gt_loader(img)], [d["bbox"] for d in dets],
            [d["conf"] for d in dets], MATCH_IOU,
        )
        tp += sum(1 for m in mapping if m >= 0)
        fp += sum(1 for m in mapping if m < 0)
        fn += len(img["gt_boxes"]) - sum(1 for m in mapping if m >= 0)
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": round(recall, 4) if recall is not None else None}


def _build_report(*, images, batches, rows, floor_dets, gt_crop_eval,
                  n_gt_crop_skipped, decode_failures, ms, wall_s, engine,
                  runtime, seed, sizes, deploy_conf, conf_sweep, cv2,
                  det_onnx, clf_onnx, clf, parity) -> dict[str, Any]:
    data_dir = e2e.ext_root() / "data"

    def gt_loader(img):
        return _gt_for_image(img, data_dir, clf.CLASSES)

    per_image_total = [
        ms["decode"][i] + ms["pre"][i] + ms["forward"][i] + ms["post"][i]
        for i in range(len(ms["decode"]))
    ]
    zero_det = sum(1 for b in batches if not b["detections"])

    # Recall sweep (parity module) + independent matcher cross-check.
    sweep = {str(c): _recall_scopes(images, floor_dets, c, parity) for c in conf_sweep}
    if str(deploy_conf) not in sweep:  # custom --deploy-conf outside the sweep
        sweep[str(deploy_conf)] = _recall_scopes(images, floor_dets, deploy_conf, parity)
    own = _own_recall(images, floor_dets, deploy_conf, gt_loader)
    par = sweep[str(deploy_conf)]["overall"]
    crosscheck = {"own_matcher": own["recall"], "parity_module": par["recall"],
                  "match": own["recall"] == par["recall"]}

    # Classifier on GT-box crops, by split (train crops were seen in training).
    by_split: dict[str, dict[str, Any]] = {}
    for entry in gt_crop_eval:
        s = by_split.setdefault(entry["split"], {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += entry["pred_species"] == entry["gt_species"]
    for s, v in by_split.items():
        v["accuracy"] = round(v["correct"] / v["n"], 4) if v["n"] else None
    if "test" in by_split:
        by_split["test"]["v0_test_ref"] = V0_TEST_ACCURACY

    # Species accuracy on deployed animal detections matched to species GT.
    ev = [r for r in rows if r["species"] and r["gt_species"]]
    sp_correct = sum(1 for r in ev if r["species"] == r["gt_species"])

    def _cls_id(name: str) -> int | None:
        return next((k for k, v in clf.CLASSES.items() if v == name), None)

    known = [(g, p) for g, p in
             ((_cls_id(r["gt_species"]), _cls_id(r["species"])) for r in ev)
             if g is not None and p is not None]
    species_on_dets = {
        "n_evaluable": len(ev),
        "correct": sp_correct,
        "accuracy": round(sp_correct / len(ev), 4) if ev else None,
        "confusion": clf.confusion_matrix(
            [g for g, _ in known], [p for _, p in known], len(clf.CLASSES)),
        "note": "deployed animal detections matched (IoU>=0.5) to a species-"
                "labelled AW GT box; missed/unmatched detections excluded",
    }

    deployed = [b["detections"] for b in batches]
    per_label = {label: sum(1 for dets in deployed for d in dets if d["label"] == label)
                 for label in ("animal", "person", "vehicle")}
    species_summary = e2e.summarize_species(deployed)

    return {
        "stage": "e2e_union_v0",
        "versions": {"cv2": cv2.__version__, "numpy": np.__version__},
        "models": {
            "detector": {"file": det_onnx.name, "sha256": _sha256(det_onnx)},
            "classifier": {"file": clf_onnx.name, "sha256": _sha256(clf_onnx)},
        },
        "params": {
            "engine": engine, "seed": seed, "sizes": sizes,
            "deploy_conf": deploy_conf, "conf_floor": CONF_FLOOR,
            "match_iou": MATCH_IOU, "warmup_images_excluded": WARMUP_IMAGES,
            "crop": {"margin": e2e.CROP_MARGIN, "min_edge_px": e2e.CROP_MIN_EDGE_PX,
                     "classify_size": e2e.CLASSIFY_INPUT,
                     "note": "in-memory crops; training used JPEG q95 cache"},
        },
        "sample": {
            "images": len(images),
            "by_stratum": {k: sum(1 for i in images if i["stratum"] == k) for k in sizes},
            "gt_boxes": sum(len(i["gt_boxes"]) for i in images),
            "decode_failures": decode_failures,
        },
        "wall_s": round(wall_s, 1),
        "latency_ms": {k: e2e._percentiles(v) for k, v in ms.items()}
        | {"detector_per_image": e2e._percentiles(per_image_total)},
        "detection": {
            "batches": len(batches), "zero_detection_images": zero_det,
            "deployed_detections": sum(len(d) for d in deployed),
            "per_label": per_label,
            "recall_sweep": sweep,
            "recall_crosscheck_at_deploy": crosscheck,
        },
        "classification": {
            "gt_crops": {"by_split": by_split,
                         "skipped_windows": n_gt_crop_skipped},
            "on_detections": species_on_dets,
            "species_summary": species_summary,
        },
    }


def _write_outputs(prefix: str, report: dict, batches: list[dict],
                   rows: list[dict], ext: Path) -> None:
    processed = ext / "processed"

    jsonl = processed / f"{prefix}_batches.jsonl"
    with jsonl.open("w") as f:
        for batch in batches:
            f.write(json.dumps(batch, sort_keys=True) + "\n")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = list(rows[0].keys()) if rows else []
        table = pa.table({name: [r.get(name) for r in rows] for name in fields})
        parquet_path = processed / f"{prefix}_records.parquet"
        pq.write_table(table, parquet_path, compression="zstd")
    except ImportError:
        parquet_path = None  # pyarrow optional for the contract outputs

    report["outputs"] = {
        "batches_jsonl": jsonl.name,
        "records_parquet": parquet_path.name if parquet_path else None,
        "report_json": f"{prefix}_report.json",
        "summary_md": f"{prefix}_summary.md",
    }
    (processed / f"{prefix}_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    (processed / f"{prefix}_summary.md").write_text(_summary_md(report))


def _summary_md(r: dict) -> str:
    lat = r["latency_ms"]

    def p5095(name: str) -> str:
        e = lat.get(name, {})
        return f"p50 {e.get('p50', '-')} ms / p95 {e.get('p95', '-')} ms"

    lines = [
        "# E2E union — measured results (detections → crops → species)",
        "",
        f"- stage: `{r['stage']}` — engine {r['params']['engine']}, "
        f"cv2 {r['versions']['cv2']}, wall {r['wall_s']} s over "
        f"{r['sample']['images']} images",
        f"- sample: {r['sample']['by_stratum']} — "
        f"{r['sample']['gt_boxes']} GT boxes, "
        f"{len(r['sample']['decode_failures'])} decode failures",
        f"- detections deployed (conf ≥ {r['params']['deploy_conf']}): "
        f"{r['detection']['deployed_detections']} "
        f"{r['detection']['per_label']}; "
        f"{r['detection']['zero_detection_images']} images with zero",
        "",
        "## Latency (warmup excluded)",
        "",
        f"- decode: {p5095('decode')}",
        f"- letterbox+blob: {p5095('pre')}",
        f"- detector forward: {p5095('forward')}",
        f"- decode+NMS: {p5095('post')}",
        f"- detector per image: {p5095('detector_per_image')}",
        f"- crop extract (per crop): {p5095('crop_extract')}",
        f"- classify (per crop): {p5095('classify')}",
        "",
        "## Detection recall vs GT (IoU ≥ 0.5, class-agnostic)",
        "",
        "| conf | scope | TP | FP | FN | recall | precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for conf, scopes in r["detection"]["recall_sweep"].items():
        for scope, m in scopes.items():
            lines.append(f"| {conf} | {scope} | {m['tp']} | {m['fp']} | {m['fn']} "
                         f"| {m['recall']} | {m['precision']} |")
    lines += [
        "",
        f"Matcher cross-check at deploy conf: own {r['detection']['recall_crosscheck_at_deploy']['own_matcher']}"
        f" vs parity module {r['detection']['recall_crosscheck_at_deploy']['parity_module']}"
        f" — {'MATCH' if r['detection']['recall_crosscheck_at_deploy']['match'] else 'MISMATCH'}",
        "",
        "## Classifier on GT-box crops (by split)",
        "",
        "| split | n | accuracy | note |",
        "|---|---|---|---|",
    ]
    for split, s in sorted(r["classification"]["gt_crops"]["by_split"].items()):
        note = f"v0 test ref {s['v0_test_ref']}" if "v0_test_ref" in s else \
            ("seen in training" if split == "train" else "")
        lines.append(f"| {split} | {s['n']} | {s['accuracy']} | {note} |")
    od = r["classification"]["on_detections"]
    lines += [
        "",
        "## Species on deployed detections",
        "",
        f"- evaluable (matched to species GT): {od['n_evaluable']}, "
        f"correct {od['correct']} → accuracy {od['accuracy']}",
        f"- {od['note']}",
        "",
        "## Species counts (deployed animal records)",
        "",
        "| species | n |",
        "|---|---|",
    ]
    for sp, n in r["classification"]["species_summary"]["counts"].items():
        lines.append(f"| {sp} | {n} |")
    lines += [
        "",
        "Detector-confidence histogram over deployed animal records: "
        + ", ".join(f"{b['bucket']}:{b['n']}"
                    for b in r["classification"]["species_summary"]["detector_conf_histogram"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["new", "classic", "auto"], default="new")
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--sizes", type=e2e.parse_sizes, default=None,
                        help="stratum=count pairs, e.g. aw_with_gt=100,mo_with_gt=75,mo_empty=25")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deploy-conf", type=float, default=DEPLOY_CONF)
    parser.add_argument("--out-prefix", default="e2e_union_v0")
    args = parser.parse_args(argv)

    report = run(engine=args.engine, sample_path=args.sample, sizes=args.sizes,
                 seed=args.seed, deploy_conf=args.deploy_conf,
                 out_prefix=args.out_prefix)
    d, c = report["detection"], report["classification"]
    rc = d["recall_sweep"][str(args.deploy_conf)]["overall"]
    test_split = c["gt_crops"]["by_split"].get("test", {})
    print(f"images={report['sample']['images']} wall={report['wall_s']}s "
          f"dets={d['deployed_detections']} zero_det={d['zero_detection_images']}")
    print(f"recall@{args.deploy_conf}={rc['recall']} (tp {rc['tp']} fn {rc['fn']} fp {rc['fp']}) "
          f"matcher-crosscheck={'OK' if d['recall_crosscheck_at_deploy']['match'] else 'FAIL'}")
    if test_split:
        print(f"gt-crop test-split accuracy={test_split['accuracy']} "
              f"(v0 ref {V0_TEST_ACCURACY})")
    print(f"species-on-detections accuracy={c['on_detections']['accuracy']} "
          f"over n={c['on_detections']['n_evaluable']}")
    print(f"-> {report['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
