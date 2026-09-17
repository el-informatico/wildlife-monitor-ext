"""Run the person precision/recall held-out evaluation.

Why this runner exists — the Wellington gap is DEEPER than availability:

- Wellington Camera Traps' **public labels carry no person class at all**:
  17 categories = 15 animals + ``empty`` + ``unclassifiable`` (enumerated
  from the published ``wellington_camera_traps.json``; annotations are
  image-level with no boxes). Person P/R is uncomputable from Wellington
  *even with every image on disk* — the S5 probe's "dataset absent"
  understated the gap.
- Person-labelled **camera-trap** GT is structurally scarce: the LILA sets
  that once had human classes removed those images for privacy. Measured on
  SWG 2018-2020's no-species box file: 781 person-annotated images, **all
  781 under ``private/``, 0 public** (the 6 vehicle-boxed images are all
  public). Idaho/WCS/ENA24 document the same removal.

Held-outs actually used (both CDLA-Permissive-1.0, images local under
``downloads/`` — never committed; only metrics and this script are):

1. **Seattle(ish) Camera Traps** (lila.science) — camera-trap DOMAIN,
   class-level GT. ``dan`` is the dataset's one human (2,484 of 24,756
   annotated entries; 2,217 stills). Deterministic sample, seed 17
   (``eval/sample_seattle.py``), still images only — the metadata's
   video entries (``.avi``/``.mp4``) are excluded because a video's
   class label does not guarantee the subject is visible in any single
   frame. Strata: 800 ``dan`` + all 60 still ``vehicle`` + 500
   ``empty`` + 380 animal-only negatives = 1,740.
   Person/vehicle P/R measured at **image level** (image flagged if any
   detection of the class ≥ threshold).
   Honest limits: single human identity (the dataset's author; images
   with other people are removed by design), one backyard, GT has no
   boxes — precision here bounds the camera-trap person story, it does
   not generalize to arbitrary people.
2. **COCO 2017 val** (5,000 images) — **box-level** person + vehicle GT,
   general domain (consumer photos, not camera traps). Person dets vs
   person boxes; vehicle dets vs car/bus/truck/motorcycle boxes; greedy
   IoU ≥ 0.5 matching, the G3 parity protocol (``eval.metrics.greedy_match``).
   Honest limit: domain shift — this is the multi-identity, multi-scene
   ceiling, not the deployment domain.

Detector: the deployed ``MDV6-yolov10-c.onnx`` through cv2.dnn (OpenCV 5
graph engine), identical letterbox/decode path to the G3/S5 stages; every
detection ≥ 0.10 is stored so thresholds are swept without re-inference.
``DEPLOY_CONF`` (0.25) from ``pipeline.e2e`` stays the headline number.

Outputs: ``processed/person_pr_report.json`` (provenance + per-threshold
tables + per-image predictions) and ``processed/person_pr_tables.md``
(the fragment eval/PERSON-PR-REPORT.md embeds verbatim).

    ../wildlife-monitor/.venv/bin/python -m eval.run_person_pr [--max-per-set N]

``--max-per-set`` caps images per held-out (first-N of the deterministic
sample) for a quick smoke pass; headline numbers come from the full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import metrics
from pipeline import e2e  # noqa: E402  (deploy conf + label map, as run_final)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "processed"
DOWNLOADS = REPO_ROOT / "downloads"

DEPLOY_CONF = e2e.DEPLOY_CONF          # 0.25 — the deployed threshold
DET_LABELS = e2e.DET_LABELS            # {0: animal, 1: person, 2: vehicle}
PERSON_CLS = [cls for cls, name in DET_LABELS.items() if name == "person"][0]
VEHICLE_CLS = [cls for cls, name in DET_LABELS.items() if name == "vehicle"][0]

CONF_FLOOR = 0.10                      # storage floor (identical to S2)
SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MATCH_IOU = metrics.MATCH_IOU          # 0.5 — parity protocol

#: COCO categories folded into MDv6's ``vehicle`` class (ids from
#: instances_val2017.json). Anything airborne/waterborne is excluded —
#: MDv6's vehicle training vocabulary is ground vehicles.
COCO_VEHICLE_CATS = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
COCO_PERSON_CAT = 1


# ---------------------------------------------------------------------------
# Offline-testable glue (no cv2 / no images — eval/tests covers these)
#

def flag_image(dets: list[dict], cls: int, conf: float) -> bool:
    """Image-level flag: any detection of ``cls`` with conf ≥ threshold."""
    return any(d["cls"] == cls and d["conf"] >= conf for d in dets)


def image_level_pr(rows: list[dict], cls: int) -> dict[str, Any]:
    """Image-level precision/recall over ``rows`` = {gt: bool, dets: [...]}.

    ``gt`` is "this image contains the class" (Seattle class-level labels),
    predictions are ``flag_image`` at each sweep threshold. Images whose GT
    is positive for the OTHER class are kept in the denominator — a person
    detection on a vehicle-only image is an honest person FP.
    """
    out: dict[str, Any] = {}
    for t in SWEEP:
        tp = fp = fn = 0
        fp_by_stratum: dict[str, int] = defaultdict(int)
        for r in rows:
            flagged = flag_image(r["dets"], cls, t)
            if r["gt"] and flagged:
                tp += 1
            elif (not r["gt"]) and flagged:
                fp += 1
                fp_by_stratum[r["stratum"]] += 1
            elif r["gt"] and not flagged:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        out[f"{t:.2f}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision and recall else None,
            "fp_by_stratum": dict(sorted(fp_by_stratum.items())),
        }
    return out


def box_level_pr(rows: list[dict], cls: int) -> dict[str, Any]:
    """Box-level precision/recall over ``rows`` = {gt: [xywh], dets: [...]}.

    Class-restricted greedy matching at ``MATCH_IOU`` using the G3 parity
    matcher (score-ordered, each GT consumed once; GT in COCO-native xywh,
    detections in the detector's xyxy — ``greedy_match`` converts). A
    detection of the wrong class can never match — per-class evaluation
    pre-filters both sides, so cross-class confusions show up as FP on one
    side and FN on the other, never as a false match.
    """
    out: dict[str, Any] = {}
    for t in SWEEP:
        tp = fp = fn = 0
        fp_on: dict[str, int] = defaultdict(int)
        for r in rows:
            dets = [d for d in r["dets"] if d["cls"] == cls and d["conf"] >= t]
            dets.sort(key=lambda d: -d["conf"])
            gt = r["gt"]
            matched = metrics.greedy_match(
                gt, [d["bbox"] for d in dets], [d["conf"] for d in dets],
                iou_threshold=MATCH_IOU)
            m = {gi for gi in matched if gi >= 0}
            tp += len(m)
            fn += len(gt) - len(m)
            for d, gi in zip(dets, matched):
                if gi < 0:
                    fp += 1
                    fp_on[r["name"]] += 1
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        out[f"{t:.2f}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision and recall else None,
            "n_gt_boxes": sum(len(r["gt"]) for r in rows),
            # FPs by image GT group — the key is the caller-supplied
            # ``name`` per row (e.g. on-GT vs off-GT images split
            # localization errors from class confusions)
            "fp_on": dict(sorted(fp_on.items())),
        }
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Held-out ingest
#

def load_seattle(max_images: int | None = None) -> list[dict]:
    """The seed-17 sample spec → rows with GT from the class-level labels."""
    root = DOWNLOADS / "seattleish"
    spec = json.loads((root / "sample_spec.json").read_text())
    labels = spec["labels"]
    rows = []
    for i, rel in enumerate(spec["files"]):
        if max_images and i >= max_images:
            break
        if Path(rel).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError(
                f"video entry in still-image sample: {rel} — regenerate "
                f"the spec with eval/sample_seattle.py")
        lab = labels[rel]
        path = root / "images" / rel
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"sample image missing: {rel}")
        rows.append({
            "image_id": rel,
            "stratum": ("dan" if "dan" in lab else
                        "vehicle" if "vehicle" in lab else
                        "empty" if "empty" in lab else "animal"),
            "gt_person": "dan" in lab,
            "gt_vehicle": "vehicle" in lab,
            "path": path,
        })
    return rows


def load_coco(max_images: int | None = None) -> list[dict]:
    """COCO val2017 → rows with xyxy GT boxes for person + vehicle."""
    root = DOWNLOADS / "coco"
    inst = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    by_image: dict[int, dict[str, list[list[float]]]] = defaultdict(
        lambda: {"person": [], "vehicle": []})
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        box = a["bbox"]  # COCO-native xywh — greedy_match converts
        if a["category_id"] == COCO_PERSON_CAT:
            by_image[a["image_id"]]["person"].append(box)
        elif a["category_id"] in COCO_VEHICLE_CATS:
            by_image[a["image_id"]]["vehicle"].append(box)
    images = sorted(inst["images"], key=lambda i: i["id"])
    if max_images:
        images = images[:max_images]
    rows = []
    for i in images:
        path = root / "val2017" / i["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"val image missing: {i['file_name']}")
        gt = by_image.get(i["id"], {"person": [], "vehicle": []})
        rows.append({
            "image_id": i["file_name"],
            "gt_person": bool(gt["person"]), "gt_vehicle": bool(gt["vehicle"]),
            "boxes_person": gt["person"], "boxes_vehicle": gt["vehicle"],
            "path": path,
        })
    return rows


# ---------------------------------------------------------------------------
# Detection (the deployed path, verbatim from the S2 runner)
#

def detect_all(rows: list[dict], onnx_path: Path) -> tuple[dict[str, list[dict]], dict]:
    import cv2

    from wildlife_monitor.detect.mdv6 import (
        MDv6OpenCV, letterbox_params, letterbox_image, to_blob,
        decode_prediction, IOU_THRESHOLD, MAX_DETECTIONS,
    )

    detector = MDv6OpenCV(onnx_path, engine="new")
    detections: dict[str, list[dict]] = {}
    ms: list[float] = []
    t_start = time.perf_counter()
    for i, r in enumerate(rows):
        bgr = cv2.imread(str(r["path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image: {r['path']}")
        h, w = bgr.shape[:2]
        params = letterbox_params(w, h, detector.input_size)
        padded = letterbox_image(bgr, params, detector.input_size)
        blob = to_blob(padded)
        t0 = time.perf_counter()
        detector.net.setInput(blob)
        output = __import__("numpy").asarray(detector.net.forward())
        dets = decode_prediction(output, params, CONF_FLOOR,
                                 IOU_THRESHOLD, MAX_DETECTIONS)
        ms.append((time.perf_counter() - t0) * 1000)
        detections[r["image_id"]] = dets.to_json()
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(rows)} images "
                  f"(fwd p50 {sorted(ms)[len(ms) // 2]:.0f} ms)", flush=True)
    meta = {
        "model": {"file": onnx_path.name, "sha256": _sha256(onnx_path)},
        "runtime": f"cv2.dnn (OpenCV {cv2.__version__}, engine new)",
        "conf_floor": CONF_FLOOR,
        "iou_threshold": IOU_THRESHOLD,
        "max_det": MAX_DETECTIONS,
        "n_images": len(rows),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "forward_ms_p50": round(sorted(ms)[len(ms) // 2], 1),
    }
    return detections, meta


# ---------------------------------------------------------------------------
# Report
#

def md_tables(seattle_pr: dict, coco_pr: dict, seattle_meta: dict,
              coco_meta: dict) -> str:
    def table(metric: dict, note: str) -> list[str]:
        lines = [f"| conf | TP | FP | FN | precision | recall | F1 |",
                 f"|---|---|---|---|---|---|---|"]
        for t in SWEEP:
            r = metric[f"{t:.2f}"]
            star = " **← deploy**" if t == DEPLOY_CONF else ""
            p = r["precision"] if r["precision"] is not None else "—"
            rc = r["recall"] if r["recall"] is not None else "—"
            f1 = r["f1"] if r["f1"] is not None else "—"
            lines.append(f"| {t:.2f}{star} | {r['tp']} | {r['fp']} | {r['fn']} "
                         f"| {p} | {rc} | {f1} |")
        lines.append(f"\n{note}")
        return lines

    out = ["## Person/vehicle P/R — held-out tables (generated; embed verbatim)", ""]
    out += [f"### Seattle(ish) — camera-trap domain, image level "
            f"({seattle_meta['n_images']} images, seed-17 sample)", ""]
    out += ["**Person**", ""]
    out += table(seattle_pr["person"],
                 "FP-by-stratum at each threshold in the JSON; single-identity "
                 "(“dan”) limit applies.")
    out += ["", "**Vehicle**", ""]
    out += table(seattle_pr["vehicle"],
                 "TP = 0 at every threshold incl. the 0.10 storage floor — "
                 "no vehicle detection fired on any vehicle-GT still (see "
                 "Reading).")
    out += ["", f"### COCO 2017 val — box level, greedy IoU ≥ 0.5 "
            f"({coco_meta['n_images']} images)", ""]
    out += ["**Person**", ""]
    out += table(coco_pr["person"],
                 "GT = person boxes, iscrowd excluded.")
    out += ["", "**Vehicle**", ""]
    out += table(coco_pr["vehicle"],
                 "Vehicle-mapped GT: car/motorcycle/bus/truck; iscrowd excluded.")
    out += [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-set", type=int, default=None,
                        help="cap images per held-out (smoke pass)")
    parser.add_argument("--remake-tables", action="store_true",
                        help="rebuild processed/person_pr_tables.md from the "
                             "saved report JSON — no inference, no model")
    args = parser.parse_args(argv)

    if args.remake_tables:
        saved = json.loads((PROCESSED / "person_pr_report.json").read_text())
        tables = PROCESSED / "person_pr_tables.md"
        tables.write_text(md_tables(
            saved["seattle_image_level"], saved["coco_box_level"],
            saved["heldouts"]["seattleish"]["detection"],
            saved["heldouts"]["coco_val2017"]["detection"]) + "\n")
        print(f"-> {tables} (remade from saved report, no inference)")
        return 0

    onnx_path = REPO_ROOT / "models" / "MDV6-yolov10-c.onnx"
    if not onnx_path.is_file():
        print(f"detector missing: {onnx_path}", file=sys.stderr)
        return 2

    print("[seattle] loading sample…", flush=True)
    seattle_rows = load_seattle(args.max_per_set)
    print(f"[seattle] {len(seattle_rows)} images "
          f"({sum(r['gt_person'] for r in seattle_rows)} person GT)", flush=True)
    seattle_dets, seattle_det_meta = detect_all(seattle_rows, onnx_path)

    print("[coco] loading val2017…", flush=True)
    coco_rows = load_coco(args.max_per_set)
    n_person = sum(r["gt_person"] for r in coco_rows)
    print(f"[coco] {len(coco_rows)} images ({n_person} person GT)", flush=True)
    coco_dets, coco_det_meta = detect_all(coco_rows, onnx_path)

    # image-level (Seattle) — GT from class labels; dets attached per row
    seattle_rows = [{**r, "dets": seattle_dets[r["image_id"]]}
                    for r in seattle_rows]
    seattle_pr = {
        "person": image_level_pr(
            [{"gt": r["gt_person"], "stratum": r["stratum"],
              "dets": r["dets"]} for r in seattle_rows], PERSON_CLS),
        "vehicle": image_level_pr(
            [{"gt": r["gt_vehicle"], "stratum": r["stratum"],
              "dets": r["dets"]} for r in seattle_rows], VEHICLE_CLS),
    }

    # box-level (COCO)
    coco_rows = [{**r, "dets": coco_dets[r["image_id"]]} for r in coco_rows]
    coco_pr = {
        "person": box_level_pr(
            [{"gt": r["boxes_person"], "dets": r["dets"],
              "name": "person_gt" if r["gt_person"] else "no_person_gt"}
             for r in coco_rows], PERSON_CLS),
        "vehicle": box_level_pr(
            [{"gt": r["boxes_vehicle"], "dets": r["dets"],
              "name": "vehicle_gt" if r["gt_vehicle"] else "no_vehicle_gt"}
             for r in coco_rows], VEHICLE_CLS),
    }

    spec = json.loads((DOWNLOADS / "seattleish" / "sample_spec.json").read_text())
    report = {
        "what": "person/vehicle P/R on person-labelled held-outs",
        "wellington_finding": (
            "public labels = 15 animal categories + empty + unclassifiable; "
            "NO person/vehicle class, image-level only — person P/R "
            "uncomputable from Wellington even with images on disk"),
        "heldouts": {
            "seattleish": {
                "source": spec["source"], "seed": spec["seed"],
                "strata": spec["strata"], "detection": seattle_det_meta,
                "limit": "single human identity (dan = the dataset's "
                         "author; images with other people are removed "
                         "by design), one yard, class-level GT",
            },
            "coco_val2017": {
                "source": "COCO 2017 val (images.cocodataset.org, CC-BY 4.0 "
                          "annotations; images under their Flickr licenses)",
                "vehicle_gt_categories": COCO_VEHICLE_CATS,
                "detection": coco_det_meta,
                "limit": "general domain, not camera traps",
            },
        },
        "deploy_conf": DEPLOY_CONF,
        "match_iou": MATCH_IOU,
        "seattle_image_level": seattle_pr,
        "coco_box_level": coco_pr,
        "predictions": {"seattleish": seattle_dets, "coco_val2017": coco_dets},
    }
    PROCESSED.mkdir(exist_ok=True)
    out = PROCESSED / "person_pr_report.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    tables = PROCESSED / "person_pr_tables.md"
    tables.write_text(md_tables(seattle_pr, coco_pr,
                                seattle_det_meta, coco_det_meta) + "\n")

    for name, pr in (("Seattle person (image)", seattle_pr["person"]),
                     ("COCO person (box)", coco_pr["person"])):
        d = pr[f"{DEPLOY_CONF:.2f}"]
        print(f"[result] {name} @{DEPLOY_CONF}: P={d['precision']} "
              f"R={d['recall']} (TP {d['tp']} / FP {d['fp']} / FN {d['fn']})")
    print(f"-> {out}\n-> {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
