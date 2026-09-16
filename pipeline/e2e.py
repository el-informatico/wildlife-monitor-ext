"""Glue logic for the E2E union stage: detections → crops → species.

This module holds the pure (numpy-only) pieces of the union stage so the
offline test suite covers them without model files, datasets, or cv2:

- deterministic sub-sampling of the S2 evaluation sample (seeded);
- crop-window math — the EXACT recipe classifier v0 was trained on
  (expand each box by 10% of its own size per side, round, clamp to the
  frame, drop windows smaller than 8 px, resize to 224);
- greedy detection↔GT matching returning the per-detection GT index
  (the S2 parity protocol matcher, but it reports the mapping instead
  of only counts, so matched detections can inherit GT species);
- african_wildlife YOLO label reading that keeps the species class id
  (detect.gt drops it — recall is class-agnostic there, but species
  evaluation needs it);
- detection-batch record assembly shaped for infra/schema.py (imported
  read-only; this repo must be able to validate what it emits against
  the ingest contract without touching infra/).

Heavy dependencies (cv2, the wildlife_monitor package, infra.schema,
pyarrow) are imported lazily inside the functions that need them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- stage parameters (mirrors of upstream constants, asserted at run) -----
CROP_MARGIN = 0.10        # fraction of box width/height added per side
CROP_MIN_EDGE_PX = 8      # skip windows smaller than this (training recipe)
CLASSIFY_INPUT = 224      # classifier v0 input (classify.model.INPUT_SIZE)
DEPLOY_CONF = 0.25        # detections below this are not deployed/uploaded
CONF_FLOOR = 0.10         # decode floor: stored so conf can sweep w/o re-run
MATCH_IOU = 0.5           # detection↔GT match threshold (S2 parity protocol)
SEED = 42                 # default sampling seed (config.SEED convention)

# Default sub-sample: exactly 1/4 of the S2 strata (400/300/100).
DEFAULT_SIZES = {"aw_with_gt": 100, "mo_with_gt": 75, "mo_empty": 25}

DET_LABELS = {0: "animal", 1: "person", 2: "vehicle"}  # mdv6.CLASS_NAMES


# ---------------------------------------------------------------------------
# Environment bootstrap (run path only — never needed by the offline tests)
# ---------------------------------------------------------------------------


def ext_root() -> Path:
    """This repo's root (pipeline/ lives directly under it)."""
    return Path(__file__).resolve().parents[1]


def main_repo_dir() -> Path:
    """Sprints-1-2 repo root (wildlife_monitor package).

    Override with $WILDLIFE_MAIN_DIR; the default mirrors that repo's
    config.ext_dir() sibling convention in the other direction.
    """
    import os

    env = os.environ.get("WILDLIFE_MAIN_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (ext_root().parent / "wildlife-monitor").resolve()


def import_schema() -> Any:
    """Import infra.schema (the ingest contract) read-only."""
    root = str(ext_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from infra import schema

    return schema


def import_wildlife(name: str) -> Any:
    """Import a wildlife_monitor submodule from the sprints-1-2 repo.

    Works both when the package is installed in the running venv (the
    usual case) and from a source checkout next to this repo.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        src = str(main_repo_dir() / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Deterministic sub-sampling
# ---------------------------------------------------------------------------


def draw_sample(images: list[dict], seed: int, sizes: dict[str, int]) -> list[dict]:
    """Deterministic per-stratum sub-sample, sorted by image_id.

    Same procedure as wildlife_monitor.detect.sample.build_sample: pool
    sorted by image_id, rng.choice without replacement, indices sorted so
    the result is independent of dict iteration order.
    """
    strata: dict[str, list[dict]] = {k: [] for k in sizes}
    for img in images:
        if img.get("stratum") in strata:
            strata[img["stratum"]].append(img)

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for name in sorted(strata):
        pool = sorted(strata[name], key=lambda e: e["image_id"])
        k = min(sizes[name], len(pool))
        if k < len(pool):
            idx = np.sort(rng.choice(len(pool), size=k, replace=False))
            pool = [pool[i] for i in idx]
        selected.extend(pool)
    return sorted(selected, key=lambda e: e["image_id"])


def parse_sizes(text: str) -> dict[str, int]:
    """Parse 'aw_with_gt=100,mo_with_gt=75,mo_empty=25' into a dict."""
    sizes: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        sizes[key.strip()] = int(value)
    if not sizes:
        raise ValueError(f"no stratum=count pairs in {text!r}")
    return sizes


# ---------------------------------------------------------------------------
# Crop extraction — the classifier v0 training recipe
# ---------------------------------------------------------------------------


def crop_window(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    margin: float = CROP_MARGIN,
    min_edge: int = CROP_MIN_EDGE_PX,
) -> tuple[int, int, int, int] | None:
    """xyxy box → integer crop window with `margin` added per side.

    Byte-for-byte the recipe of scripts/train_classifier_v0.py: each side
    grows by margin * box_width (resp. box_height), coordinates are
    rounded then clamped to the frame, and windows below `min_edge` on
    either axis are dropped (None) exactly like skipped training crops.
    """
    w, h = x2 - x1, y2 - y1
    cx1 = x1 - margin * w
    cy1 = y1 - margin * h
    cx2 = x2 + margin * w
    cy2 = y2 + margin * h
    ix1, iy1 = max(0, int(round(cx1))), max(0, int(round(cy1)))
    ix2, iy2 = min(img_w, int(round(cx2))), min(img_h, int(round(cy2)))
    if ix2 - ix1 < min_edge or iy2 - iy1 < min_edge:
        return None
    return ix1, iy1, ix2, iy2


def crop_window_xywh(
    box: "list[float] | tuple[float, float, float, float]",
    img_w: int,
    img_h: int,
    margin: float = CROP_MARGIN,
    min_edge: int = CROP_MIN_EDGE_PX,
) -> tuple[int, int, int, int] | None:
    """Absolute xywh GT box → crop window (same recipe as crop_window)."""
    x, y, w, h = (float(v) for v in box)
    ix1 = max(0, int(round(x - margin * w)))
    iy1 = max(0, int(round(y - margin * h)))
    ix2 = min(img_w, int(round(x + w + margin * w)))
    iy2 = min(img_h, int(round(y + h + margin * h)))
    if ix2 - ix1 < min_edge or iy2 - iy1 < min_edge:
        return None
    return ix1, iy1, ix2, iy2


def crop_window_yolo(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
    margin: float = CROP_MARGIN,
    min_edge: int = CROP_MIN_EDGE_PX,
) -> tuple[int, int, int, int] | None:
    """YOLO-normalized box → crop window, byte-identical to the trainer.

    Reproduces the exact center-form arithmetic of
    scripts/train_classifier_v0.py (expand by margin * box size per
    side, round, clamp, drop < min_edge) so GT crops evaluated here are
    pixel-identical to the crops classifier v0 was trained on — no
    float-association drift from converting to corner form first.
    """
    x1 = (xc * img_w - bw * img_w / 2) - margin * bw * img_w
    y1 = (yc * img_h - bh * img_h / 2) - margin * bh * img_h
    x2 = (xc * img_w + bw * img_w / 2) + margin * bw * img_w
    y2 = (yc * img_h + bh * img_h / 2) + margin * bh * img_h
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
    if ix2 - ix1 < min_edge or iy2 - iy1 < min_edge:
        return None
    return ix1, iy1, ix2, iy2


def extract_crop(bgr: np.ndarray, window: tuple[int, int, int, int], size: int = CLASSIFY_INPUT) -> np.ndarray:
    """Crop a window out of a BGR image and resize to size×size."""
    import cv2

    x1, y1, x2, y2 = window
    return cv2.resize(bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Ground truth with species (african_wildlife keeps the YOLO class id)
# ---------------------------------------------------------------------------


def aw_species_boxes(label_path: Path, img_w: int, img_h: int, classes: dict[int, str]) -> list[dict]:
    """YOLO label file → [{"bbox": xywh_abs, "category": "animal",
    "species": name, "yolo": (xc, yc, bw, bh)}] in file order (the same
    order detect.gt emits boxes, so indices are interchangeable)."""
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        aw, ah = bw * img_w, bh * img_h
        boxes.append(
            {
                "bbox": [xc * img_w - aw / 2, yc * img_h - ah / 2, aw, ah],
                "category": "animal",
                "species": classes.get(cls, str(cls)),
                "yolo": (xc, yc, bw, bh),
            }
        )
    return boxes


def aw_label_path(data_dir: Path, rel_path: str) -> Path:
    """african-wildlife/images/<split>/<name> → labels/<split>/<name>.txt."""
    img = data_dir / rel_path
    return img.parents[2] / "labels" / img.parent.name / img.with_suffix(".txt").name


# ---------------------------------------------------------------------------
# Detection ↔ GT matching (S2 parity protocol, mapping variant)
# ---------------------------------------------------------------------------


def match_dets_to_gt(
    gt_boxes_xywh: list[list[float]],
    det_boxes_xyxy: list[list[float]],
    det_scores: list[float],
    iou_threshold: float = MATCH_IOU,
) -> list[int]:
    """Greedy score-ordered matching; returns the GT index per detection.

    Same protocol as wildlife_monitor.detect.parity.match_image (IoU ≥
    threshold, class-agnostic, each GT box consumed at most once, greedy
    by detection score), but the per-detection mapping is returned so a
    matched detection can inherit the GT box's species. -1 = unmatched.
    """
    mdv6 = import_wildlife("wildlife_monitor.detect.mdv6")

    order = np.argsort(-np.asarray(det_scores, dtype=np.float64), kind="stable")
    gts = np.asarray(
        [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes_xywh],
        dtype=np.float64,
    ).reshape(-1, 4)
    dets = np.asarray(det_boxes_xyxy, dtype=np.float64).reshape(-1, 4)

    det_gt = [-1] * len(det_boxes_xyxy)
    matched_gt: set[int] = set()
    for di in order:
        gi = -1
        if len(gts):
            ious = mdv6.iou_matrix(dets[[int(di)]], gts)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and best not in matched_gt:
                matched_gt.add(best)
                gi = best
        det_gt[int(di)] = gi
    return det_gt


# ---------------------------------------------------------------------------
# Record assembly — infra/schema.py detection-batch shape
# ---------------------------------------------------------------------------


def detection_record(
    det: dict,
    img_w: int,
    img_h: int,
    species: str | None = None,
) -> dict:
    """One mdv6-style detection ({bbox xyxy px, conf, cls}) → schema record.

    Normalized bbox via infra.schema.bbox_px_to_norm (rounded to 4 dp for
    stable output); species is attached only to label == "animal".
    """
    label = DET_LABELS[int(det["cls"])]
    norm = import_schema().bbox_px_to_norm(det["bbox"], img_w, img_h)
    rec: dict[str, Any] = {
        "label": label,
        "confidence": round(float(det["conf"]), 4),
        "bbox": [round(v, 4) for v in norm],
    }
    if label == "animal" and species is not None:
        rec["species"] = species
    return rec


def build_batch(
    image_meta: dict,
    det_records: list[dict],
    key: str,
    camera_id: str | None = None,
    model_name: str = "MDV6-yolov10-c",
    runtime: str | None = None,
) -> dict:
    """Assemble one schema_version-1 detection batch for an image."""
    schema = import_schema()
    source: dict[str, Any] = {
        "key": key,
        "image_id": image_meta["image_id"],
        "width": int(image_meta["width"]),
        "height": int(image_meta["height"]),
        "dataset": image_meta.get("source_dataset"),
    }
    if camera_id:
        source["camera_id"] = camera_id
    if image_meta.get("is_night") is not None:
        source["is_night"] = bool(image_meta["is_night"])
    model: dict[str, Any] = {"name": model_name}
    if runtime:
        model["runtime"] = runtime
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "source": source,
        "model": model,
        "detections": det_records,
    }


def validate_batch(doc: dict) -> int:
    """Validate one assembled batch against the ingest contract.

    Returns the number of valid detections; raises SchemaError otherwise
    (structural problems) — the same poison/skip policy split as the
    handler, but the producer side validates before emitting.
    """
    schema = import_schema()
    batch = schema.validate_batch(doc)
    n = 0
    for i, raw in enumerate(batch.detections):
        schema.validate_detection(raw, where=f"detections[{i}]")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def confidence_histogram(confidences: list[float], lo: float, hi: float, step: float = 0.1) -> list[dict]:
    """Bucketed confidence counts in [lo, hi] (upper-inclusive last bucket).

    The last edge is clamped to `hi`, so a lo/hi span that is not a whole
    multiple of `step` (e.g. 0.25..1.0 by 0.1) still ends exactly at hi.
    """
    n = int(round((hi - lo) / step))
    edges = [round(min(lo + i * step, hi), 2) for i in range(n + 1)]
    counts = [0] * (len(edges) - 1)
    for c in confidences:
        for i in range(len(counts)):
            upper_inclusive = i == len(counts) - 1
            if edges[i] <= c < edges[i + 1] or (upper_inclusive and c == edges[i + 1]):
                counts[i] += 1
                break
    return [
        {"bucket": f"[{edges[i]:.2f},{edges[i + 1]:.2f}]", "n": n}
        for i, n in enumerate(counts)
    ]


def summarize_species(det_records_per_image: list[list[dict]]) -> dict:
    """Per-species counts + confidence distribution over animal records."""
    species_counts: dict[str, int] = {}
    animal_confs: list[float] = []
    for records in det_records_per_image:
        for rec in records:
            if rec["label"] != "animal":
                continue
            name = rec.get("species", "unclassified")
            species_counts[name] = species_counts.get(name, 0) + 1
            animal_confs.append(rec["confidence"])
    out: dict[str, Any] = {
        "counts": dict(sorted(species_counts.items())),
        "detector_conf_histogram": confidence_histogram(animal_confs, DEPLOY_CONF, 1.0),
    }
    return out


def load_manifest_join(manifest_path: Path) -> dict[str, dict]:
    """image_id → {location_id, camera_ts, species} from the ETL manifest."""
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path, columns=["image_id", "location_id", "camera_ts", "species"])
    join: dict[str, dict] = {}
    ids = table.column("image_id").to_pylist()
    locs = table.column("location_id").to_pylist()
    ts = table.column("camera_ts").to_pylist()
    sp = table.column("species").to_pylist()
    for i, image_id in enumerate(ids):
        join[image_id] = {"location_id": locs[i], "camera_ts": ts[i], "species": sp[i]}
    return join
