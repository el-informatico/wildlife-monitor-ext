"""Runtime helpers for the G3 evaluation runner.

Everything that needs cv2, the models, or the datasets lives here (or in
run_eval) so eval.metrics stays numpy-only and offline-testable. The
module reuses the e2e stage's environment bootstrap (repo roots, the
sprints-1-2 wildlife_monitor package) and its GT/crop helpers verbatim —
the G3 numbers must be computed with the same recipes the union stage
and the trainer used, not near-copies of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import e2e
from . import metrics

WORK_RESOLUTION = 1024  # motion.bench.MAX_LONG_SIDE — working decode cap


def load_manifest() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = e2e.ext_root() / "processed" / "manifest.parquet"
    return pq.read_table(path).to_pylist()


def decode_capped(path: Path, max_long_side: int = WORK_RESOLUTION) -> np.ndarray | None:
    """Decode BGR, long side capped (the motion-stage working resolution)."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = max_long_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return bgr


def mean_saturation(bgr: np.ndarray) -> float:
    """Mean HSV saturation — the IR (night) flag's input signal."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean())


class IRFlagCache:
    """image_id → content-based IR flag, persisted under processed/.

    The flag is a pure function of the image pixels, so it is computed
    once and reused across stages and re-runs (the persisted cache also
    records the threshold — a threshold change invalidates it).
    """

    def __init__(self, threshold: float = metrics.IR_SAT_THRESHOLD):
        self.threshold = threshold
        self.path = e2e.ext_root() / "processed" / "g3_ir_flags.json"
        self.flags: dict[str, bool] = {}
        self.sats: dict[str, float] = {}
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            if blob.get("threshold") == threshold:
                self.flags = {k: bool(v) for k, v in blob.get("flags", {}).items()}
                self.sats = blob.get("mean_sat", {})

    def of(self, image_id: str, bgr: np.ndarray) -> bool:
        """Flag from an already-decoded working-resolution frame (the
        motion pass decodes each frame once and feeds both the
        saturation measurement and MOG2)."""
        if image_id not in self.flags:
            sat = round(mean_saturation(bgr), 3)
            self.sats[image_id] = sat
            self.flags[image_id] = metrics.ir_flag(sat, self.threshold)
        return self.flags[image_id]

    def get(self, image_id: str, rel_path: Path | str) -> bool | None:
        """Flag for one image; decodes and caches on first touch."""
        if image_id not in self.flags:
            bgr = decode_capped(rel_path)
            if bgr is None:
                return None
            return self.of(image_id, bgr)
        return self.flags[image_id]

    def flush(self) -> None:
        self.path.write_text(json.dumps(
            {"threshold": self.threshold, "flags": self.flags, "mean_sat": self.sats},
            indent=1, sort_keys=True) + "\n")


def detect_image(detector, bgr: np.ndarray, conf_floor: float) -> list[dict]:
    """MDv6 over one BGR frame — the e2e stage's inline recipe, verbatim
    (letterbox square, swapRB=False BGR blob, decode at the floor)."""
    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    h, w = bgr.shape[:2]
    params = mdv6.letterbox_params(w, h, detector.input_size)
    padded = mdv6.letterbox_image(bgr, params, detector.input_size)
    blob = mdv6.to_blob(padded)
    detector.net.setInput(blob)
    output = np.asarray(detector.net.forward())
    return mdv6.decode_prediction(output, params, conf_floor,
                                  mdv6.IOU_THRESHOLD, mdv6.MAX_DETECTIONS).to_json()


def load_models(engine: str = "new"):
    """(detector, classifier) — the two ONNX artifacts, cv2.dnn engines."""
    ext = e2e.ext_root()
    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    clf = e2e.import_wildlife("wildlife_monitor.classify.model")
    detector = mdv6.MDv6OpenCV(ext / "models" / "MDV6-yolov10-c.onnx", engine=engine)
    classifier = clf.ClassifierOpenCV(ext / "models" / "classifier_v0_resnet18.onnx",
                                      engine=engine)
    return detector, classifier


def missouri_gt():
    """Missouri bbox GT (xywh absolute) — config.manifests_dir() resolves
    to this repo's manifests/, so the default constructor is correct."""
    gt = e2e.import_wildlife("wildlife_monitor.detect.gt")
    return gt.MissouriGT()


def gt_loader(data_dir: Path, classes: dict[int, str], mo_gt=None):
    """Image dict (manifest or sample row) → GT boxes, xywh absolute, with
    YOLO species on african_wildlife and species None on missouri (its GT
    collapses to 'animal'). Same loading path the e2e stage used."""
    def load(img: dict) -> list[dict]:
        if img["source_dataset"] == "african_wildlife":
            return e2e.aw_species_boxes(
                e2e.aw_label_path(data_dir, img["rel_path"]),
                img["width"], img["height"], classes,
            )
        boxes = mo_gt.boxes_for(img["rel_path"]) if mo_gt else []
        return [{"bbox": b["bbox"], "category": b["category"], "species": None}
                for b in boxes]
    return load


def matched_flags(gt_boxes: list[dict], dets: list[dict], conf: float) -> list[bool]:
    """Per-deployment-detection GT-match flags at `conf` (pure matcher)."""
    deployed = [d for d in dets if d["conf"] >= conf]
    mapping = metrics.greedy_match(
        [b["bbox"] for b in gt_boxes],
        [d["bbox"] for d in deployed],
        [d["conf"] for d in deployed],
        metrics.MATCH_IOU,
    )
    return [m >= 0 for m in mapping]


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(prefix: str, report: dict, csv_text: str, md_text: str) -> dict[str, str]:
    """Write report/metrics under processed/; returns the file names."""
    processed = e2e.ext_root() / "processed"
    processed.mkdir(exist_ok=True)
    names = {
        "report_json": f"{prefix}_report.json",
        "metrics_csv": f"{prefix}_metrics.csv",
        "metrics_md": f"{prefix}_metrics.md",
    }
    (processed / names["report_json"]).write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    (processed / names["metrics_csv"]).write_text(csv_text)
    (processed / names["metrics_md"]).write_text(md_text)
    return names
