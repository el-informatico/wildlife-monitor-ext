"""The heavy side of the sprint: load processed artifacts, assemble corpus
events, and run the live pass-2 re-analysis with real cv2.dnn pixels.

Everything here sits behind lazy imports (pyarrow, cv2, numpy, the
wildlife_monitor package) so the agent's offline unit tests never need
them — :mod:`agent.scenario` stays the pure, stdlib-only assembly layer.

Paths follow the repo convention (pipeline/run_e2e.py resolves the same
roots): ``processed/`` for the S2/G3 artifacts, ``data/`` for pixels,
``models/`` for ONNX weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import policy
from .scenario import attach_tz, build_event, zones_from_manifest_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "processed"
UNION_BATCHES = PROCESSED_DIR / "e2e_union_v0_batches.jsonl"
MANIFEST_PARQUET = PROCESSED_DIR / "manifest.parquet"
IR_FLAGS = PROCESSED_DIR / "g3_ir_flags.json"
DATA_DIR = REPO_ROOT / "data"
DET_ONNX = REPO_ROOT / "models" / "MDV6-yolov10-c.onnx"

#: Detector engine string — mirrors pipeline/run_e2e.run's default so
#: pass-1 (corpus) and pass-2 (re-analysis) pixels run the same backend.
DET_ENGINE = "new"

MANIFEST_COLUMNS = ["image_id", "source_dataset", "rel_path", "location_id",
                    "camera_ts", "zone_restricted"]


# ---------------------------------------------------------------------------
# Corpus loaders
#


def load_union_batches(path: Path = UNION_BATCHES) -> list[dict[str, Any]]:
    """The 200 S2/E2E union batches (schema_version 1, one JSON object per
    line — detections already in the contract's normalized shape)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_ir_flags(path: Path = IR_FLAGS) -> dict[str, bool]:
    """G3's content-IR night flags, keyed by image_id (authoritative night)."""
    return {k: bool(v) for k, v in
            json.loads(path.read_text())["flags"].items()}


def load_manifest_rows(path: Path = MANIFEST_PARQUET,
                       columns: list[str] = MANIFEST_COLUMNS) -> list[dict[str, Any]]:
    """Manifest rows as plain dicts (lazy pyarrow — the only heavy import
    on the corpus path)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    names = table.column_names
    return [dict(zip(names, vals)) for vals in zip(
        *(table.column(n).to_pylist() for n in names))]


def corpus_events_and_zones(
        batches: list[dict[str, Any]] | None = None,
        manifest_rows: list[dict[str, Any]] | None = None,
        ir_flags: dict[str, bool] | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Join batches × manifest × IR flags into agent events + zone policy.

    event identity: the image_id IS the event_id for pass-1 corpus events
    (unique, traceable straight back to pixels). Batch rows carry no
    timestamp or zone — both come from the manifest via image_id, and a
    batch image missing from the manifest raises rather than guessing.
    """
    batches = batches if batches is not None else load_union_batches()
    manifest_rows = (manifest_rows if manifest_rows is not None
                     else load_manifest_rows())
    ir_flags = ir_flags if ir_flags is not None else load_ir_flags()

    by_image = {row["image_id"]: row for row in manifest_rows}
    events = []
    for batch in batches:
        src = batch["source"]
        iid = src["image_id"]
        try:
            man = by_image[iid]
        except KeyError:
            raise ValueError(
                f"batch image {iid!r} absent from manifest — cannot join "
                f"camera_ts/zone") from None
        events.append(build_event(
            event_id=iid,
            camera_id=src["camera_id"],
            dataset=src["dataset"],
            source_key=src["key"],
            captured_at=attach_tz(man["camera_ts"], src["dataset"]),
            clock_source=policy.CLOCK_MANIFEST_SYNTHETIC,
            detections=batch["detections"],
            night_content_ir=ir_flags.get(iid, False),
            night_synthetic_flag=bool(src["is_night"]),
            model=batch["model"],
        ))
    zones = zones_from_manifest_rows(
        {"camera_id": row["location_id"], "zone_restricted": row["zone_restricted"]}
        for row in manifest_rows)
    return events, zones


# ---------------------------------------------------------------------------
# Live pass-2: the night re-analysis (demo path — real cv2.dnn pixels)
#


def _motion_parity_modules():
    """cv2 + the mdv6 glue, via pipeline.e2e's import conventions."""
    import cv2  # noqa: F401  (returned; the caller runs the forward pass)

    from pipeline import e2e

    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    return cv2, e2e, mdv6


def load_detector():
    """MDv6OpenCV over the repo's ONNX — the same weights the S2/E2E pass
    ran, so pass-2 differs only in preprocessing and decode floor.
    Returns ``(detector, cv2, e2e, mdv6)`` — everything
    :func:`night_reanalysis_detections` needs, unpack it in one line."""
    cv2, e2e, mdv6 = _motion_parity_modules()
    detector = mdv6.MDv6OpenCV(DET_ONNX, engine=DET_ENGINE)
    return detector, cv2, e2e, mdv6


def night_reanalysis_detections(bgr, detector, cv2, e2e, mdv6) -> list[dict[str, Any]]:
    """Second MDv6 pass on one BGR frame, motion-gate recipe.

    grayscale → CLAHE(clip=3.0, tile=(8,8)) — exactly what
    infra/handler/motion_app.py does to content-IR frames — replicated to
    3 channels, then the SAME letterbox/blob/forward chain as pass 1
    (pipeline/run_e2e.py), decoded at :data:`policy.REANALYSIS_CONF_FLOOR`
    instead of the deploy floor. Returns contract-shaped detections
    (normalized bbox, 4-dp confidence) via pipeline.e2e.detection_record —
    identical post-processing to pass 1, so any output difference is
    attributable to preprocessing + floor, nothing else.
    """
    import numpy as np

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    pre = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    params = mdv6.letterbox_params(w, h, detector.input_size)
    padded = mdv6.letterbox_image(pre, params, detector.input_size)
    blob = mdv6.to_blob(padded)  # swapRB=False — pinned BGR semantics
    detector.net.setInput(blob)
    output = np.asarray(detector.net.forward())
    dets = mdv6.decode_prediction(output, params, policy.REANALYSIS_CONF_FLOOR,
                                  mdv6.IOU_THRESHOLD, mdv6.MAX_DETECTIONS)
    return [e2e.detection_record(det, w, h) for det in dets.to_json()]


def load_image_bgr(rel_path: str):
    """One corpus frame from data/ (manifest rel_path, not the S3 key)."""
    cv2, _, _ = _motion_parity_modules()
    bgr = cv2.imread(str(DATA_DIR / rel_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"image not decodable: {DATA_DIR / rel_path}")
    return bgr


def build_pass2_event(pass1_event: dict[str, Any], detections: list[dict[str, Any]],
                      rel_path: str = "") -> dict[str, Any]:
    """The re-analysis output re-enters the loop as a pass-2 event.

    Same camera/clock/zone context (the question is the pixels, not the
    context); ``reanalysis_of`` chains back to the pass-1 event and the
    model record carries the re-parameterization that produced it.
    """
    return build_event(
        event_id=f"{pass1_event['event_id']}#p2",
        camera_id=pass1_event["camera_id"],
        dataset=pass1_event["dataset"],
        source_key=pass1_event["source_key"],
        captured_at=pass1_event["captured_at"],
        clock_source=pass1_event["clock_source"],
        detections=detections,
        night_content_ir=pass1_event["night"]["content_ir"],
        night_synthetic_flag=pass1_event["night"]["synthetic_flag"],
        model={"runtime": pass1_event["model"]["runtime"],
               "preprocess": policy.REANALYSIS_PREPROCESS,
               "conf_floor": policy.REANALYSIS_CONF_FLOOR,
               "pass": 2},
        pass_=2,
        reanalysis_of=pass1_event["event_id"],
    )
