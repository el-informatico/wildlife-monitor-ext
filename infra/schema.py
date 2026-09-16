"""Detection-payload contract for the ingestion path (schema_version 1).

The OpenCV 5 inference node analyzes a media file and uploads one JSON
"detection batch" to S3 under ``detections/<camera>/<...>.json``. The S3
event triggers the ingest Lambda, which validates the object against this
module and writes normalized records to DynamoDB. This module is the single
source of truth for that contract; it is stdlib-only so it can be vendored
into the Lambda zip as-is and imported by the pipeline side too.

Payload shape::

    {
      "schema_version": 1,
      "source": {
        "key":         "raw/cam-01/2026-09-16/a001.jpg",  # S3 key of the media (required)
        "image_id":    "aw-00007",                        # pipeline image id (required)
        "dataset":     "african_wildlife",                # optional
        "width":       550,                               # px (required, > 0)
        "height":      412,                               # px (required, > 0)
        "is_night":    false,                             # optional
        "captured_at": "2026-09-16T10:15:00Z",            # optional ISO-8601 with tz
        "camera_id":   "cam-01"                           # optional; derived from key otherwise
      },
      "model": {
        "name":    "MDV6-yolov10-c",                      # required
        "runtime": "cv2.dnn (OpenCV 5.0.0)"               # optional
      },
      "detections": [
        {
          "label":      "animal",       # animal | person | vehicle (MDv6 categories)
          "species":    "zebra",        # optional, only for label == animal
          "confidence": 0.5689,         # number in [0, 1]
          "bbox":       [0.248, 0.219, 0.658, 0.822]
                                        # normalized [x1, y1, x2, y2], each in [0, 1]
        }
      ]
    }

Design decisions (deliberate, documented for reviewers):

- **Normalized bboxes.** The S2 pipeline emits pixel boxes; producers convert
  with :func:`bbox_px_to_norm` before upload so records are resolution
  independent and comparable across cameras. ``width``/``height`` travel in
  the payload for provenance.
- **Clamping tolerance.** Real detector output can slightly exceed the frame
  before clamping (observed in S2 outputs, e.g. y = -0.79 px). Coordinates
  within ``BBOX_CLAMP_TOLERANCE`` of [0, 1] are clamped; anything further out
  is rejected as a producer bug.
- **Error policy.** Top-level structural problems (missing source, wrong
  schema_version, non-list detections, ...) raise :class:`SchemaError` — the
  handler treats those as poison and fails the invocation (DLQ). Per-record
  problems raise :class:`SchemaError` too, but the handler skips only the
  offending detection and keeps ingesting the rest (data-quality policy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1

#: MDv6 detection categories (person/vehicle present for triage policy,
#: animals get a secondary species classification).
LABELS = frozenset({"animal", "person", "vehicle"})

#: Classifier species tokens: lowercase, resilient to dashes/underscores.
SPECIES_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

#: Detector boxes may overshoot the frame by this much (fraction of the
#: frame edge) before they are considered a producer bug.
BBOX_CLAMP_TOLERANCE = 0.02

#: Minimum normalized box width/height; zero-area boxes are invalid.
BBOX_MIN_EDGE = 1e-4

# Tolerated float slop for comparisons.
_EPS = 1e-9


class SchemaError(ValueError):
    """A payload (or one of its detections) violates the contract."""


@dataclass(frozen=True)
class Detection:
    label: str
    species: str | None
    confidence: float
    # Normalized [x1, y1, x2, y2], each within [0, 1], x2 > x1, y2 > y1.
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Batch:
    schema_version: int
    source: dict[str, Any]
    model: dict[str, Any]
    # Raw per-detection mappings; the handler validates them one by one
    # with validate_detection() so a single bad record skips only itself.
    detections: tuple[Any, ...]


# ---------------------------------------------------------------------------
# Producers' helper
# ---------------------------------------------------------------------------


def bbox_px_to_norm(bbox: Sequence[float], width: int, height: int) -> list[float]:
    """Convert a pixel box [x1, y1, x2, y2] to normalized coordinates.

    Intended for the inference-node side: the S2 pipeline emits pixel boxes
    against a known frame size; the upload contract wants normalized ones.
    """
    if len(bbox) != 4:
        raise SchemaError(f"bbox must have 4 elements, got {len(bbox)}")
    if width <= 0 or height <= 0:
        raise SchemaError(f"width/height must be positive, got {width}x{height}")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_str(doc: Mapping, key: str, where: str) -> str:
    value = doc.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{where}.{key} must be a non-empty string, got {value!r}")
    return value


def _clamp_bbox(raw: Any, where: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise SchemaError(f"{where}.bbox must be a list of 4 numbers, got {raw!r}")
    try:
        coords = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise SchemaError(f"{where}.bbox must contain numbers, got {raw!r}") from None

    tol = BBOX_CLAMP_TOLERANCE + _EPS
    clamped = []
    for c in coords:
        if c < -tol or c > 1.0 + tol:
            raise SchemaError(
                f"{where}.bbox coordinate {c!r} outside tolerated range "
                f"[-{BBOX_CLAMP_TOLERANCE}, {1 + BBOX_CLAMP_TOLERANCE}]"
            )
        clamped.append(min(1.0, max(0.0, c)))

    x1, y1, x2, y2 = clamped
    if x2 - x1 < BBOX_MIN_EDGE or y2 - y1 < BBOX_MIN_EDGE:
        raise SchemaError(
            f"{where}.bbox must have x2 > x1 and y2 > y1 with non-zero area, got {clamped!r}"
        )
    return (x1, y1, x2, y2)


def validate_detection(raw: Any, where: str = "detections[i]") -> Detection:
    """Validate one detection record; raises :class:`SchemaError`."""
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{where} must be an object, got {type(raw).__name__}")

    label = raw.get("label")
    if label not in LABELS:
        raise SchemaError(f"{where}.label must be one of {sorted(LABELS)}, got {label!r}")

    species = raw.get("species")
    if species is not None:
        if label != "animal":
            raise SchemaError(
                f"{where}.species is only valid with label 'animal', got label {label!r}"
            )
        if not isinstance(species, str) or not SPECIES_PATTERN.match(species):
            raise SchemaError(
                f"{where}.species must match {SPECIES_PATTERN.pattern!r}, got {species!r}"
            )

    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise SchemaError(f"{where}.confidence must be a number, got {confidence!r}")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise SchemaError(f"{where}.confidence must be in [0, 1], got {confidence!r}")

    bbox = _clamp_bbox(raw.get("bbox"), where)

    return Detection(label=label, species=species, confidence=confidence, bbox=bbox)


def _validate_source(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SchemaError("source must be an object")

    key = _require_str(raw, "key", "source")
    if key.startswith("/") or ".." in key.split("/") or not key.strip():
        raise SchemaError(f"source.key must be a relative object key, got {key!r}")

    _require_str(raw, "image_id", "source")

    width = raw.get("width")
    height = raw.get("height")
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SchemaError(f"source.{name} must be a positive integer, got {value!r}")

    out: dict[str, Any] = {
        "key": key,
        "image_id": raw["image_id"],
        "width": width,
        "height": height,
    }
    for optional in ("dataset", "camera_id"):
        if raw.get(optional) is not None:
            out[optional] = _require_str(raw, optional, "source")
    if raw.get("is_night") is not None:
        is_night = raw["is_night"]
        if not isinstance(is_night, bool):
            raise SchemaError(f"source.is_night must be a boolean, got {is_night!r}")
        out["is_night"] = is_night
    if raw.get("captured_at") is not None:
        out["captured_at"] = _validate_ts(raw["captured_at"], "source.captured_at")
    return out


def _validate_model(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SchemaError("model must be an object with at least a name")
    name = _require_str(raw, "name", "model")
    out: dict[str, Any] = {"name": name}
    if raw.get("runtime") is not None:
        out["runtime"] = _require_str(raw, "runtime", "model")
    return out


def _validate_ts(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{where} must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SchemaError(f"{where} is not valid ISO-8601: {value!r}") from None
    if parsed.tzinfo is None:
        raise SchemaError(f"{where} must carry a timezone, got {value!r}")
    return value


def validate_batch(doc: Any) -> Batch:
    """Validate a whole detection batch; raises :class:`SchemaError`."""
    if not isinstance(doc, Mapping):
        raise SchemaError(f"payload must be a JSON object, got {type(doc).__name__}")

    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )

    source = _validate_source(doc.get("source"))
    model = _validate_model(doc.get("model"))

    raw_dets = doc.get("detections")
    if not isinstance(raw_dets, list):
        raise SchemaError(f"detections must be a list, got {type(raw_dets).__name__}")

    return Batch(
        schema_version=version,
        source=source,
        model=model,
        detections=tuple(raw_dets),
    )  # detections themselves are validated per-record by the handler
