"""Motion-gate Lambda: S3 motion-job events → MOG2 → DynamoDB motion rows.

Triggered by ``s3:ObjectCreated:*`` on ``motion-jobs/*.json`` in the ingest
bucket. A motion job describes ONE camera location: an ordered list of
frames (capture order — the driver sorts by capture timestamp before
writing the job). For each frame the handler fetches the image bytes from
S3 and runs the project's motion recipe inside this Lambda:

    decode → cap long side at 1024 (INTER_AREA) → grayscale
           → CLAHE (night frames only) → MOG2 → foreground ratio

MOG2 parameters, warmup handling and the trigger threshold mirror the
local motion benchmark exactly (history=50, varThreshold=24.0,
detectShadows=False, learningRate=-1.0; first 2 frames per camera are
warmup; a frame "passes" when fg_ratio > 0.02). The OpenCV import is the
point of the exercise: this function runs on the custom execution image
built by scripts/build-lambda-image.sh, which carries the project's
source-built OpenCV 5 — cv2.__version__ is read INSIDE the handler and
logged/stored on every row as evidence.

DynamoDB single-table layout (see infra/README.md):

- motion item:     pk = ``CAM#<camera>``,  sk = ``<ts>|<frame_id>``
- job marker:      pk = ``IDEP#<sha256>``, sk = ``MOTIONJOB``
- GSI gsi-source:  gs2pk = ``SRC#<media key>``, gs2sk = sk
- no gsi-species keys: motion rows simply do not appear in that index
  (sparse GSI — a documented DynamoDB pattern).

Idempotency copies the ingest handler's contract: a marker keyed by
(bucket, key, versionId-or-eTag) short-circuits replays, and every row is
written under ``attribute_not_exists(pk) AND attribute_not_exists(sk)``
in chunked TransactWriteItems, so a re-processed job can never duplicate
a row (deterministic frame ids collide instead).

Failure policy: an unreadable/unparseable JOB file is poison (raise
:class:`MotionJobError` → async invoke fails → DLQ); an undecodable FRAME
is a data-quality skip (counted, row not written), matching the local
benchmark's behaviour.

Runtime: Python 3.12 + cv2 + numpy + the AWS-provided boto3. The zip is
one file; cv2/numpy come from the execution image.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

# --- the motion recipe (mirrors the local benchmark; keep in sync) -----------
WARMUP_FRAMES_PER_CAMERA = 2   # MOG2 has no background model yet
TRIGGER_FG_THRESHOLD = 0.02    # motion trigger on foreground ratio
MAX_LONG_SIDE = 1024           # working resolution cap
MOG2_HISTORY = 50
MOG2_VAR_THRESHOLD = 24.0
MOG2_LEARNING_RATE = -1.0      # automatic, from history size
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# DynamoDB TransactWriteItems caps at 100 items; keep room for the marker.
_TRANSACT_CHUNK = 60

_serializer = TypeSerializer()

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

_client_cache: dict[str, Any] = {}


class MotionJobError(RuntimeError):
    """The motion job violates the contract; fail so the DLQ catches it."""


def _table_name() -> str:
    return os.environ.get("DYNAMO_TABLE", "wfm-detections-dev")


def _jobs_prefix() -> str:
    return os.environ.get("MOTION_JOBS_PREFIX", "motion-jobs/")


def _max_frames() -> int:
    return int(os.environ.get("MAX_JOB_FRAMES", "2500"))


def _client(service: str) -> Any:
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
    cache_key = f"{service}@{endpoint}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = boto3.client(service, endpoint_url=endpoint)
    return _client_cache[cache_key]


def _reset_clients() -> None:
    _client_cache.clear()


def _log(level: str, msg: str, **fields: Any) -> None:
    if _LEVELS.get(level, 20) < _LEVELS.get(LOG_LEVEL, 20):
        return
    print(
        json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
             "level": level, "msg": msg, **fields},
            default=str,
        ),
        flush=True,
    )


# --- pure helpers (unit-tested offline) --------------------------------------

def _utcnow_ms() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _canon_ts(value: str) -> str:
    """Canonical UTC timestamp with fixed-width milliseconds (sort-safe sk)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _num(value: float | int) -> Decimal:
    # DynamoDB numbers must be Decimal (botocore rejects float).
    return Decimal(str(value))


def _frame_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]


def _idempotency_key(bucket: str, key: str, version_id: str | None, etag: str | None) -> str:
    digest = hashlib.sha256(
        f"{bucket}|{key}|{version_id or etag or ''}".encode("utf-8")
    ).hexdigest()
    return f"IDEP#{digest}"


def gate_for(frame_index: int, fg_ratio: float) -> str:
    """warmup / passed / filtered — the benchmark's rule, verbatim."""
    if frame_index < WARMUP_FRAMES_PER_CAMERA:
        return "warmup"
    return "passed" if fg_ratio > TRIGGER_FG_THRESHOLD else "filtered"


def validate_job(doc: Any) -> dict[str, Any]:
    """Structural validation of a motion-job document.

    Raises MotionJobError on violation (poison). Frames keep only the
    fields the handler needs; unknown extras are ignored.
    """
    if not isinstance(doc, dict):
        raise MotionJobError("job must be a JSON object")
    if doc.get("schema_version") != 1:
        raise MotionJobError(f"unsupported schema_version: {doc.get('schema_version')!r}")
    camera = doc.get("camera_id")
    if not isinstance(camera, str) or not camera:
        raise MotionJobError("camera_id must be a non-empty string")
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MotionJobError("frames must be a non-empty list")
    if len(frames) > _max_frames():
        raise MotionJobError(f"job has {len(frames)} frames, cap is {_max_frames()}")
    clean = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise MotionJobError(f"frames[{i}] must be an object")
        key = frame.get("key")
        if not isinstance(key, str) or not key:
            raise MotionJobError(f"frames[{i}].key must be a non-empty string")
        captured_at = frame.get("captured_at")
        if not isinstance(captured_at, str) or not captured_at:
            raise MotionJobError(f"frames[{i}].captured_at must be a string")
        try:
            _canon_ts(captured_at)
        except ValueError as err:
            raise MotionJobError(
                f"frames[{i}].captured_at is not an ISO timestamp: {captured_at!r}"
            ) from err
        is_night = frame.get("is_night")
        if not isinstance(is_night, bool):
            raise MotionJobError(f"frames[{i}].is_night must be a boolean")
        clean.append(
            {
                "key": key,
                "captured_at": captured_at,
                "is_night": is_night,
                "image_id": frame.get("image_id") if isinstance(frame.get("image_id"), str) else None,
            }
        )
    return {"camera_id": camera, "dataset": doc.get("dataset"), "frames": clean}


def _frame_item(
    camera: str,
    frame: dict[str, Any],
    fg_ratio: float,
    gate: str,
    dims: tuple[int, int],
    cv2_version: str,
    now: str,
) -> dict[str, Any]:
    ts = _canon_ts(frame["captured_at"])
    frame_id = _frame_id(frame["key"])
    sk = f"{ts}|{frame_id}"
    item: dict[str, Any] = {
        "pk": f"CAM#{camera}",
        "sk": sk,
        "entity": "motion",
        "frame_id": frame_id,
        "gs2pk": f"SRC#{frame['key']}",
        "gs2sk": sk,
        "gate": gate,
        "warmup": gate == "warmup",
        "fg_ratio": _num(round(fg_ratio, 6)),
        "is_night": frame["is_night"],
        "width": dims[0],
        "height": dims[1],
        "camera_id": camera,
        "source_key": frame["key"],
        "model_name": "MOG2",
        "model_runtime": f"opencv-{cv2_version}+aws-lambda-python3.12",
        "captured_at": ts,
        "ingested_at": now,
    }
    if frame["image_id"] is not None:
        item["image_id"] = frame["image_id"]
    return item


def _typed(item: dict[str, Any]) -> dict[str, Any]:
    return {k: _serializer.serialize(v) for k, v in item.items()}


def _is_condition_rejection(err: ClientError) -> bool:
    """True when the write was rejected by attribute_not_exists.

    Inside TransactWriteItems a failed condition surfaces as
    TransactionCanceledException with per-item CancellationReasons; only
    an all-ConditionalCheckFailed cancellation is an idempotent replay —
    anything else must propagate. Items whose condition passed report no
    reason (null on AWS; the string "None" on LocalStack/moto).
    """
    code = err.response.get("Error", {}).get("Code", "")
    if code == "ConditionalCheckFailedException":
        return True
    if code != "TransactionCanceledException":
        return False
    reasons = err.response.get("CancellationReasons") or []
    codes = [
        r.get("Code")
        for r in reasons
        if isinstance(r, dict) and r.get("Code") not in (None, "", "None")
    ]
    return bool(codes) and all(c == "ConditionalCheckFailed" for c in codes)


def _write_items(table: str, items: list[dict[str, Any]]) -> str | None:
    """Chunked TransactWriteItems; returns None, or 'duplicate' when the
    first chunk (the one carrying the job marker, items[0]) was rejected
    as already present."""
    for start in range(0, len(items), _TRANSACT_CHUNK):
        chunk = items[start : start + _TRANSACT_CHUNK]
        transact_items = [
            {
                "Put": {
                    "TableName": table,
                    "Item": _typed(item),
                    "ConditionExpression": "attribute_not_exists(pk) AND attribute_not_exists(sk)",
                }
            }
            for item in chunk
        ]
        try:
            _client("dynamodb").transact_write_items(TransactItems=transact_items)
        except ClientError as err:
            if _is_condition_rejection(err):
                if start == 0:
                    _ensure_marker(table, items[0])
                    return "duplicate"
                _log("WARNING", "chunk already ingested (item-level condition)",
                     chunk_start=start, chunk_len=len(chunk))
            else:
                raise
    return None


def _ensure_marker(table: str, marker: dict[str, Any]) -> None:
    """Best-effort put of the job marker after a cancelled transaction."""
    if marker.get("entity") != "motion-marker":
        return
    try:
        _client("dynamodb").put_item(
            TableName=table,
            Item=_typed(marker),
            ConditionExpression="attribute_not_exists(pk)",
        )
    except ClientError as err:
        if not _is_condition_rejection(err):
            raise


# --- the cv2-bound image pipeline (swapped out in offline tests) -------------

class MotionEngine:
    """decode → resize cap → grayscale → CLAHE(night) → MOG2, on the
    execution image's source-built OpenCV 5. Everything cv2/numpy-shaped
    lives here so the rest of the handler stays pure."""

    def __init__(self) -> None:
        import cv2  # noqa: DLC101 — the whole point: OpenCV inside the Lambda
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
        )
        self.cv2_version = cv2.__version__

    def new_tracker(self) -> Any:
        return self._cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )

    def frame_gray(self, body: bytes, is_night: bool) -> Any:
        """Decode image bytes to the grayscale frame MOG2 consumes.

        Returns None when the bytes are not a decodable image (data
        quality, not an outage). Long side is capped at MAX_LONG_SIDE
        with INTER_AREA; night frames get CLAHE.
        """
        cv2 = self._cv2
        buf = self._np.frombuffer(body, dtype=self._np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = MAX_LONG_SIDE / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(
                bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if is_night:
            gray = self.clahe.apply(gray)
        return gray

    @staticmethod
    def dims(gray: Any) -> tuple[int, int]:
        h, w = gray.shape[:2]
        return int(w), int(h)

    @staticmethod
    def fg_ratio(mask: Any) -> float:
        import numpy as np

        return float(np.count_nonzero(mask) / mask.size)


def _load_engine() -> MotionEngine:
    return MotionEngine()


# --- event processing --------------------------------------------------------

def process_record(record: dict[str, Any], engine: MotionEngine | None = None) -> dict[str, Any]:
    s3 = record["s3"]
    bucket = s3["bucket"]["name"]
    key = urllib.parse.unquote_plus(s3["object"]["key"])
    version_id = s3["object"].get("versionId")
    if version_id in (None, "", "null"):
        version_id = None
    etag = (s3["object"].get("eTag") or "").strip('"')

    stats: dict[str, Any] = {
        "bucket": bucket, "key": key, "duplicate": False,
        "frames_total": 0, "frames_processed": 0, "frames_failed": 0,
        "gates": {"warmup": 0, "passed": 0, "filtered": 0},
        "cv2_version": None,
    }

    if not key.startswith(_jobs_prefix()) or not key.endswith(".json"):
        _log("DEBUG", "ignoring non-motion-job object", bucket=bucket, key=key)
        stats["ignored"] = True
        return stats

    marker_pk = _idempotency_key(bucket, key, version_id, etag)
    got = _client("dynamodb").get_item(
        TableName=_table_name(),
        Key={"pk": {"S": marker_pk}, "sk": {"S": "MOTIONJOB"}},
        ConsistentRead=True,
    )
    if "Item" in got:
        _log("INFO", "duplicate event: job version already ingested",
             bucket=bucket, key=key, version=version_id)
        stats["duplicate"] = True
        return stats

    raw = _client("s3").get_object(
        Bucket=bucket, Key=key, **({"VersionId": version_id} if version_id else {})
    )["Body"].read()
    try:
        doc = json.loads(raw)
        job = validate_job(doc)
    except (UnicodeDecodeError, json.JSONDecodeError, MotionJobError) as err:
        _log("ERROR", "poison motion job", bucket=bucket, key=key, error=str(err))
        raise MotionJobError(f"{key}: {err}") from err

    if engine is None:
        engine = _load_engine()
    stats["cv2_version"] = engine.cv2_version

    camera = job["camera_id"]
    tracker = engine.new_tracker()
    now = _utcnow_ms()
    items: list[dict[str, Any]] = [
        _marker_with(marker_pk, bucket, key, camera, len(job["frames"]), now)
    ]
    ms_decode_total = 0.0
    ms_mog2_total = 0.0

    for i, frame in enumerate(job["frames"]):
        stats["frames_total"] += 1
        try:
            body = _client("s3").get_object(Bucket=bucket, Key=frame["key"])["Body"].read()
        except ClientError as err:
            stats["frames_failed"] += 1
            _log("WARNING", "frame object missing; skipping", key=frame["key"],
                 index=i, error=err.response.get("Error", {}).get("Code"))
            continue
        t0 = time.perf_counter()
        gray = engine.frame_gray(body, frame["is_night"])
        if gray is None:
            stats["frames_failed"] += 1
            _log("WARNING", "frame undecodable; skipping", key=frame["key"], index=i)
            continue
        t1 = time.perf_counter()
        mask = tracker.apply(gray, learningRate=MOG2_LEARNING_RATE)
        fg_ratio = engine.fg_ratio(mask)
        t2 = time.perf_counter()
        ms_decode_total += (t1 - t0) * 1000
        ms_mog2_total += (t2 - t1) * 1000

        gate = gate_for(i, fg_ratio)
        stats["frames_processed"] += 1
        stats["gates"][gate] += 1
        items.append(
            _frame_item(
                camera, frame, fg_ratio, gate, engine.dims(gray),
                engine.cv2_version, now,
            )
        )

    outcome = _write_items(_table_name(), items)
    if outcome == "duplicate":
        stats["duplicate"] = True
        _log("INFO", "duplicate event: marker condition rejected the write",
             bucket=bucket, key=key)
    else:
        _log(
            "INFO", "motion job done",
            bucket=bucket, key=key, camera=camera,
            frames=stats["frames_total"], processed=stats["frames_processed"],
            failed=stats["frames_failed"], gates=stats["gates"],
            cv2_version=engine.cv2_version,
            ms_decode_mean=round(ms_decode_total / max(stats["frames_processed"], 1), 1),
            ms_mog2_mean=round(ms_mog2_total / max(stats["frames_processed"], 1), 1),
        )
    return stats


def _marker_with(
    marker_pk: str, bucket: str, key: str, camera: str, frame_count: int, now: str
) -> dict[str, Any]:
    return {
        "pk": marker_pk,
        "sk": "MOTIONJOB",
        "entity": "motion-marker",
        "source_bucket": bucket,
        "source_key": key,
        "camera_id": camera,
        "frames": frame_count,
        "ingested_at": now,
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    # The evidence line: OpenCV, imported inside this Lambda.
    try:
        probe = _load_engine()
        cv2_version = probe.cv2_version
    except Exception as err:  # pragma: no cover - image misconfigured
        _log("ERROR", "cv2 unavailable in execution environment", error=str(err))
        raise
    _log("INFO", "invocation start",
         request_id=getattr(context, "aws_request_id", None),
         records=len(event.get("Records", [])),
         cv2_version=cv2_version)
    results = []
    for record in event.get("Records", []):
        if record.get("eventSource") != "aws:s3":
            _log("WARNING", "ignoring non-S3 record",
                 source=record.get("eventSource"))
            continue
        if not str(record.get("eventName", "")).startswith("ObjectCreated"):
            _log("WARNING", "ignoring non-ObjectCreated event",
                 name=record.get("eventName"))
            continue
        results.append(process_record(record, engine=probe))
    _log("INFO", "invocation done", results=results)
    return {"processed": len(results), "results": results}
