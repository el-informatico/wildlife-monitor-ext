"""Ingest Lambda: S3 detection-batch events → normalized DynamoDB records.

Triggered by ``s3:ObjectCreated:*`` on ``detections/*.json`` in the ingest
bucket. For each event record the handler:

1. computes an idempotency key from (bucket, key, versionId-or-eTag) and
   short-circuits if that exact object version was already ingested;
2. fetches and JSON-decodes the object — unreadable payloads raise
   :class:`PoisonPayloadError` so the async invocation fails and the message
   lands in the DLQ (producer contract broken, worth a human look);
3. validates the batch against ``schema`` (vendored next to this file);
   structural violations are poison (raise), individual bad detections are
   skipped with a warning (data quality, not an outage);
4. writes the detection records and the ingest marker in chunked
   ``TransactWriteItems`` where every item carries
   ``attribute_not_exists(pk) AND attribute_not_exists(sk)`` — replaying an
   event (same version or a re-put with a new version id) can never
   duplicate a row: the deterministic ids collide and the transaction is
   rejected, which the handler treats as "already ingested".

DynamoDB single-table layout (see infra/README.md for the full access-pattern
table):

- detection item:  pk = ``CAM#<camera>``,   sk = ``<ts>|<det_id>``
- ingest marker:   pk = ``IDEMP#<sha256>``, sk = ``BATCH``
- GSI gsi-species: gs1pk = ``SPECIES#<species-or-label>``, gs1sk = sk
- GSI gsi-source:  gs2pk = ``SRC#<source key>``,           gs2sk = sk

Runtime: Python 3.12, stdlib + the AWS-provided boto3 — nothing else, so the
zip is two files (``app.py`` + ``schema.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

from schema import SchemaError, validate_batch, validate_detection

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def _table_name() -> str:
    # Read per-invocation so tests can point the handler at their table.
    return os.environ.get("DYNAMO_TABLE", "wfm-detections-dev")


def _detections_prefix() -> str:
    return os.environ.get("DETECTIONS_PREFIX", "detections/")


def _max_detections() -> int:
    return int(os.environ.get("MAX_DETECTIONS", "200"))


# DynamoDB TransactWriteItems caps at 100 items; keep room for the marker.
_TRANSACT_CHUNK = 60

_serializer = TypeSerializer()


def _typed(item: dict[str, Any]) -> dict[str, Any]:
    """Convert plain-python attribute values to DynamoDB wire types."""
    return {k: _serializer.serialize(v) for k, v in item.items()}

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

_client_cache: dict[str, Any] = {}


class PoisonPayloadError(RuntimeError):
    """The uploaded batch violates the contract; fail so the DLQ catches it."""


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


def _derive_camera(source_key: str) -> str:
    """Camera id from the media key: first path segment, skipping 'raw/'."""
    parts = [p for p in source_key.split("/") if p]
    if not parts:
        return "unknown"
    if parts[0] == "raw" and len(parts) > 1:
        return parts[1]
    return parts[0]


def _idempotency_key(bucket: str, key: str, version_id: str | None, etag: str | None) -> str:
    digest = hashlib.sha256(
        f"{bucket}|{key}|{version_id or etag or ''}".encode("utf-8")
    ).hexdigest()
    return f"IDEMP#{digest}"


def _batch_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]


def _build_items(
    batch: Any,
    bucket: str,
    key: str,
    version_id: str | None,
    etag: str | None,
    event_time: str,
    detections: list[Any],
) -> list[dict[str, Any]]:
    source = batch.source
    ts = _canon_ts(source.get("captured_at") or event_time)
    camera = source.get("camera_id") or _derive_camera(source["key"])
    batch_id = _batch_id(source["key"])
    now = _utcnow_ms()

    marker = {
        "pk": _idempotency_key(bucket, key, version_id, etag),
        "sk": "BATCH",
        "entity": "ingest-marker",
        "source_bucket": bucket,
        "source_key": key,
        "source_version": version_id,
        "media_key": source["key"],
        "detection_count": len(detections),
        "ingested_at": now,
    }

    items: list[dict[str, Any]] = [marker]
    for i, det in enumerate(detections):
        det_id = f"{batch_id}-{i:03d}"
        sk = f"{ts}|{det_id}"
        item: dict[str, Any] = {
            "pk": f"CAM#{camera}",
            "sk": sk,
            "entity": "detection",
            "det_id": det_id,
            "gs1pk": f"SPECIES#{det.species or det.label}",
            "gs1sk": sk,
            "gs2pk": f"SRC#{source['key']}",
            "gs2sk": sk,
            "label": det.label,
            "species": det.species,
            "confidence": _num(det.confidence),
            "bbox": [_num(c) for c in det.bbox],
            "source_bucket": bucket,
            "source_key": source["key"],
            "image_id": source["image_id"],
            "width": source["width"],
            "height": source["height"],
            "model_name": batch.model["name"],
            "captured_at": ts,
            "ingested_at": now,
        }
        if source.get("dataset") is not None:
            item["dataset"] = source["dataset"]
        if source.get("is_night") is not None:
            item["is_night"] = source["is_night"]
        if batch.model.get("runtime") is not None:
            item["model_runtime"] = batch.model["runtime"]
        items.append(item)
    return items


def _write_items(table: str, items: list[dict[str, Any]]) -> str | None:
    """Chunked TransactWriteItems; returns None, or 'duplicate' when the
    first chunk (the one carrying the ingest marker, items[0]) was rejected
    as already present."""
    for start in range(0, len(items), _TRANSACT_CHUNK):
        chunk = items[start : start + _TRANSACT_CHUNK]
        transact_items = [
            {
                "Put": {
                    "TableName": table,
                    "Item": _typed(item),
                    # Item-level idempotency: deterministic ids collide on replay.
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
                    # The transaction rolled back atomically, so the marker
                    # for THIS object version was not written either (e.g.
                    # same media re-uploaded under a new version id: the
                    # detection ids collide but the marker is new). Record
                    # the version as seen without duplicating any rows.
                    _ensure_marker(table, items[0])
                    return "duplicate"
                _log("WARNING", "chunk already ingested (item-level condition)",
                     chunk_start=start, chunk_len=len(chunk))
            else:
                raise
    return None


def _ensure_marker(table: str, marker: dict[str, Any]) -> None:
    """Best-effort put of the ingest marker after a cancelled transaction."""
    if marker.get("entity") != "ingest-marker":
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


def _is_condition_rejection(err: ClientError) -> bool:
    """True when the write was rejected by attribute_not_exists.

    Inside TransactWriteItems a failed condition surfaces as
    TransactionCanceledException with per-item CancellationReasons; only
    an all-ConditionalCheckFailed cancellation is an idempotent replay —
    anything else (validation, throttle, ...) must propagate. Items whose
    condition passed report no reason (null on AWS; the string "None" on
    LocalStack/moto) and are ignored.
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


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    s3 = record["s3"]
    bucket = s3["bucket"]["name"]
    key = urllib.parse.unquote_plus(s3["object"]["key"])
    version_id = s3["object"].get("versionId")
    if version_id in (None, "", "null"):
        version_id = None
    etag = (s3["object"].get("eTag") or "").strip('"')
    event_time = record.get("eventTime") or _utcnow_ms()

    stats: dict[str, Any] = {
        "bucket": bucket, "key": key, "duplicate": False,
        "detections_written": 0, "detections_skipped": 0, "detections_total": 0,
    }

    if not key.startswith(_detections_prefix()) or not key.endswith(".json"):
        _log("DEBUG", "ignoring non-detection object", bucket=bucket, key=key)
        stats["ignored"] = True
        return stats

    marker_pk = _idempotency_key(bucket, key, version_id, etag)
    got = _client("dynamodb").get_item(
        TableName=_table_name(),
        Key={"pk": {"S": marker_pk}, "sk": {"S": "BATCH"}},
        ConsistentRead=True,
    )
    if "Item" in got:
        _log("INFO", "duplicate event: object version already ingested",
             bucket=bucket, key=key, version=version_id)
        stats["duplicate"] = True
        return stats

    obj = _client("s3").get_object(
        Bucket=bucket, Key=key, **({"VersionId": version_id} if version_id else {})
    )
    raw = obj["Body"].read()
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        _log("ERROR", "poison payload: not valid JSON", bucket=bucket, key=key,
             bytes=len(raw), error=str(err))
        raise PoisonPayloadError(f"{key}: not valid JSON: {err}") from err

    try:
        batch = validate_batch(doc)
    except SchemaError as err:
        _log("ERROR", "poison payload: structural schema violation",
             bucket=bucket, key=key, error=str(err))
        raise PoisonPayloadError(f"{key}: schema violation: {err}") from err

    detections: list[Any] = []
    for i, raw_det in enumerate(batch.detections):
        try:
            detections.append(validate_detection(raw_det, f"detections[{i}]"))
        except SchemaError as err:
            stats["detections_skipped"] += 1
            _log("WARNING", "skipping invalid detection", bucket=bucket, key=key,
                 index=i, error=str(err))
    if len(detections) > _max_detections():
        _log("WARNING", "truncating detection batch", bucket=bucket, key=key,
             total=len(detections), cap=_max_detections())
        detections = detections[: _max_detections()]
    stats["detections_total"] = len(batch.detections)

    items = _build_items(batch, bucket, key, version_id, etag, event_time, detections)
    outcome = _write_items(_table_name(), items)
    if outcome == "duplicate":
        stats["duplicate"] = True
        _log("INFO", "duplicate event: marker condition rejected the write",
             bucket=bucket, key=key)
    else:
        stats["detections_written"] = len(detections)
        _log("INFO", "ingested detection batch", bucket=bucket, key=key,
             media_key=batch.source["key"], camera=batch.source.get(
                 "camera_id") or _derive_camera(batch.source["key"]),
             written=len(detections), skipped=stats["detections_skipped"])
    return stats


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    _log("INFO", "invocation start",
         request_id=getattr(context, "aws_request_id", None),
         records=len(event.get("Records", [])))
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
        results.append(process_record(record))
    _log("INFO", "invocation done", results=results)
    return {"processed": len(results), "results": results}
