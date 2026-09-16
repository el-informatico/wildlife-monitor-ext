"""Unit tests for infra/handler/app.py against moto (no LocalStack needed)."""

import hashlib
import json
from decimal import Decimal

import pytest

from conftest import BUCKET, TABLE, FakeContext, load_fixture, make_event

MEDIA_KEY = "raw/cam-01/2026-09-16/a001.jpg"
BATCH_KEY = "detections/cam-01/2026-09-16T10-15-00Z.json"


def put_batch(stack, doc, key=BATCH_KEY):
    body = json.dumps(doc).encode("utf-8")
    resp = stack["s3"].put_object(Bucket=BUCKET, Key=key, Body=body)
    return resp.get("VersionId"), resp["ETag"].strip('"')


def event_for(key, version_id, etag):
    ev = make_event(key, version_id=version_id, etag=etag)
    return ev


def run(stack, event):
    return stack["app"].lambda_handler(event, FakeContext())


def count_items(stack, key_condition, index=None, value=None):
    kwargs = {"IndexName": index} if index else {}
    resp = stack["table"].query(
        KeyConditionExpression=key_condition,
        ExpressionAttributeValues=value or {},
        **kwargs,
    )
    return resp["Items"]


def detections_for_camera(stack, camera="cam-01"):
    return count_items(stack, "pk = :c", value={":c": f"CAM#{camera}"})


def test_happy_path_writes_normalized_records(stack):
    version, etag = put_batch(stack, load_fixture())
    result = run(stack, event_for(BATCH_KEY, version, etag))

    assert result["processed"] == 1
    r = result["results"][0]
    assert r["detections_written"] == 2
    assert r["detections_skipped"] == 0
    assert r["duplicate"] is False

    items = detections_for_camera(stack)
    assert len(items) == 2
    items.sort(key=lambda i: i["sk"])

    batch_id = hashlib.sha256(MEDIA_KEY.encode()).hexdigest()[:12]
    ts = "2026-09-16T10:15:00.000Z"  # captured_at canonicalized to UTC ms
    first, second = items
    assert first["sk"] == f"{ts}|{batch_id}-000"
    assert first["det_id"] == f"{batch_id}-000"
    assert first["pk"] == "CAM#cam-01"
    assert first["entity"] == "detection"
    assert first["label"] == "animal"
    assert first["species"] == "zebra"
    assert first["confidence"] == Decimal("0.5689")
    assert first["bbox"] == [Decimal("0.2484"), Decimal("0.2193"),
                             Decimal("0.6582"), Decimal("0.8228")]
    assert first["gs1pk"] == "SPECIES#zebra"
    assert first["gs2pk"] == f"SRC#{MEDIA_KEY}"
    assert first["source_key"] == MEDIA_KEY
    assert first["image_id"] == "aw-00007"
    assert first["width"] == 550
    assert first["model_name"] == "MDV6-yolov10-c"
    # species-less detection indexes under its label
    assert second["gs1pk"] == "SPECIES#animal"
    assert "species" in second and second["species"] is None


def test_gsi_species_query(stack):
    version, etag = put_batch(stack, load_fixture())
    run(stack, event_for(BATCH_KEY, version, etag))
    zebra = count_items(
        stack, "gs1pk = :s", index="gsi-species", value={":s": "SPECIES#zebra"}
    )
    animal = count_items(
        stack, "gs1pk = :s", index="gsi-species", value={":s": "SPECIES#animal"}
    )
    assert len(zebra) == 1
    assert len(animal) == 1


def test_gsi_source_query(stack):
    version, etag = put_batch(stack, load_fixture())
    run(stack, event_for(BATCH_KEY, version, etag))
    items = count_items(
        stack, "gs2pk = :s", index="gsi-source", value={":s": f"SRC#{MEDIA_KEY}"}
    )
    assert len(items) == 2


def test_same_event_twice_is_idempotent(stack):
    version, etag = put_batch(stack, load_fixture())
    event = event_for(BATCH_KEY, version, etag)
    first = run(stack, event)
    second = run(stack, event)

    assert first["results"][0]["duplicate"] is False
    assert second["results"][0]["duplicate"] is True
    assert len(detections_for_camera(stack)) == 2


def test_reput_new_version_does_not_duplicate(stack):
    version1, etag1 = put_batch(stack, load_fixture())
    run(stack, event_for(BATCH_KEY, version1, etag1))
    # Same content re-uploaded: new version id, deterministic det ids collide.
    version2, etag2 = put_batch(stack, load_fixture())
    assert version2 != version1
    result = run(stack, event_for(BATCH_KEY, version2, etag2))

    assert len(detections_for_camera(stack)) == 2
    scanned = stack["table"].scan(FilterExpression="entity = :e",
                                  ExpressionAttributeValues={":e": "ingest-marker"})
    # One marker per object version; detections never duplicated.
    assert scanned["Count"] == 2


def test_invalid_detection_skipped_not_fatal(stack):
    doc = load_fixture()
    doc["detections"].append(
        {"label": "animal", "species": "zebra", "confidence": 1.5,
         "bbox": [0.1, 0.1, 0.5, 0.5]}
    )
    version, etag = put_batch(stack, doc)
    result = run(stack, event_for(BATCH_KEY, version, etag))

    r = result["results"][0]
    assert r["detections_written"] == 2
    assert r["detections_skipped"] == 1
    assert len(detections_for_camera(stack)) == 2


def test_poison_non_json_raises(stack):
    resp = stack["s3"].put_object(Bucket=BUCKET, Key=BATCH_KEY, Body=b"not json {{{")
    with pytest.raises(stack["app"].PoisonPayloadError, match="not valid JSON"):
        run(stack, event_for(BATCH_KEY, resp.get("VersionId"), resp["ETag"].strip('"')))


def test_poison_structural_schema_violation_raises(stack):
    doc = load_fixture()
    doc["schema_version"] = 99
    version, etag = put_batch(stack, doc)
    with pytest.raises(stack["app"].PoisonPayloadError, match="schema violation"):
        run(stack, event_for(BATCH_KEY, version, etag))


def test_non_detection_prefix_ignored(stack):
    resp = stack["s3"].put_object(Bucket=BUCKET, Key="raw/cam-01/a.jpg", Body=b"\xff\xd8")
    result = run(stack, event_for("raw/cam-01/a.jpg",
                                  resp.get("VersionId"), resp["ETag"].strip('"')))
    assert result["results"][0].get("ignored") is True
    assert detections_for_camera(stack) == []


def test_empty_detections_writes_marker_only(stack):
    doc = load_fixture()
    doc["detections"] = []
    version, etag = put_batch(stack, doc)
    result = run(stack, event_for(BATCH_KEY, version, etag))

    assert result["results"][0]["detections_written"] == 0
    assert detections_for_camera(stack) == []
    app = stack["app"]
    marker_pk = app._idempotency_key(BUCKET, BATCH_KEY, version, etag)
    marker = count_items(stack, "pk = :m", value={":m": marker_pk})
    assert len(marker) == 1
    assert marker[0]["sk"] == "BATCH"
    assert marker[0]["detection_count"] == 0


def test_chunked_writes_for_large_batches(stack):
    doc = load_fixture()
    doc["detections"] = [
        {"label": "animal", "species": None, "confidence": 0.5,
         "bbox": [0.05 * (i % 19), 0.1, 0.05 * (i % 19) + 0.04, 0.6]}
        for i in range(130)
    ]
    version, etag = put_batch(stack, doc)
    result = run(stack, event_for(BATCH_KEY, version, etag))

    assert result["results"][0]["detections_written"] == 130
    assert len(detections_for_camera(stack)) == 130


def test_camera_derived_from_media_key_when_absent(stack):
    doc = load_fixture()
    doc["source"].pop("camera_id")
    doc["source"]["key"] = "raw/cam-42/2026-09-16/b002.jpg"
    version, etag = put_batch(stack, doc)
    run(stack, event_for(BATCH_KEY, version, etag))
    assert {i["pk"] for i in detections_for_camera(stack, "cam-42")} == {"CAM#cam-42"}

    doc2 = load_fixture()
    doc2["source"].pop("camera_id")
    doc2["source"]["key"] = "african-wildlife/images/test/1 (154).jpg"
    put_batch(stack, doc2, key="detections/cam-01/b002.json")
    ev = make_event("detections/cam-01/b002.json", etag="e2")
    run(stack, ev)
    assert detections_for_camera(stack, "african-wildlife")


def test_structured_logging_is_json(stack, capsys):
    version, etag = put_batch(stack, load_fixture())
    run(stack, event_for(BATCH_KEY, version, etag))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines
    parsed = [json.loads(ln) for ln in lines]
    assert all({"ts", "level", "msg"} <= set(p) for p in parsed)
    assert any(p["msg"] == "ingested detection batch" and p["written"] == 2
               for p in parsed)


def test_non_s3_records_skipped(stack):
    result = run(stack, {"Records": [{"eventSource": "aws:sqs"}]})
    assert result == {"processed": 0, "results": []}


def test_object_removed_events_skipped(stack):
    ev = make_event(BATCH_KEY, event_name="ObjectRemoved:Delete")
    result = run(stack, ev)
    assert result == {"processed": 0, "results": []}
