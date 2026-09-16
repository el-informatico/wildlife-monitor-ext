"""Unit tests for handler/motion_app.py (the motion-gate Lambda).

cv2/numpy never load here: a FakeEngine scripts the foreground ratios,
so these tests exercise the handler's real glue — job validation, gate
bookkeeping, frame skip policy, DynamoDB writes and both halves of the
idempotency contract — against moto. Evidence that the actual OpenCV 5
import works inside the Lambda comes from the LocalStack run documented
in G1-AWS-REPORT.md (every row carries model_runtime=opencv-5.0.0+...).
"""

import json
from decimal import Decimal

import pytest

from conftest import BUCKET, FakeContext, make_event
from handler import motion_app

UNDECODABLE = b"not-an-image"


class FakeGray:
    shape = (480, 640)  # h, w — enough for MotionEngine.dims


class FakeMask:
    def __init__(self, ratio):
        self.ratio = ratio


class FakeEngine:
    """Stands in for MotionEngine: scripted ratios, records recipe calls."""

    def __init__(self, ratios=(0.5,)):
        self.ratios = list(ratios)
        self.cv2_version = "5.0.0-test"
        self.calls = 0
        self.learning_rates = []
        self.night_flags = []

    def new_tracker(self):
        return self  # tracker.apply is FakeEngine.apply

    def apply(self, gray, learningRate=None):
        self.learning_rates.append(learningRate)
        ratio = self.ratios[min(self.calls, len(self.ratios) - 1)]
        self.calls += 1
        return FakeMask(ratio)

    def frame_gray(self, body, is_night):
        if body == UNDECODABLE:
            return None
        self.night_flags.append(is_night)
        return FakeGray()

    @staticmethod
    def dims(gray):
        h, w = gray.shape[:2]
        return int(w), int(h)

    @staticmethod
    def fg_ratio(mask):
        return mask.ratio


@pytest.fixture
def mstack(stack):
    motion_app._reset_clients()
    yield stack
    motion_app._reset_clients()


def frame(key, seconds, night=False, image_id=None):
    doc = {
        "key": key,
        "captured_at": f"2026-09-16T10:00:{seconds:02d}",
        "is_night": night,
    }
    if image_id is not None:
        doc["image_id"] = image_id
    return doc


def put_job(s3, frames, camera="missouri-C01", dataset="missouri"):
    doc = {"schema_version": 1, "camera_id": camera,
           "dataset": dataset, "frames": frames}
    key = f"motion-jobs/{dataset}/{camera}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(doc).encode("utf-8"))
    head = s3.head_object(Bucket=BUCKET, Key=key)
    return key, head.get("VersionId"), head.get("ETag", "").strip('"')


def put_frames(s3, keys):
    for key in keys:
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"fake-image-bytes")


def run_job(mstack, key, version, etag, engine):
    record = make_event(key, version_id=version, etag=etag)["Records"][0]
    return motion_app.process_record(record, engine=engine)


def motion_rows(mstack):
    resp = mstack["table"].scan(
        FilterExpression="#e = :m",
        ExpressionAttributeNames={"#e": "entity"},
        ExpressionAttributeValues={":m": "motion"},
    )
    return resp["Items"]


def marker_key(key, version, etag):
    return motion_app._idempotency_key(BUCKET, key, version, etag)


def test_happy_path_writes_rows_and_marker(mstack):
    s3 = mstack["s3"]
    keys = [f"raw/missouri/images/frame{i}.jpg" for i in range(5)]
    put_frames(s3, keys[:-1] + [keys[-1]])  # all five
    # index 3 dips under the threshold -> filtered; one night frame
    frames = [
        frame(keys[0], 0),
        frame(keys[1], 5),
        frame(keys[2], 10, night=True, image_id="mo-0002"),
        frame(keys[3], 15),
        frame(keys[4], 20),
    ]
    key, version, etag = put_job(s3, frames)
    engine = FakeEngine(ratios=[1.0, 1.0, 0.5, 0.01, 0.5])

    stats = run_job(mstack, key, version, etag, engine)

    assert stats["duplicate"] is False
    assert stats["cv2_version"] == "5.0.0-test"
    assert stats["frames_total"] == 5
    assert stats["frames_processed"] == 5
    assert stats["frames_failed"] == 0
    assert stats["gates"] == {"warmup": 2, "passed": 2, "filtered": 1}

    rows = motion_rows(mstack)
    assert len(rows) == 5
    by_sk = {r["sk"]: r for r in rows}
    warm = [r for r in rows if r["gate"] == "warmup"]
    assert len(warm) == 2 and all(r["warmup"] is True for r in warm)
    assert all(r["warmup"] is False for r in rows if r["gate"] != "warmup")
    # sk = canonical ts | frame id; naive input canonicalised to UTC ms
    fid = motion_app._frame_id(keys[2])
    row = by_sk[f"2026-09-16T10:00:10.000Z|{fid}"]
    assert row["pk"] == "CAM#missouri-C01"
    assert row["captured_at"] == "2026-09-16T10:00:10.000Z"
    assert row["source_key"] == keys[2]
    assert row["image_id"] == "mo-0002"
    assert row["is_night"] is True
    assert row["fg_ratio"] == Decimal("0.5")
    assert row["model_name"] == "MOG2"
    assert row["model_runtime"] == "opencv-5.0.0-test+aws-lambda-python3.12"
    assert row["width"] == 640 and row["height"] == 480
    # sparse gsi-species: motion rows carry the source GSI, never gs1 keys
    assert row["gs2pk"] == f"SRC#{keys[2]}" and row["gs2sk"] == row["sk"]
    assert "gs1pk" not in row and "gs1sk" not in row

    marker = mstack["table"].get_item(
        Key={"pk": marker_key(key, version, etag), "sk": "MOTIONJOB"}
    )["Item"]
    assert marker["entity"] == "motion-marker"
    assert marker["frames"] == 5 and marker["camera_id"] == "missouri-C01"

    # the recipe's MOG2 learning rate reaches tracker.apply untouched
    assert engine.learning_rates == [-1.0] * 5
    assert engine.night_flags == [False, False, True, False, False]


def test_duplicate_event_short_circuits_on_marker(mstack):
    s3 = mstack["s3"]
    frames = [frame("raw/missouri/images/a.jpg", 0),
              frame("raw/missouri/images/b.jpg", 30)]
    put_frames(s3, [f["key"] for f in frames])
    key, version, etag = put_job(s3, frames)
    engine = FakeEngine(ratios=[0.5, 0.5])
    run_job(mstack, key, version, etag, engine)

    stats = run_job(mstack, key, version, etag, engine)

    assert stats["duplicate"] is True
    assert stats["frames_processed"] == 0
    assert len(motion_rows(mstack)) == 2  # unchanged


def test_new_version_reput_collides_on_rows(mstack):
    """Same job re-put as a NEW version: the marker misses but every row
    address already exists, so the transact rejects and nothing is added."""
    s3 = mstack["s3"]
    frames = [frame("raw/missouri/images/a.jpg", 0),
              frame("raw/missouri/images/b.jpg", 30)]
    put_frames(s3, [f["key"] for f in frames])
    key, v1, e1 = put_job(s3, frames)
    run_job(mstack, key, v1, e1, FakeEngine(ratios=[0.5, 0.5]))

    _, v2, e2 = put_job(s3, frames)  # same body, new version
    stats = run_job(mstack, key, v2, e2, FakeEngine(ratios=[0.5, 0.5]))

    assert stats["duplicate"] is True
    assert stats["frames_processed"] == 2  # it did the work...
    assert len(motion_rows(mstack)) == 2   # ...but wrote nothing new
    # the new version's marker still lands (the rolled-back chunk-0 put
    # is re-attempted best-effort)
    marker = mstack["table"].get_item(
        Key={"pk": marker_key(key, v2, e2), "sk": "MOTIONJOB"}
    )["Item"]
    assert marker["entity"] == "motion-marker"


def test_undecodable_and_missing_frames_skip(mstack):
    s3 = mstack["s3"]
    s3.put_object(Bucket=BUCKET, Key="raw/missouri/images/good.jpg",
                  Body=b"bytes")
    s3.put_object(Bucket=BUCKET, Key="raw/missouri/images/bad.jpg",
                  Body=UNDECODABLE)
    frames = [
        frame("raw/missouri/images/good.jpg", 0),
        frame("raw/missouri/images/bad.jpg", 10),   # undecodable -> skip
        frame("raw/missouri/images/absent.jpg", 20),  # not in S3 -> skip
        frame("raw/missouri/images/good.jpg", 30),
    ]
    key, version, etag = put_job(s3, frames)
    engine = FakeEngine(ratios=[0.9, 0.9, 0.9])

    stats = run_job(mstack, key, version, etag, engine)

    assert stats["frames_total"] == 4
    assert stats["frames_processed"] == 2
    assert stats["frames_failed"] == 2
    rows = motion_rows(mstack)
    assert len(rows) == 2
    assert not any("absent" in r["source_key"] for r in rows)
    assert not any("bad.jpg" in r["source_key"] for r in rows)


@pytest.mark.parametrize("body, reason", [
    (b"this is not json", "undecodable bytes"),
    (json.dumps({"schema_version": 2, "camera_id": "c", "frames": []}), "bad schema"),
    (json.dumps({"schema_version": 1, "frames": [frame("k", 0)]}), "no camera"),
])
def test_poison_job_raises_to_dlq(mstack, body, reason):
    key = "motion-jobs/missouri/poison.json"
    mstack["s3"].put_object(Bucket=BUCKET, Key=key, Body=body)
    head = mstack["s3"].head_object(Bucket=BUCKET, Key=key)
    record = make_event(key, version_id=head.get("VersionId"),
                        etag=head.get("ETag"))["Records"][0]
    with pytest.raises(motion_app.MotionJobError):
        motion_app.process_record(record, engine=FakeEngine())
    assert motion_rows(mstack) == []


def test_non_job_objects_are_ignored(mstack):
    record = make_event("raw/camera1/img.jpg", version_id="v")["Records"][0]
    stats = motion_app.process_record(record, engine=FakeEngine())
    assert stats["ignored"] is True and stats["frames_total"] == 0


@pytest.mark.parametrize("doc, reason", [
    ("a string", "not an object"),
    ({"schema_version": 1}, "no camera"),
    ({"schema_version": 1, "camera_id": ""}, "empty camera"),
    ({"schema_version": 1, "camera_id": "c", "frames": []}, "no frames"),
    ({"schema_version": 1, "camera_id": "c", "frames": ["k"]}, "frame not obj"),
    (
        {"schema_version": 1, "camera_id": "c",
         "frames": [{"key": "k", "captured_at": "10:00", "is_night": False}]},
        "bad timestamp",
    ),
    (
        {"schema_version": 1, "camera_id": "c",
         "frames": [{"key": "k", "captured_at": "2026-09-16T10:00:00",
                     "is_night": "yes"}]},
        "is_night not bool",
    ),
])
def test_validate_job_rejects(doc, reason):
    with pytest.raises(motion_app.MotionJobError):
        motion_app.validate_job(doc)


def test_validate_job_enforces_frame_cap(monkeypatch):
    monkeypatch.setenv("MAX_JOB_FRAMES", "2")
    frames = [frame(f"k{i}", i) for i in range(3)]
    with pytest.raises(motion_app.MotionJobError, match="cap"):
        motion_app.validate_job(
            {"schema_version": 1, "camera_id": "c", "frames": frames}
        )


def test_validate_job_keeps_only_needed_fields():
    doc = {
        "schema_version": 1, "camera_id": "c", "dataset": "missouri",
        "surprise": True,
        "frames": [
            {"key": "k", "captured_at": "2026-09-16T10:00:00", "is_night": True,
             "image_id": "id-1", "extra": 7},
            {"key": "k2", "captured_at": "2026-09-16T10:00:01",
             "is_night": False, "image_id": 42},
        ],
    }
    clean = motion_app.validate_job(doc)
    assert clean["camera_id"] == "c"
    assert clean["frames"][0] == {
        "key": "k", "captured_at": "2026-09-16T10:00:00",
        "is_night": True, "image_id": "id-1",
    }
    # a non-string image_id is dropped rather than kept/coerced
    assert clean["frames"][1]["image_id"] is None


def test_lambda_handler_routes_only_s3_created(mstack, monkeypatch):
    s3 = mstack["s3"]
    frames = [frame("raw/missouri/images/a.jpg", 0)]
    put_frames(s3, [frames[0]["key"]])
    key, version, etag = put_job(s3, frames)
    engine = FakeEngine(ratios=[0.5])
    monkeypatch.setattr(motion_app, "_load_engine", lambda: engine)

    event = {
        "Records": [
            {"eventSource": "aws:sqs", "body": "{}"},          # not S3
            make_event("motion-jobs/missouri/gone.json",
                       event_name="ObjectRemoved:Delete")["Records"][0],
            make_event(key, version_id=version, etag=etag)["Records"][0],
        ]
    }
    result = motion_app.lambda_handler(event, FakeContext())

    assert result["processed"] == 1
    assert result["results"][0]["gates"] == {"warmup": 1, "passed": 0, "filtered": 0}
    assert len(motion_rows(mstack)) == 1


def test_write_chunks_beyond_one_transaction(mstack, monkeypatch):
    """Rows spanning several TransactWriteItems chunks all land."""
    s3 = mstack["s3"]
    frames = [frame(f"raw/missouri/images/f{i}.jpg", i) for i in range(5)]
    put_frames(s3, [f["key"] for f in frames])
    key, version, etag = put_job(s3, frames)
    monkeypatch.setattr(motion_app, "_TRANSACT_CHUNK", 2)  # marker+1, 2, 2

    stats = run_job(mstack, key, version, etag,
                    FakeEngine(ratios=[0.5] * 5))

    assert stats["frames_processed"] == 5
    assert stats["frames_failed"] == 0
    assert len(motion_rows(mstack)) == 5
