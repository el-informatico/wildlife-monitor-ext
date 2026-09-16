"""Pure planning logic for the G1 batch load (no boto3, no cv2 — testable).

Turns ETL manifest rows into the exact S3 layout the motion Lambda
consumes, plus the item addresses the verifier expects to find in
DynamoDB afterwards. Everything here is deterministic so the driver's
verification can re-derive primary keys without touching the handler.

Layout produced (bucket = the infra ingest bucket):

    raw/<rel_path>                                  frame images
    motion-jobs/<dataset>/<camera>-<seq>.json       one job per camera
                                                    (split at max frames)

The timestamp/frame-id helpers mirror handler/motion_app.py; the offline
tests cross-check the two implementations against each other so they
cannot drift.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

MEDIA_PREFIX = "raw/"
JOBS_PREFIX = "motion-jobs/"


def select_records(
    rows: list[dict[str, Any]],
    dataset: str | None = None,
    cameras: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter manifest rows by dataset / camera ids / total limit.

    The exporter writes rows sorted by (dataset, camera, capture time), so
    a limit is a prefix slice: whole cameras first, possibly one partial
    camera at the tail (its frames are still sequence-ordered, so MOG2
    warmup behaves; useful for small smoke loads).
    """
    if dataset:
        rows = [r for r in rows if r["source_dataset"] == dataset]
    if cameras:
        wanted = set(cameras)
        rows = [r for r in rows if r["location_id"] in wanted]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return list(rows)


def canon_ts(value: str) -> str:
    """Mirror of motion_app._canon_ts: naive ISO is treated as UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def frame_id(source_key: str) -> str:
    """Mirror of motion_app._frame_id."""
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]


def media_key(row: dict[str, Any]) -> str:
    return MEDIA_PREFIX + row["rel_path"]


def item_address(camera_id: str, source_key: str, captured_at: str) -> dict[str, str]:
    """pk/sk the motion row for one frame must land at (verification)."""
    ts = canon_ts(captured_at)
    fid = frame_id(source_key)
    return {"pk": f"CAM#{camera_id}", "sk": f"{ts}|{fid}"}


def build_plan(
    rows: list[dict[str, Any]],
    max_frames_per_job: int = 2000,
) -> list[dict[str, Any]]:
    """Group selected rows into per-camera jobs.

    Each job: {job_key, camera_id, dataset, frames, media} where frames
    are capture-time-sorted and capped at max_frames_per_job (a long
    camera is split into sequentially numbered jobs — MOG2 restarts per
    job, so splitting re-runs warmup; kept >= the handler cap).
    """
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_camera.setdefault(row["location_id"], []).append(row)

    jobs: list[dict[str, Any]] = []
    for camera in sorted(by_camera):
        cam_rows = sorted(
            by_camera[camera], key=lambda r: (r["camera_ts"], r["image_id"])
        )
        dataset_slug = cam_rows[0]["source_dataset"].replace("/", "-")
        for seq, start in enumerate(range(0, len(cam_rows), max_frames_per_job)):
            chunk = cam_rows[start : start + max_frames_per_job]
            suffix = f"-{seq:04d}" if len(cam_rows) > max_frames_per_job else ""
            jobs.append(
                {
                    "job_key": f"{JOBS_PREFIX}{dataset_slug}/{camera}{suffix}.json",
                    "camera_id": camera,
                    "dataset": cam_rows[0]["source_dataset"],
                    "frames": [
                        {
                            "key": media_key(r),
                            "image_id": r["image_id"],
                            "captured_at": r["camera_ts"],
                            "is_night": bool(r["is_night"]),
                        }
                        for r in chunk
                    ],
                    "media": [
                        {"key": media_key(r), "rel_path": r["rel_path"]}
                        for r in chunk
                    ],
                }
            )
    return jobs


def job_document(job: dict[str, Any]) -> dict[str, Any]:
    """The JSON body uploaded at job_key (schema_version 1)."""
    return {
        "schema_version": 1,
        "camera_id": job["camera_id"],
        "dataset": job["dataset"],
        "frames": [
            {
                "key": f["key"],
                "image_id": f["image_id"],
                "captured_at": f["captured_at"],
                "is_night": f["is_night"],
            }
            for f in job["frames"]
        ],
    }


def idempotency_pk(bucket: str, key: str, version_id: str | None, etag: str | None) -> str:
    """Mirror of motion_app._idempotency_key (marker primary key)."""
    digest = hashlib.sha256(
        f"{bucket}|{key}|{version_id or etag or ''}".encode("utf-8")
    ).hexdigest()
    return f"IDEP#{digest}"


def expected_addresses(jobs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Every motion-row address a plan should produce in DynamoDB."""
    out: list[dict[str, str]] = []
    for job in jobs:
        for f in job["frames"]:
            entry = item_address(job["camera_id"], f["key"], f["captured_at"])
            entry["source_key"] = f["key"]
            out.append(entry)
    return out
