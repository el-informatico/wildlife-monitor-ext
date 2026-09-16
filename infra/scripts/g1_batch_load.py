#!/usr/bin/env python
"""G1 batch load driver: manifest JSON → S3 → motion Lambda → DynamoDB.

Runs on infra/.venv (boto3, no pyarrow) against the LocalStack stack:

    .venv/bin/python scripts/g1_batch_load.py \
        --manifest build/media-manifest.json \
        --dataset african_wildlife

Phases (all idempotent; select with --phases):
  media    upload frame images to raw/<rel_path> (HEAD-first skip)
  jobs     upload motion-jobs/<dataset>/<camera>.json (skips jobs whose
           completion marker already exists) — this triggers the Lambda
  wait     poll DynamoDB until every job marker lands and the motion-row
           count is stable (or --timeout)
  replay   idempotency evidence: (a) direct invoke of one job's event →
           marker short-circuit, (b) re-put of that job as a NEW version →
           row-level condition collisions must not add rows
  verify   scan vs expected addresses, gate tallies, 3 spot-checks, DLQ
           peek, throughput — writes the JSON metrics report

Feed it the JSON extract produced by scripts/export_manifest_json.py
(run that one on the pipeline venv, which has pyarrow).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.parse
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_plan  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "build" / "g1-load-report.json"
PHASE_ORDER = ["media", "jobs", "wait", "replay", "verify"]


def make_client(service: str, endpoint: str) -> Any:
    return boto3.client(
        service,
        endpoint_url=endpoint,
        config=Config(retries={"max_attempts": 3, "mode": "standard"},
                      connect_timeout=10, read_timeout=120),
    )


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-friendly: {type(obj)}")


# --- phases -------------------------------------------------------------------

def phase_media(
    s3: Any, bucket: str, jobs: list[dict[str, Any]], data_root: Path, workers: int
) -> dict[str, Any]:
    """Upload every frame image the plan references; drop missing files."""
    media = sorted(
        {m["key"]: m for job in jobs for m in job["media"]}.values(),
        key=lambda m: m["key"],
    )
    stats = {"planned": len(media), "uploaded": 0, "skipped_existing": 0,
             "missing_local": 0, "bytes": 0}
    missing_keys: set[str] = set()

    def local_path(m: dict[str, Any]) -> Path:
        return data_root / m["rel_path"]

    for m in media:
        if not local_path(m).is_file():
            stats["missing_local"] += 1
            missing_keys.add(m["key"])

    todo = [m for m in media if m["key"] not in missing_keys]

    def remote_size(m: dict[str, Any]) -> tuple[str, int | None]:
        try:
            return m["key"], int(s3.head_object(Bucket=bucket, Key=m["key"])["ContentLength"])
        except ClientError:  # 404: not uploaded yet
            return m["key"], None
    # HEAD check up front (cheap on LocalStack, makes reruns fast).
    existing: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for key, size in pool.map(remote_size, todo):
            if size is not None:
                existing.add(key)
                stats["skipped_existing"] += 1
                stats["bytes"] += size

    def put(m: dict[str, Any]) -> int:
        body = local_path(m).read_bytes()
        for attempt in range(3):
            try:
                s3.put_object(Bucket=bucket, Key=m["key"], Body=body)
                return len(body)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        return 0  # unreachable

    fresh = [m for m in todo if m["key"] not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for nbytes in pool.map(put, fresh):
            stats["uploaded"] += 1
            stats["bytes"] += nbytes

    if missing_keys:
        for job in jobs:
            job["frames"] = [f for f in job["frames"] if f["key"] not in missing_keys]
        jobs[:] = [j for j in jobs if j["frames"]]
    return stats


def job_marker_pk(s3: Any, bucket: str, job_key: str) -> str | None:
    """Marker pk for a job's CURRENT object version, if the object exists."""
    try:
        head = s3.head_object(Bucket=bucket, Key=job_key)
    except ClientError:
        return None
    version = head.get("VersionId")
    etag = head.get("ETag", "").strip('"')
    return g1_plan.idempotency_pk(bucket, job_key, version, etag)


def marker_exists(dynamo: Any, table: str, marker_pk: str) -> bool:
    got = dynamo.get_item(
        TableName=table,
        Key={"pk": {"S": marker_pk}, "sk": {"S": "MOTIONJOB"}},
        ConsistentRead=True,
    )
    return "Item" in got


def phase_jobs(
    s3: Any, dynamo: Any, bucket: str, table: str, jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Put job files (skip already-completed ones); return upload records."""
    stats = {"uploaded": 0, "skipped_complete": 0}
    uploads: list[dict[str, str]] = []
    for job in jobs:
        marker_pk = job_marker_pk(s3, bucket, job["job_key"])
        if marker_pk and marker_exists(dynamo, table, marker_pk):
            stats["skipped_complete"] += 1
            continue
        body = json.dumps(g1_plan.job_document(job)).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=job["job_key"], Body=body,
                      ContentType="application/json")
        head = s3.head_object(Bucket=bucket, Key=job["job_key"])
        uploads.append(
            {
                "job_key": job["job_key"],
                "camera_id": job["camera_id"],
                "version_id": head.get("VersionId") or "",
                "etag": head.get("ETag", "").strip('"'),
                "marker_pk": g1_plan.idempotency_pk(
                    bucket, job["job_key"], head.get("VersionId"),
                    head.get("ETag", "").strip('"'),
                ),
            }
        )
        stats["uploaded"] += 1
    stats["uploads"] = uploads
    return stats


def count_motion_rows(dynamo: Any, table: str) -> int:
    total = 0
    kwargs: dict[str, Any] = {
        "TableName": table,
        "Select": "COUNT",
        "FilterExpression": "#e = :m",
        "ExpressionAttributeNames": {"#e": "entity"},
        "ExpressionAttributeValues": {":m": {"S": "motion"}},
    }
    while True:
        page = dynamo.scan(**kwargs)
        total += page["Count"]
        if "LastEvaluatedKey" not in page:
            return total
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def phase_wait(
    dynamo: Any, table: str, uploads: list[dict[str, str]], timeout: int
) -> dict[str, Any]:
    """Poll markers + row count until complete-and-stable or timeout."""
    stats = {"markers_found": 0, "rows": 0, "timed_out": False, "stable": False}
    deadline = time.monotonic() + timeout
    last_rows, last_rows_at = -1, time.monotonic()
    while time.monotonic() < deadline:
        found = sum(
            1 for u in uploads if marker_exists(dynamo, table, u["marker_pk"])
        )
        rows = count_motion_rows(dynamo, table)
        stats["markers_found"], stats["rows"] = found, rows
        if found == len(uploads) and rows == last_rows and rows > 0:
            if time.monotonic() - last_rows_at >= 8:  # stable across polls
                stats["stable"] = True
                return stats
        if rows != last_rows:
            last_rows, last_rows_at = rows, time.monotonic()
        time.sleep(2)
    stats["timed_out"] = True
    return stats


def synthetic_event(bucket: str, upload: dict[str, str]) -> dict[str, Any]:
    return {
        "Records": [
            {
                "eventSource": "aws:s3",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {
                        "key": urllib.parse.quote(upload["job_key"], safe="/"),
                        "versionId": upload["version_id"],
                        "eTag": upload["etag"],
                    },
                },
            }
        ]
    }


def phase_replay(
    s3: Any, dynamo: Any, lam: Any, bucket: str, table: str,
    function_name: str, jobs: list[dict[str, Any]],
    uploads: list[dict[str, str]], timeout: int,
) -> dict[str, Any]:
    """Two idempotency probes on one completed job (the last uploaded)."""
    out: dict[str, Any] = {}
    target_jobs = [j for j in jobs if j["job_key"] == uploads[-1]["job_key"]]
    job = target_jobs[0]
    upload = uploads[-1]
    rows_before = count_motion_rows(dynamo, table)

    # (a) same-version event redelivery → marker short-circuit
    resp = lam.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(synthetic_event(bucket, upload)),
    )
    payload = json.loads(resp["Payload"].read())
    stats = payload["results"][0] if payload.get("results") else {}
    out["marker_short_circuit"] = {
        "function_error": resp.get("FunctionError"),
        "duplicate": bool(stats.get("duplicate")),
        "rows_written": stats.get("frames_processed", 0),
    }

    # (b) re-put as a NEW version → row-level collisions must absorb it
    body = json.dumps(g1_plan.job_document(job)).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=job["job_key"], Body=body,
                  ContentType="application/json")
    head = s3.head_object(Bucket=bucket, Key=job["job_key"])
    marker_pk = g1_plan.idempotency_pk(
        bucket, job["job_key"], head.get("VersionId"), head.get("ETag", "").strip('"')
    )
    deadline = time.monotonic() + timeout
    marker_v2 = False
    while time.monotonic() < deadline:
        if marker_exists(dynamo, table, marker_pk):
            marker_v2 = True
            break
        time.sleep(2)
    time.sleep(3)  # let the write settle
    rows_after = count_motion_rows(dynamo, table)
    out["new_version_reput"] = {
        "marker_v2_found": marker_v2,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "unchanged": rows_after == rows_before,
    }
    return out


def scan_motion_rows(dynamo: Any, table: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "TableName": table,
        "FilterExpression": "#e = :m",
        "ExpressionAttributeNames": {"#e": "entity"},
        "ExpressionAttributeValues": {":m": {"S": "motion"}},
    }
    while True:
        page = dynamo.scan(**kwargs)
        for item in page["Items"]:
            flat = {k: list(v.values())[0] for k, v in item.items()}
            rows.append(flat)
        if "LastEvaluatedKey" not in page:
            return rows
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def phase_verify(
    dynamo: Any, sqs: Any | None, table: str, jobs: list[dict[str, Any]],
    expected_rows: list[dict[str, str]], dlq_name: str,
) -> dict[str, Any]:
    expected = {(e["pk"], e["sk"]): e for e in expected_rows}
    rows = scan_motion_rows(dynamo, table)
    actual = {(r["pk"], r["sk"]): r for r in rows}
    gates = {"warmup": 0, "passed": 0, "filtered": 0}
    nights = 0
    for r in rows:
        gates[r.get("gate", "?")] = gates.get(r.get("gate", "?"), 0) + 1
        nights += 1 if r.get("is_night") in (True, "true") else 0
    non_warmup = gates["passed"] + gates["filtered"]
    spot_picks = [expected_rows[0], expected_rows[len(expected_rows) // 2],
                  expected_rows[-1]] if expected_rows else []
    spot_checks = []
    for pick in spot_picks:
        got = dynamo.get_item(
            TableName=table,
            Key={"pk": {"S": pick["pk"]}, "sk": {"S": pick["sk"]}},
            ConsistentRead=True,
        ).get("Item")
        if not got:
            spot_checks.append({"pick": pick, "found": False})
            continue
        item = {k: list(v.values())[0] for k, v in got.items()}
        spot_checks.append(
            {
                "pick": pick,
                "found": True,
                "source_key_ok": item.get("source_key") == pick["source_key"],
                "gate": item.get("gate"),
                "fg_ratio": float(item.get("fg_ratio", Decimal(0))),
                "model_runtime": item.get("model_runtime"),
                "opencv_evidence": str(item.get("model_runtime", "")).startswith("opencv-"),
            }
        )
    dlq_messages = None
    if sqs is not None:
        try:
            url = sqs.get_queue_url(QueueName=dlq_name)["QueueUrl"]
            url = url.replace("localhost.localstack.cloud:4566",
                              urllib.parse.urlsplit(sqs.meta.endpoint_url).netloc)
            dlq_messages = len(
                sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10)
                .get("Messages", [])
            )
        except Exception as err:
            dlq_messages = f"unavailable: {type(err).__name__}"
    return {
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "missing_rows": sorted(f"{pk}|{sk}" for pk, sk in set(expected) - set(actual)),
        "extra_rows": sorted(f"{pk}|{sk}" for pk, sk in set(actual) - set(expected)),
        "gates": gates,
        "night_rows": nights,
        "pass_rate_non_warmup": round(gates["passed"] / non_warmup, 4) if non_warmup else None,
        "spot_checks": spot_checks,
        "dlq_messages": dlq_messages,
    }


# --- entry point ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON extract from scripts/export_manifest_json.py")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--cameras", default=None,
                        help="comma-separated location_id filter")
    parser.add_argument("--limit", type=int, default=None,
                        help="truncate selection (prefix slice)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bucket", default="wfm-raw-media-dev")
    parser.add_argument("--table", default="wfm-detections-dev")
    parser.add_argument("--endpoint",
                        default=os.environ.get("AWS_ENDPOINT_URL", "http://[::1]:4566"))
    parser.add_argument("--function-name", default="wfm-motion-gate-dev")
    parser.add_argument("--dlq-name", default="wfm-detection-ingest-dev-dlq")
    parser.add_argument("--phases", default=",".join(PHASE_ORDER),
                        help=f"subset/order of: {','.join(PHASE_ORDER)}")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames-per-job", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds for the wait/replay phases")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASE_ORDER]
    if unknown:
        parser.error(f"unknown phases: {unknown}")

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = g1_plan.select_records(
        rows, dataset=args.dataset,
        cameras=args.cameras.split(",") if args.cameras else None,
        limit=args.limit,
    )
    jobs = g1_plan.build_plan(selected, args.max_frames_per_job)
    expected_rows = g1_plan.expected_addresses(jobs)
    cameras = sorted({j["camera_id"] for j in jobs})
    report: dict[str, Any] = {
        "endpoint": args.endpoint, "bucket": args.bucket, "table": args.table,
        "manifest": str(args.manifest), "dataset": args.dataset,
        "selection": {
            "rows_selected": len(selected), "cameras": len(cameras),
            "jobs": len(jobs), "expected_frames": len(expected_rows),
            "night_frames": sum(1 for f in selected if f.get("is_night")),
            "camera_ids": cameras,
        },
    }
    print(f"plan: {len(selected)} rows -> {len(jobs)} jobs "
          f"({len(expected_rows)} expected motion rows)")

    s3 = make_client("s3", args.endpoint)
    dynamo = make_client("dynamodb", args.endpoint)
    lam = make_client("lambda", args.endpoint)
    try:
        sqs = make_client("sqs", args.endpoint)
    except Exception:
        sqs = None

    started = time.monotonic()
    uploads: list[dict[str, str]] = []
    jobs_wall = None

    if "media" in phases:
        t0 = time.monotonic()
        report["media"] = phase_media(s3, args.bucket, jobs, args.data_root, args.workers)
        wall = max(time.monotonic() - t0, 0.1)
        report["media"]["wall_s"] = round(wall, 1)
        n = max(report["media"]["uploaded"] + report["media"]["skipped_existing"], 1)
        report["media"]["imgs_per_min"] = round(n / (wall / 60), 1)
        print(f"media: {report['media']}")

    if "jobs" in phases:
        t0 = time.monotonic()
        job_stats = phase_jobs(s3, dynamo, args.bucket, args.table, jobs)
        uploads = job_stats.pop("uploads")
        report["jobs"] = job_stats
        jobs_wall = round(time.monotonic() - t0, 1)
        report["jobs"]["wall_s"] = jobs_wall
        print(f"jobs: uploaded={job_stats['uploaded']} "
              f"skipped_complete={job_stats['skipped_complete']}")

    if "wait" in phases:
        if not uploads:  # rerun with only wait/verify: rebuild from markers
            for job in jobs:
                pk = job_marker_pk(s3, args.bucket, job["job_key"])
                if pk:
                    head = s3.head_object(Bucket=args.bucket, Key=job["job_key"])
                    uploads.append({
                        "job_key": job["job_key"], "camera_id": job["camera_id"],
                        "version_id": head.get("VersionId") or "",
                        "etag": head.get("ETag", "").strip('"'),
                        "marker_pk": pk,
                    })
            if not uploads:
                parser.error("wait phase found no uploaded job objects")
        t0 = time.monotonic()
        report["wait"] = phase_wait(dynamo, args.table, uploads, args.timeout)
        wait_wall = max(time.monotonic() - t0, 0.1)
        report["wait"]["wall_s"] = round(wait_wall, 1)
        frames = len(expected_rows)
        report["processing"] = {
            "frames": frames,
            "frames_per_min": round(frames / (wait_wall / 60), 1),
        }
        print(f"wait: {report['wait']} -> {report['processing']}")

    if "replay" in phases:
        if not uploads:
            parser.error("replay phase needs the jobs phase (or its uploads)")
        report["replay"] = phase_replay(
            s3, dynamo, lam, args.bucket, args.table, args.function_name,
            jobs, uploads, args.timeout,
        )
        print(f"replay: {json.dumps(report['replay'], default=jsonable)}")

    if "verify" in phases:
        report["verify"] = phase_verify(
            dynamo, sqs, args.table, jobs, expected_rows, args.dlq_name,
        )
        v = report["verify"]
        print(f"verify: expected={v['expected_rows']} actual={v['actual_rows']} "
              f"gates={v['gates']} pass_rate={v['pass_rate_non_warmup']} "
              f"dlq={v['dlq_messages']}")

    report["wall_s_total"] = round(time.monotonic() - started, 1)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, default=jsonable, indent=2)
    print(f"report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
