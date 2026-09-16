#!/usr/bin/env python3
"""End-to-end smoke test for the sprint-3 ingest slice on LocalStack.

Orchestrates the full loop with no cloud credentials:

    docker compose up (pinned localstack:2026.08.2)
      → terraform apply (bucket + notification + lambda + dynamodb + dlq)
      → upload the unit-test fixture batch to s3://<bucket>/detections/...
      → poll DynamoDB until the two normalized detection rows appear
      → assert record contents (keys, species, confidence, bbox, GSI keys)
      → re-upload the same batch (new version) and assert no duplicates
      → assert the on-failure DLQ destination config (0 retries → DLQ)
      → upload a poison batch and poll the DLQ until it receives the event
      → terraform destroy + docker compose down (unless --keep)

Exit code 0 = every step passed. Requires: docker, the pinned terraform in
infra/.tools (scripts/install-terraform.sh), and the uv dev env for boto3.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import boto3
from botocore.config import Config
from urllib.parse import urlparse, urlunparse

INFRA = pathlib.Path(__file__).resolve().parent.parent
TERRAFORM = INFRA / ".tools" / "terraform"
TF_DIR = INFRA / "terraform"
FIXTURE = INFRA / "tests" / "fixtures" / "detection_batch_sample.json"

REGION = "us-east-1"
# IPv6 loopback: see infra/terraform/variables.tf — plain 127.0.0.1 is
# unreachable on WSL2 mirrored-networking hosts, [::1] works everywhere
# docker publishes ports.
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://[::1]:4566")
MEDIA_KEY = "raw/cam-01/2026-09-16/a001.jpg"
BATCH_KEY = "detections/cam-01/2026-09-16T10-15-00Z.json"
POISON_KEY = "detections/cam-01/poison.json"

POLL_SECONDS = 90.0
POLL_INTERVAL = 2.0

failures: list[str] = []


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def sh(cmd: list[str], cwd: pathlib.Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, flush=True)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(label)


def poll(fn, timeout: float = POLL_SECONDS, interval: float = POLL_INTERVAL):
    """Call fn() every interval until it returns truthy or timeout hits."""
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(interval)
    return result


def client(service: str):
    return boto3.client(
        service,
        region_name=REGION,
        endpoint_url=ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"max_attempts": 2}),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true",
        help="leave the stack up after the run (skip destroy + compose down)",
    )
    args = parser.parse_args()

    s3 = client("s3")
    dynamo = client("dynamodb")
    lam = client("lambda")
    sqs = client("sqs")

    step("start LocalStack (docker compose up --wait)")
    sh(["docker", "compose", "up", "--wait", "-d"], cwd=INFRA)
    health = sh(["docker", "inspect", "--format", "{{.State.Health.Status}}",
                 "wfm-localstack"]).strip()
    check("localstack container healthy", health == "healthy", health)

    step("terraform apply")
    if not (TF_DIR / ".terraform").exists():
        sh([str(TERRAFORM), "init", "-input=false"], cwd=TF_DIR)
    sh([str(TERRAFORM), "apply", "-input=false", "-auto-approve"], cwd=TF_DIR)
    raw = sh([str(TERRAFORM), "output", "-json"], cwd=TF_DIR)
    out = {k: v["value"] for k, v in json.loads(raw).items()}
    bucket, table = out["bucket_name"], out["table_name"]
    fn_name, dlq_url = out["function_name"], out["dlq_url"]
    # LocalStack hands out queue URLs on its canonical hostname
    # (localhost.localstack.cloud), whose public wildcard DNS points at
    # 127.0.0.1 — unreachable on WSL2 mirrored-networking hosts. boto3 sends
    # receive_message to the QueueUrl's own host, so rewrite the netloc to our
    # endpoint; the path (/<account>/<queue>) is plain path-style addressing
    # the edge gateway accepts.
    q, e = urlparse(dlq_url), urlparse(ENDPOINT)
    dlq_url = urlunparse(q._replace(scheme=e.scheme, netloc=e.netloc))
    print(f"bucket={bucket} table={table} function={fn_name} dlq={dlq_url}")

    step(f"upload fixture batch s3://{bucket}/{BATCH_KEY}")
    body = FIXTURE.read_bytes()
    put = s3.put_object(Bucket=bucket, Key=BATCH_KEY, Body=body)
    version1 = put.get("VersionId")
    check("object stored with versionId", bool(version1), f"VersionId={version1}")

    step("poll DynamoDB gsi-source for the two normalized detections")
    def rows():
        resp = dynamo.query(
            TableName=table, IndexName="gsi-source",
            KeyConditionExpression="gs2pk = :k",
            ExpressionAttributeValues={":k": {"S": f"SRC#{MEDIA_KEY}"}},
        )
        return resp.get("Items") or []
    items = poll(rows)
    check("2 detection rows via gsi-source", len(items or []) == 2,
          f"got {len(items or [])}")
    if items:
        row = min(items, key=lambda i: i["sk"]["S"])  # -000 = the zebra
        sk = row["sk"]["S"]
        check("pk = CAM#cam-01", row["pk"]["S"] == "CAM#cam-01", row["pk"]["S"])
        check("sk = <captured_at>|<det_id>",
              sk.startswith("2026-09-16T10:15:00.000Z|") and sk.endswith("-000"), sk)
        check("entity = detection", row.get("entity", {}).get("S") == "detection")
        check("species = zebra", row.get("species", {}).get("S") == "zebra",
              str(row.get("species")))
        check("confidence = 0.5689",
              row.get("confidence", {}).get("N") == "0.5689",
              row.get("confidence", {}).get("N", ""))
        bbox = [float(n["N"]) for n in row.get("bbox", {}).get("L", [])]
        check("bbox normalized [0.2484, 0.2193, 0.6582, 0.8228]",
              all(abs(a - b) < 1e-3 for a, b in
                  zip(bbox, [0.2484, 0.2193, 0.6582, 0.8228])) and len(bbox) == 4,
              str(bbox))
        check("gs1pk = SPECIES#zebra",
              row.get("gs1pk", {}).get("S") == "SPECIES#zebra",
              str(row.get("gs1pk")))
        check("image_id = aw-00007",
              row.get("image_id", {}).get("S") == "aw-00007")

    step("idempotency: re-upload same batch (new version) → no duplicates")
    put2 = s3.put_object(Bucket=bucket, Key=BATCH_KEY, Body=body)
    version2 = put2.get("VersionId")
    check("re-put produced new versionId",
          bool(version2) and version2 != version1, f"{version1} → {version2}")

    def marker_count():
        resp = dynamo.scan(TableName=table,
                           FilterExpression="entity = :e",
                           ExpressionAttributeValues={":e": {"S": "ingest-marker"}})
        return resp.get("Count", 0)
    markers = poll(lambda: marker_count() == 2 and 2)
    check("one ingest marker per object version", markers == 2,
          f"markers={marker_count()}")
    time.sleep(5)  # let the second invocation settle
    check("still 2 detection rows after re-upload", len(rows() or []) == 2,
          f"got {len(rows() or [])}")

    step("DLQ wiring: on-failure destination, 0 retries")
    cfg = lam.get_function_event_invoke_config(FunctionName=fn_name)
    dest = (cfg.get("DestinationConfig", {}).get("OnFailure", {})
            .get("Destination", ""))
    check("on-failure destination = DLQ arn", dest == out["dlq_arn"], dest)
    check("maximum_retry_attempts = 0", cfg.get("MaximumRetryAttempts") == 0,
          str(cfg.get("MaximumRetryAttempts")))

    step(f"poison batch → DLQ receives the failed event ({POISON_KEY})")
    s3.put_object(Bucket=bucket, Key=POISON_KEY,
                  Body=b"this is not json {{{")
    def dlq_messages():
        resp = sqs.receive_message(QueueUrl=dlq_url, MaxNumberOfMessages=10,
                                   WaitTimeSeconds=2)
        return resp.get("Messages") or []
    msgs = poll(dlq_messages, timeout=120.0)
    check("failed invocation landed in DLQ", bool(msgs),
          f"{len(msgs or [])} message(s)")
    if msgs:
        payload = json.loads(msgs[0]["Body"])
        # Standard SQS destination envelope: {"Records": [...]} wrapped in
        # requestContext/... — either way the poison key must be in there.
        check("DLQ payload references the poison object",
              POISON_KEY in json.dumps(payload),
              str(payload)[:120] + "…")

    step("teardown")
    if args.keep:
        print("--keep: stack left running (terraform destroy + compose down skipped)")
    else:
        sh([str(TERRAFORM), "destroy", "-input=false", "-auto-approve"], cwd=TF_DIR)
        sh(["docker", "compose", "down", "--volumes"], cwd=INFRA)
        print("stack destroyed, containers down")

    print()
    if failures:
        print(f"SMOKE: FAIL ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE: PASS — all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
