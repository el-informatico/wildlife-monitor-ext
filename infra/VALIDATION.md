# infra — validation record

Measured results for every mandated check, run on the dev workstation
(WSL2, docker daemon with `userns-remap: default`). Commands are copied
verbatim; outputs are the real ones. Date: 2026-09-16.

## 1. Unit tests (moto — no LocalStack required)

```
$ cd infra && uv run pytest -q
63 passed in 3.97s
```

Coverage: schema validation (valid/invalid batches, pixel→normalized bbox,
timestamp canonicalization), handler normalization (pk/sk, GSI keys,
deterministic det ids, confidence/bbox typing), idempotency (marker fast
path, duplicate event, re-put with new version → transaction condition
rejection → marker backfill, zero duplicate rows), error policy (poison
payload raises, per-record skip), structured logging shape.

## 2. LocalStack bring-up (docker compose)

```
$ cd infra && docker compose up -d --wait
Container wfm-localstack Started
Container wfm-localstack Healthy
```

- Image: `localstack/localstack:4.14.0` — **1.12 GB on disk** after the
  first pull (plus ~517 MB for `public.ecr.aws/lambda/python:3.12`, pulled
  lazily by the first Lambda invocation).
- Why 4.14.0 and not current: LocalStack 2026.x requires an account auth
  token to start (distribution change enforced April 2026; verified
  empirically — 2026.08.2 exits with code 55 "License activation failed…
  No credentials"). 4.14.0 (2026-02-26) is the last anonymous release, which
  keeps this repo credential-free. The `localstack-community` Docker repo
  was deleted upstream, so the future upgrade path is `localstack/localstack`
  CalVer + `LOCALSTACK_AUTH_TOKEN`.

## 3. Terraform apply

Toolchain: terraform **1.16.3** (pinned; 1.17.0 is listed in the release
index but its artifacts 404), providers locked at
`hashicorp/aws 6.64.0` and `hashicorp/archive 2.8.1`
(`infra/terraform/.terraform.lock.hcl` is committed).

```
$ cd infra/terraform && ../.tools/terraform apply -input=false -auto-approve
...
Apply complete! Resources: 11 added, 0 changed, 0 destroyed.
```

11 resources: S3 bucket + versioning + notification + lambda permission,
DynamoDB table + 2 GSIs + PITR, SQS DLQ + queue policy, IAM role + inline
policy, Lambda function + event-invoke config (on-failure destination,
0 retries). `terraform destroy` removes all 11.

## 4. End-to-end smoke test

```
$ cd infra && uv run python scripts/localstack_smoke.py
```

Full lifecycle: compose up → terraform apply → upload the unit-test fixture
batch to `detections/…` → poll DynamoDB → assert normalized row contents →
re-put the same key (new version) → assert exactly one marker per version
and zero duplicate detection rows → assert DLQ wiring (on-failure
destination, `maximum_retry_attempts = 0`) → upload a poison batch →
receive it from the DLQ → terraform destroy + `compose down --volumes`.

Verbatim output of the authoritative run (clean slate: fresh compose up,
fresh terraform apply; teardown at the end):

```
=== start LocalStack (docker compose up --wait) ===
[PASS] localstack container healthy — healthy

=== terraform apply ===
bucket=wfm-raw-media-dev table=wfm-detections-dev function=wfm-detection-ingest-dev dlq=http://[::1]:4566/000000000000/wfm-detection-ingest-dev-dlq

=== upload fixture batch s3://wfm-raw-media-dev/detections/cam-01/2026-09-16T10-15-00Z.json ===
[PASS] object stored with versionId — VersionId=AaCrRq7rtsU9NptPhrd2kJ_IZ7dNr1_T

=== poll DynamoDB gsi-source for the two normalized detections ===
[PASS] 2 detection rows via gsi-source — got 2
[PASS] pk = CAM#cam-01 — CAM#cam-01
[PASS] sk = <captured_at>|<det_id> — 2026-09-16T10:15:00.000Z|1117bf6bd5d0-000
[PASS] entity = detection
[PASS] species = zebra — {'S': 'zebra'}
[PASS] confidence = 0.5689 — 0.5689
[PASS] bbox normalized [0.2484, 0.2193, 0.6582, 0.8228] — [0.2484, 0.2193, 0.6582, 0.8228]
[PASS] gs1pk = SPECIES#zebra — {'S': 'SPECIES#zebra'}
[PASS] image_id = aw-00007

=== idempotency: re-upload same batch (new version) → no duplicates ===
[PASS] re-put produced new versionId — AaCrRq7rtsU9NptPhrd2kJ_IZ7dNr1_T → AaCrRq7sKwYDcDsL5Y5HDESIlCDlD0Zi
[PASS] one ingest marker per object version — markers=2
[PASS] still 2 detection rows after re-upload — got 2

=== DLQ wiring: on-failure destination, 0 retries ===
[PASS] on-failure destination = DLQ arn — arn:aws:sqs:us-east-1:000000000000:wfm-detection-ingest-dev-dlq
[PASS] maximum_retry_attempts = 0 — 0

=== poison batch → DLQ receives the failed event (detections/cam-01/poison.json) ===
[PASS] failed invocation landed in DLQ — 1 message(s)
[PASS] DLQ payload references the poison object — {'version': '1.0', 'timestamp': '2026-09-16T17:32:20.639Z', 'requestContext': {'requestId': '4a6f4a29-6ec3-436e-9d76-5f1…

=== teardown ===
stack destroyed, containers down

SMOKE: PASS — all checks green
```

21/21 checks passed; exit code 0. Wall-clock ≈ 4 min on a warm docker cache
(first-ever run adds ~3–4 min for the two image pulls, §2).

## 5. Environment quirks hit (and worked around in-repo)

1. **LocalStack 2026.x licensing** — see §2; pinned to 4.14.0.
2. **WSL2 mirrored networking** — connections from inside WSL2 to
   docker-published `127.0.0.1:<port>` never arrive (the IPv4 loopback is
   mirrored to the Windows host). `[::1]` works, so every endpoint
   (`aws_endpoint`, `AWS_ENDPOINT_URL`) defaults to `http://[::1]:4566`.
   Symptom if ignored: terraform/boto3 hang with ConnectTimeout while
   `curl` (Happy Eyeballs → `::1`) appears fine.
3. **docker userns-remap vs the Lambda executor** — two distinct failures:
   - the LocalStack container could not use `/var/run/docker.sock` (shows
     up as nobody:nogroup under remap) → `userns_mode: host` in compose;
   - execution-environment containers failed to start with an *empty*
     ContainerException, retrying forever. Root cause: LocalStack copies its
     lambda runtime (owned by uid 101000, its image build user) into the
     remapped env container; the daemon rejects the copy — "container ID
     101000 cannot be mapped to a host ID". Fix: init hook
     `scripts/localstack-init/01-fix-lambda-runtime-ownership.sh` re-owns
     the runtime to 0:0 at container start (always mappable). `--userns=host`
     via `LAMBDA_DOCKER_FLAGS` is not an option — LocalStack's flag parser
     rejects it.
4. **SQS queue URLs** — LocalStack hands out queue URLs on its canonical
   hostname (`localhost.localstack.cloud`, whose public wildcard DNS points
   at 127.0.0.1 — unreachable per quirk 2). The smoke script rewrites the
   netloc to its endpoint before calling `receive_message`.

## 6. Honest caveats

- Validated against LocalStack only, by design (no cloud credentials).
  LocalStack's DynamoDB/S3/SQS/Lambda are re-implementations; behavior
  parity for the transaction-cancellation code path was additionally covered
  by moto unit tests (including moto's quirk of returning the string
  `"None"` in `CancellationReasons` where AWS sends null).
- The IAM policy for CloudWatch logs uses `arn:aws:logs:*:*:*`; tighten to
  the real account on a first cloud deploy.
- `force_destroy = true` on the S3 bucket is a dev convenience for clean
  teardowns — remove for prod.
- The aws provider emits a deprecation notice for `hash_key`/`range_key` in
  `aws_dynamoxx_table` (6.64 still accepts them; the advertised
  `key_schema` block is not yet implemented — apply fails with
  "Unsupported block type"). Comment in `dynamodb.tf` documents this; the
  next provider major is the migration point.
- First Lambda invocation pays the runtime-image pull (~517 MB) unless the
  image is already cached; subsequent invocations reuse the warm
  environment.
