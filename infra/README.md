# infra — S3→Lambda→DynamoDB ingest slice (sprint 3)

IaC-only foundation for the OpenCV Wildlife Monitor: the EC2/OpenCV inference
node (sprints 1–2, unchanged) writes detection results as JSON "batch
sidecars"; this slice gives those results a durable, queryable home on AWS —
provisioned and verified entirely against LocalStack, with **no cloud
credentials anywhere in the repo**.

```
                                   infra/ (this directory)
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                                                                           │
 │  EC2 / OpenCV node (sprint 1–2 pipeline — NOT touched by sprint 3)        │
 │   MDv6-yolov10-c + classifier_v0 → detections per image                   │
 │        │ writes detection-batch sidecar (schema_version 1)                │
 │        ▼                                                                  │
 │  ┌────────────────────────┐   s3:ObjectCreated:* (prefix detections/,     │
 │  │ S3  wfm-raw-media-dev  │   suffix .json), versioning ENABLED           │
 │  │  raw/…   (media)       │──────────────┐                                │
 │  │  detections/…(batch    │              │ async invoke                   │
 │  │   sidecars, .json)     │              ▼                                │
 │  └────────────────────────┘  ┌─────────────────────────┐   on-failure    │
 │                              │ Lambda                  │──────────────┐   │
 │                              │ wfm-detection-ingest    │              │   │
 │                              │  app.py + schema.py     │              ▼   │
 │                              │  · validate (poison vs  │   ┌──────────────┐│
 │                              │    per-record policy)   │   │ SQS DLQ      ││
 │                              │  · normalize bbox 0–1   │   │ …-dlq (14d)  ││
 │                              │  · idempotent writes    │   └──────────────┘│
 │                              └───────────┬─────────────┘                   │
 │                                          │ TransactWriteItems             │
 │                                          ▼   (attribute_not_exists)       │
 │  ┌───────────────────────────────────────────────────────────────────┐    │
 │  │ DynamoDB wfm-detections-dev (single table, on-demand billing)     │    │
 │  │  pk=CAM#<camera>          sk=<captured_at>|<det_id>   (base)      │    │
 │  │  gsi-species gs1pk=SPECIES#<sp|label> gs1sk=sk       "what/where" │    │
 │  │  gsi-source  gs2pk=SRC#<media key>    gs2sk=sk       "per image"  │    │
 │  │  ingest marker pk=IDEMP#<sha256(bucket|key|version)> sk=BATCH     │    │
 │  └───────────────────────────────────────────────────────────────────┘    │
 └───────────────────────────────────────────────────────────────────────────┘
        Deployed by terraform/ against LocalStack 4.14.0 (docker-compose.yml)
```

## Why these pieces

- **S3 (versioned)** — the contract with the producer. Raw media lands under
  `raw/`, detection batches under `detections/`. Versioning makes idempotency
  airtight: every re-put is a new versionId, and ingest markers are keyed by
  `sha256(bucket|key|versionId-or-eTag)`.
- **Lambda (`handler/app.py`)** — receives `s3:ObjectCreated:*` events for
  `detections/*.json`, validates each batch against `schema.py` (vendored into
  the zip — the sprints 1–2 modules stay untouched), and writes one
  normalized DynamoDB item per detection: camera, timestamp, species/label,
  confidence, bbox `[x1,y1,x2,y2]` normalized to `[0,1]`, source key, model.
- **DynamoDB single table** — access patterns below. One table beats
  one-table-per-entity at this scale and keeps GSI fan-out cheap.
- **SQS DLQ** — async S3 invocations that raise (poison payloads) land in the
  DLQ via the function's on-failure destination with
  `maximum_retry_attempts = 0`, so a broken producer surfaces in seconds
  instead of after a retry storm.

### Access patterns → keys

| Question | Index | Key condition |
|---|---|---|
| Timeline of what camera X saw | base | `pk = CAM#<camera>` (sk sorts by time) |
| All zebras (or unlabeled animals) | gsi-species | `gs1pk = SPECIES#zebra` |
| Everything detected in image Y | gsi-source | `gs2pk = SRC#<media key>` |
| Was object version V already ingested? | base | `pk = IDEP#<sha256>` (ConsistentRead) |

### Error policy (deliberate, tested)

| Failure | Handling | Where it ends up |
|---|---|---|
| Not JSON / structural schema violation | raise `PoisonPayloadError` | DLQ (human looks) |
| One bad detection in an otherwise-valid batch | skip + WARNING log | 0 rows for it, rest ingested |
| Same event delivered twice / re-put with new version | marker + `attribute_not_exists` transaction guards | 0 duplicate rows |

Idempotency is two layers: a fast-path `GetItem` on the ingest marker keyed by
object version, and deterministic detection ids (`sha256(source_key)[:12]-NNN`)
written under `attribute_not_exists(pk) AND attribute_not_exists(sk)` in
chunked `TransactWriteItems` (60/chunk). A cancelled transaction whose
cancellation reasons are all `ConditionalCheckFailed` is treated as
"already ingested"; anything else (throttle, validation) propagates.

## Motion gate — Gate G1's AWS half

A second Lambda (`handler/motion_app.py`, `wfm-motion-gate-<env>`) runs the
project's MOG2 motion recipe **inside** the function, on the ETL's media:

```
  ETL manifest (parquet, read-only)
        │ scripts/export_manifest_json.py   (pipeline venv — pyarrow)
        ▼
  build/media-manifest.json
        │ scripts/g1_batch_load.py           (infra venv — boto3)
        │   · upload frames → raw/<rel_path>
        │   · write motion-jobs/<dataset>/<camera>.json   (one job per
        │     camera, frames in capture order — triggers the Lambda)
        ▼
  ┌────────────────────────┐  s3:ObjectCreated:* (prefix motion-jobs/)
  │ S3  wfm-raw-media-dev  │──────────────► Lambda wfm-motion-gate
  │  raw/…  motion-jobs/…  │                python3.12 + OpenCV 5 image
  └────────────────────────┘                decode→cap 1024→gray→CLAHE(night)
                                            →MOG2→fg_ratio→gate
        marker pk=IDEP#<sha256(bucket|key|version)> sk=MOTIONJOB
        motion row pk=CAM#<camera> sk=<ts>|<frame_id>  entity=motion
        (gsi-source populated; gsi-species sparse — motion rows skip gs1*)
```

- **Recipe parity** — history=50, varThreshold=24.0, detectShadows=False,
  learningRate=-1.0, CLAHE(3.0,(8,8)) on night frames only, first 2 frames
  per camera warmup, pass when fg_ratio > 0.02: mirrored from the local
  benchmark (constants at the top of `motion_app.py`), and the offline
  tests pin the boundary (0.02 exactly → filtered).
- **Idempotency** copies the ingest handler's contract: job marker keyed by
  object version short-circuits replays; rows carry deterministic
  `sk = <canonical ts>|sha256(source_key)[:12]` and are written under
  `attribute_not_exists` in chunked transactions, so a re-processed job
  cannot duplicate rows.
- **Failure policy** — poison job JSON raises `MotionJobError` → DLQ;
  undecodable/missing frames are skips (counted in `frames_failed`).
- **OpenCV 5 in the runtime** — the function's execution image is
  `wfm-lambda-py312-opencv5:g1` (Ubuntu 24.04 + the project's source-built
  OpenCV 5, harvested from the pipeline venv by `scripts/build-lambda-image.sh`).
  LocalStack routes python3.12 functions onto it via
  `LAMBDA_RUNTIME_IMAGE_MAPPING` (.env), and the `02-` init hook lets the
  daemon-local image be used. `cv2.__version__` is read inside the handler
  and stamped on every row as `model_runtime` — the import is the point of
  the gate, so it carries its own evidence. On real AWS the same handler
  deploys unchanged (zip + container image instead of the mapping).

Measured end-to-end over the African Wildlife dataset (1,504 images, 24
cameras): commands, raw outputs and the verdict live in
**[G1-AWS-REPORT.md](G1-AWS-REPORT.md)**.

## Why Terraform (and not SAM/CDK)

The repo is Python + uv, but that argues for boto3 in the *handler and tests*,
not for CDK: CDK would drag a Node.js toolchain and synthesized-CloudFormation
debugging into a repo that has none. SAM is CloudFormation-centric — its
`sam local` loop is nice but the deploy story still routes through CFN stacks
and its LocalStack support is second-hand. Terraform is one pinned static
binary (downloaded to the gitignored `infra/.tools/` by
`scripts/install-terraform.sh`), its `endpoints` overrides target LocalStack
first-class, and the same config deploys to real AWS by setting
`aws_endpoint = ""`. Handler/tests stay pure Python under uv; HCL is the only
new language, and it is declarative and reviewable.

## Layout

```
infra/
├── docker-compose.yml        # pinned LocalStack 4.14.0 + docker.sock + userns_mode
├── .env.example              # LocalStack endpoints/dummy creds (no secrets)
├── schema.py                 # detection-batch contract (stdlib only)
├── handler/app.py            # the ingest Lambda (stdlib + boto3)
├── handler/motion_app.py     # the motion-gate Lambda (MOG2 on the exec image)
├── terraform/                # provider/S3/DynamoDB/2×Lambda/SQS + outputs
├── tests/                    # 90 pytest tests (moto — LocalStack NOT required)
│   └── fixtures/detection_batch_sample.json   # real S2 output shape
├── scripts/
│   ├── install-terraform.sh  # pinned binary → infra/.tools/ (gitignored)
│   ├── build-lambda-image.sh # pipeline venv's OpenCV 5 → wfm-lambda-py312-opencv5:g1
│   ├── localstack-init/      # container init hooks (userns + local-image workarounds)
│   ├── localstack_smoke.py   # up → apply → upload → assert → DLQ → teardown
│   ├── export_manifest_json.py  # ETL parquet → JSON (pipeline venv; pyarrow)
│   ├── g1_plan.py            # pure planning: manifest rows → jobs/addresses
│   └── g1_batch_load.py      # G1 driver: media+jobs upload → wait → verify
├── VALIDATION.md             # measured results of every mandated check
├── G1-AWS-REPORT.md          # Gate G1 AWS half: the 1,504-image load, measured
└── pyproject.toml            # uv project: dev deps boto3/moto/pytest only
```

## Running it

```bash
cd infra
cp .env.example .env               # optional; defaults are LocalStack-shaped
./scripts/install-terraform.sh     # terraform 1.16.3 → infra/.tools/

uv run pytest -q                   # unit tests (moto, no docker needed)

uv run python scripts/localstack_smoke.py        # full loop, tears down after
uv run python scripts/localstack_smoke.py --keep # leave the stack running
```

Manual equivalent of the smoke test:

```bash
docker compose up -d --wait
cd terraform && ../.tools/terraform init && ../.tools/terraform apply
uv run python - <<'EOF'            # poke at the deployed stack
import boto3, pathlib, json
s3 = boto3.client("s3", region_name="us-east-1",
                  endpoint_url="http://[::1]:4566",
                  aws_access_key_id="test", aws_secret_access_key="test")
s3.put_object(Bucket="wfm-raw-media-dev",
              Key="detections/cam-01/2026-09-16T10-15-00Z.json",
              Body=pathlib.Path("tests/fixtures/detection_batch_sample.json").read_bytes())
EOF
# …then query DynamoDB (table wfm-detections-dev, gsi-source) or check the DLQ
```

`http://[::1]:4566` (IPv6 loopback) is deliberate: this dev box runs WSL2
mirrored networking, where connections to a docker-published `127.0.0.1:port`
from inside WSL never arrive, while `[::1]` works on any docker host that
publishes on `[::]`. Set `aws_endpoint`/`AWS_ENDPOINT_URL` to
`http://localhost:4566` if your host prefers plain IPv4.

### Running the G1 batch load (motion gate)

Prerequisites: the stack above deployed, `.env` sourcing the
`LAMBDA_RUNTIME_IMAGE_MAPPING` line (and the image built once by
`scripts/build-lambda-image.sh` — see its header). Then, from `infra/`:

```bash
set -a && source .env && set +a

# 1. manifest extract (needs pyarrow → the pipeline venv; parquet is read-only)
<path-to-pipeline-venv>/bin/python scripts/export_manifest_json.py \
    --parquet ../processed/manifest.parquet --out build/media-manifest.json

# 2. load + verify (infra venv: boto3)
.venv/bin/python scripts/g1_batch_load.py \
    --manifest build/media-manifest.json --dataset african_wildlife
```

Every phase is idempotent — rerunning skips already-uploaded media
(HEAD-first) and already-completed jobs (marker check). Phases can be
selected (`--phases wait,verify`) and a small smoke is
`--dataset african_wildlife --limit 5`. The driver writes its JSON metrics
to `build/g1-load-report.json`.

## Sprint mapping

- **Sprint 3 (this)** — ingest contract (schema.py), normalized store,
  idempotency, DLQ, LocalStack-verified IaC. No pipeline code touched.
- **Sprint 4** — EC2 node uploads raw media + sidecars into the bucket
  (producer side of the contract: `bbox_px_to_norm` in schema.py converts the
  pipeline's pixel boxes).
- **Sprint 5** — dashboards/query layer reads gsi-species / base-key
  timelines; new GSIs can be added without touching the handler.

## Caveats

- LocalStack is pinned to **4.14.0** (2026-02-26): the last release that runs
  without a LocalStack account auth token. 2026.x images refuse to start
  anonymously (distribution change enforced April 2026). To move to CalVer
  releases, set `LOCALSTACK_AUTH_TOKEN` in `.env` and bump the tag.
- Two workarounds exist purely for this dev host's docker daemon
  (`userns-remap: default`): `userns_mode: host` on the LocalStack container,
  and the `scripts/localstack-init/` hook that re-owns the lambda runtime to
  uid 0 so it can be copied into remapped execution-environment containers.
  Both are harmless on daemons without userns-remap.
- The IAM log-policy resource uses a wildcard account fragment
  (`arn:aws:logs:*:*:*`) — tighten it to the real account id on a first cloud
  deploy.
- `force_destroy = true` on the bucket is a dev convenience for clean
  teardowns; remove it for prod.
- handler/app.py is Python 3.12 / stdlib+boto3 only, so the zip stays two
  files and needs no dependency build step.
