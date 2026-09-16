# Gate G1 — AWS half: verification report

**Date:** 2026-09-16 · **Stack:** LocalStack 4.14.0 (docker-compose) + Terraform ·
**Dataset:** African Wildlife (ETL manifest, read-only)

## Verdict

**G1's AWS half is CLOSED — no degradation taken.**

> ≥1 dataset cargado end-to-end en DynamoDB vía Lambda con OpenCV 5 importado
> y MOG2 corriendo sobre ≥1.000 imágenes de prueba.

What ran: **1,504 real dataset images** (all of African Wildlife — 24 cameras,
53–74 frames each, 729 night frames) went ETL-manifest → S3 → S3 event →
Lambda → MOG2 (OpenCV 5.0.0 imported **inside** the function) → DynamoDB.
**1,504 of 1,504 expected motion rows** landed, 0 missing, 0 extra, 0 frames
failed, 0 DLQ messages, and both halves of the idempotency contract were
exercised live. The fallback (degrade to Missouri+AW / pre-Lambda motion)
was **not** needed.

The one number that needs honest context is the MOG2 filter pass-rate of
**1.0** — see [§7](#7-mog2-pass-rate-read-straight). Short version: the
metric is real, the recipe is exactly the local benchmark's, and this
dataset simply puts every non-warmup frame far above the 2% threshold.

---

## 1. The path that ran

```
processed/manifest.parquet (ETL output, READ-ONLY)
   │ scripts/export_manifest_json.py        (pipeline venv — pyarrow)
   ▼ build/media-manifest.json  (1,504 AW rows + Missouri, sorted)
   │ scripts/g1_batch_load.py               (infra venv — boto3)
   │   media phase:  1,504 images → s3://wfm-raw-media-dev/raw/<rel_path>
   │   jobs phase:   24 × motion-jobs/african_wildlife/<camera>.json
   ▼                    (each put fires s3:ObjectCreated:* → Lambda)
Lambda wfm-motion-gate-dev  (python3.12, exec image wfm-lambda-py312-opencv5:g1)
   │  per frame: S3 GET → decode → cap 1024 (INTER_AREA) → gray
   │             → CLAHE(3.0,(8,8)) if night → MOG2 → fg_ratio → gate
   ▼  TransactWriteItems (chunked, attribute_not_exists)
DynamoDB wfm-detections-dev
   motion rows:  pk=CAM#<camera>  sk=<canon ts>|<frame_id>  entity=motion
   job markers:  pk=IDEP#<sha256(bucket|key|version)>       sk=MOTIONJOB
```

New pieces (all under `infra/`, nothing outside touched):

| Piece | What it is |
|---|---|
| `handler/motion_app.py` | the motion-gate Lambda; recipe constants mirrored from the local benchmark and pinned by tests |
| `terraform/{main,lambda,s3,variables}.tf` | second function + permission + event-invoke config (0 retries, on-failure → existing DLQ); one bucket notification now routes both prefixes |
| `scripts/build-lambda-image.sh` | harvests the pipeline venv's source-built OpenCV 5 into `wfm-lambda-py312-opencv5:g1` (Ubuntu 24.04) |
| `scripts/export_manifest_json.py` | parquet → JSON extract (only pyarrow consumer) |
| `scripts/g1_plan.py` | pure planning: manifest rows → per-camera jobs, expected DynamoDB addresses |
| `scripts/g1_batch_load.py` | the driver: media/jobs/wait/replay/verify phases, all idempotent |
| `tests/test_g1_plan.py`, `tests/test_motion_handler.py` | 27 new offline tests |

## 2. OpenCV 5 inside the Lambda — the hard requirement

Layerless Lambda python3.12 has no cv2 and OpenCV 5 has no manylinux wheels
(the project builds it from source), so the function runs on a **custom
execution image**: `scripts/build-lambda-image.sh` copies the pipeline
venv's cv2 5.0.0 binding + `libopencv_*.so.5.0.0` onto an Ubuntu 24.04
python3.12 base (rewriting the cmake-baked absolute paths to portable
ones). LocalStack routes python3.12 functions onto it via
`LAMBDA_RUNTIME_IMAGE_MAPPING` (`.env`), and the `02-` init hook lets the
daemon-local image be referenced.

Evidence that this is cv2 *inside the function*, not asserted from outside:

- every invocation logs, before any work:
  `{"msg": "invocation start", ..., "cv2_version": "5.0.0"}` — read from
  `cv2.__version__` inside the handler (`_load_engine()`);
- every one of the 1,504 rows carries
  `model_runtime = "opencv-5.0.0+aws-lambda-python3.12"`, built from that
  same in-handler `cv2.__version__`;
- per-frame timings (`ms_decode_mean`, `ms_mog2_mean`) come from actually
  running `imdecode`/`createBackgroundSubtractorMOG2` per frame.

**Fallback ladder (documented, not taken):**

1. ✅ MOG2 inside the Lambda on the custom image — worked end-to-end.
2. Plan B (unused): same handler, OpenCV packaged as a Lambda layer.
3. Plan C (unused, would have weakened the gate): compute motion outside
   the AWS path and only persist results in the Lambda.

**Local parity check** (recorded before the full run): the 5 smoke frames
processed outside the Lambda with the pipeline venv's OpenCV 5.0.0 and the
identical constants produced the same fg_ratios to 6 decimals as the
Lambda's rows for those frames (all `1.000000` — see §7 for why that is
the correct value, not a bug). Same OpenCV, same recipe, same numbers,
inside and outside.

## 3. Exact commands

From `infra/` (paths repo-relative; `<pipeline-venv>` = the sprints 1–2
pipeline virtualenv, python 3.12 with the source-built OpenCV 5):

```bash
# one-time: custom runtime image + stack up (see README "Running it")
WFM_PIPELINE_VENV=<pipeline-venv> ./scripts/build-lambda-image.sh
# .env: uncomment LAMBDA_RUNTIME_IMAGE_MAPPING={"python3.12": "wfm-lambda-py312-opencv5:g1"}
docker compose up -d --wait
cd terraform && ../.tools/terraform init && ../.tools/terraform apply && cd ..

set -a && source .env && set +a          # [::1]:4566 endpoint, dummy creds

# 1. manifest extract (parquet opened READ-ONLY; nothing under processed/ is written)
<pipeline-venv>/bin/python scripts/export_manifest_json.py \
    --parquet ../processed/manifest.parquet --out build/media-manifest.json

# 2. 5-image smoke (validates the whole loop cheaply)
.venv/bin/python scripts/g1_batch_load.py \
    --manifest build/media-manifest.json --dataset african_wildlife \
    --limit 5 --report-out build/g1-smoke-report.json

# 3. the full load (this is the run §4–§7 report on)
.venv/bin/python scripts/g1_batch_load.py \
    --manifest build/media-manifest.json --dataset african_wildlife \
    2>&1 | tee build/g1-load-run.log
```

Repro caveat, learned the hard way: the smoke's replay phase re-puts L00's
job as a new version, so its completion marker makes the full run's jobs
phase skip L00 (5 of 1,504 frames). Before the full run those leftovers
were cleaned (7 marker/motion items for L00 deleted, job object purged of
all versions). If you smoke first, do the same — or smoke against a
different camera via `--cameras`.

Verification side-checks run afterwards (infra venv, same env):

```bash
.venv/bin/python - <<'EOF'   # scan + fg_ratio distribution over all rows
import boto3, statistics
d = boto3.client("dynamodb", endpoint_url="http://[::1]:4566")
...  # full-loop scan FilterExpression entity=:m collecting gate + fg_ratio
EOF
# plus CloudWatch Logs queries on /aws/lambda/wfm-motion-gate-dev
# (filterPattern '"invocation start"' and '"motion job done"')
```

## 4. Real outputs — the run, verbatim

`build/g1-load-run.log` (stdout of command 3 above):

```
plan: 1504 rows -> 24 jobs (1504 expected motion rows)
media: {'planned': 1504, 'uploaded': 1499, 'skipped_existing': 5, 'missing_local': 0, 'bytes': 104780436, 'wall_s': 6.7, 'imgs_per_min': 13424.3}
jobs: uploaded=24 skipped_complete=0
wait: {'markers_found': 24, 'rows': 1504, 'timed_out': False, 'stable': True, 'wall_s': 20.5} -> {'frames': 1504, 'frames_per_min': 4407.8}
replay: {"marker_short_circuit": {"function_error": null, "duplicate": true, "rows_written": 0}, "new_version_reput": {"marker_v2_found": true, "rows_before": 1504, "rows_after": 1504, "unchanged": true}}
verify: expected=1504 actual=1504 gates={'warmup': 48, 'passed': 1456, 'filtered': 0} pass_rate=1.0 dlq=0
report -> build/g1-load-report.json
```

(`skipped_existing: 5` = the smoke's five images, already in the bucket —
the HEAD-first skip doing its job; they were still processed as part of
L00's job.) Full JSON metrics: `build/g1-load-report.json` (selection,
media, wait, replay, verify blocks, `wall_s_total: 33.2`).

Lambda's own logs (CloudWatch Logs via LocalStack, verbatim):

```
{"ts": "2026-09-16T18:28:07.691+00:00", "level": "INFO", "msg": "invocation start", "request_id": "d4bc1cbc-635e-41e5-a133-afebe55c024a", "records": 1, "cv2_version": "5.0.0"}
{"ts": "2026-09-16T18:29:43.184+00:00", ... "msg": "invocation done", "results": [{"bucket": "wfm-raw-media-dev", "key": "motion-jobs/african_wildlife/african_wildlife-L20.json", "duplicate": false, "frames_total": 67, "frames_processed": 67, "frames_failed": 0, "gates": {"warmup": 2, ...
REPORT RequestId: 4b05f39f-42cc-4c5f-9fa5-ed039afe22ff	Duration: 2868.97 ms	Billed Duration: 2869 ms	Memory Size: 1024 MB	Max Memory Used: 1024 MB
```

Aggregated across all 25 `motion job done` lines (24 full-run jobs + the
smoke's L00): **frames=1,509, failed=0**, `ms_decode_mean` 2.6–13.2,
`ms_mog2_mean` 2.1–15.6.

## 5. Verification detail

### DynamoDB scan vs source count

| Check | Result |
|---|---|
| Manifest rows selected (AW) | 1,504 |
| Expected motion-row addresses (re-derived from plan) | 1,504 |
| Actual `entity=motion` rows (full scan) | **1,504** |
| Missing / extra addresses | **0 / 0** |
| Job completion markers | 24 / 24 |
| Frames failed (undecodable + missing) | 0 |
| DLQ messages | 0 |

### Spot-checks (driver-picked: first / middle / last expected row)

All 3 found, `source_key` matches the manifest-derived expectation, each
carries `model_runtime = opencv-5.0.0+aws-lambda-python3.12`
(`opencv_evidence: true`). One of them against its **manifest source row**,
verbatim side-by-side:

```
manifest (build/media-manifest.json)          DynamoDB (get_item, ConsistentRead)
──────────────────────────────────────       ──────────────────────────────────────
image_id:      aw-01234                      image_id:    aw-01234
location_id:   african_wildlife-L11          camera_id:   african_wildlife-L11   pk=CAM#african_wildlife-L11
camera_ts:     2026-03-14T12:00:00           captured_at: 2026-03-14T12:00:00.000Z
rel_path:      african-wildlife/images/      source_key:  raw/african-wildlife/images/train/4 (46).jpg
               train/4 (46).jpg              gs2pk:       SRC#raw/african-wildlife/images/train/4 (46).jpg
is_night:      false                         is_night:    false
                                              gate: passed  warmup: false  fg_ratio: 1
                                              frame_id: f98f8a49270a  (= sha256(source_key)[:12])
                                              width: 640  height: 480
                                              model_name: MOG2  model_runtime: opencv-5.0.0+aws-lambda-python3.12
```

Every field traces back to the source row; the timestamp is the
canonicalized UTC form the sort key requires.

### Idempotency (both halves, live)

1. **Marker short-circuit** — the completed L23 job's event, re-invoked
   directly with the *same* versionId: `duplicate: true, rows_written: 0`
   (the GetItem fast path; no frame was ever fetched).
2. **Row-level collisions** — the same job re-put as a *new* version (new
   marker key, so the fast path misses): the Lambda reprocesses all frames,
   every row Put rejects on `attribute_not_exists(pk/sk)`, the cancelled
   chunk-0 transaction is classified all-`ConditionalCheckFailed` →
   treated as replay: `rows_before: 1504 → rows_after: 1504, unchanged:
   true`, and the new version's marker still lands.

## 6. Throughput and latency

| Measurement | Value | What it actually measures |
|---|---|---|
| Media upload | 13,424 imgs/min (104.8 MB, 6.7 s) | LocalStack in-process HTTP — **not** an AWS number |
| End-to-end processing | 4,407.8 frames/min (1,504 frames / 20.5 s) | LocalStack spawning function containers + 2 s poll granularity + 8 s stability window |
| Per job (53–67 frames, 1024 MB) | 580–2,869 ms per REPORT line | one invocation, decode+MOG2+writes |
| Per frame (in-function means) | decode 2.6–13.2 ms · MOG2 2.1–15.6 ms | the OpenCV work itself |

The honest framing: on LocalStack the wall-clock numbers demonstrate the
loop works at this scale, not AWS capacity. The portable numbers are the
per-frame means — single-digit-to-low-teens ms per frame for OpenCV decode
+ MOG2 at ≤1024 px on one vCPU-equivalent, i.e. one 2,000-frame camera-job
sits comfortably inside the 900 s timeout.

## 7. MOG2 pass-rate — read straight

`gates = {warmup: 48, passed: 1,456, filtered: 0}` →
`pass_rate_non_warmup = 1.0`. Facts, not spin:

- 48 warmup = 24 cameras × 2 first frames, exactly as the recipe dictates.
- fg_ratio distribution over the 1,456 non-warmup rows: **min 0.6177,
  p10 1.0, median 1.0** — the closest any frame comes to the threshold is
  ~31× above it. Nothing is near 0.02; the filter had nothing to filter.
- Why: African Wildlife's "cameras" are ETL location groups over a test
  corpus, not contiguous trap sequences — background changes frame to
  frame, and MOG2 (α = 1/50 via learningRate=-1) flags essentially the
  whole frame. The parity check (§2) confirms the same values with the
  local OpenCV outside the Lambda, so this is the recipe's true behaviour
  on this data, not an integration artifact.
- The rejection path itself is real and pinned by tests:
  `gate_for` is unit-tested at the boundary (fg_ratio = 0.02 exactly →
  `filtered`; 0.0200001 → `passed`), and the moto tests drive scripted
  ratios through the full handler producing `filtered` rows.

If a dataset where the filter actually bites is wanted, contiguous
camera-trap sequences (Missouri's timelapse clips) are the natural
follow-up; the gate text did not require it and AW alone satisfies the
≥1,000-image bar.

## 8. Failures

Zero, end to end: `frames_failed = 0` on all 24 jobs, `missing_local = 0`,
DLQ empty, no function errors in any invocation. The failure *paths*
(poison job → `MotionJobError` → DLQ; undecodable/missing frame → skip +
count) are exercised in `tests/test_motion_handler.py` against moto.

## 9. Test suite

`uv run pytest -q` (or `.venv/bin/python -m pytest -q`): **90 passed** —
63 pre-existing + 27 new (10 motion-handler cases incl. 3 poison and 7
validation params, 8 planning-parity cases, …). No cv2 required: a
FakeEngine scripts the ratios so the tests exercise routing, gates, skips,
chunking and both idempotency halves against moto.

## 10. Artifacts

- `build/g1-load-report.json` — machine-readable metrics (gitignored)
- `build/g1-load-run.log` — run stdout (gitignored)
- `build/media-manifest.json` — the extract the driver consumed (gitignored)
- This report + `README.md` ("Motion gate") are the committed record.
