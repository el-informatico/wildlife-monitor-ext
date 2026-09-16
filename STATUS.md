# STATUS — wildlife-monitor-ext

This repo holds the extension work around the sprints-1-2 pipeline that
lives in the sibling `wildlife-monitor` repo (its STATUS.md tracks S1/S2
and classifier v0): untracked datasets/models/artifacts, the sprint-3
`infra/` ingest slice, and — as of 2026-09-16 — the `pipeline/` E2E
union stage documented below.

## infra — S3 → Lambda → DynamoDB ingest slice (2026-09-16)

Detection-batch contract (`infra/schema.py`, stdlib-only, v1), ingest
Lambda with unit tests, LocalStack compose + Terraform stack, and an
end-to-end smoke test. See `infra/README.md`.

## E2E union — detections → crops → species → contract records (2026-09-16)

Closes the pending item from the sprints-1-2 STATUS.md: one reproducible
stage runs the REAL detector over a deterministic sub-sample of the S2
ETL evaluation set, crops every deployed animal detection with the exact
recipe classifier v0 was trained on, classifies it, and emits normalized
records that validate against the ingest contract (v1). `infra/` was not
modified; `infra.schema` is imported read-only.

### What ran

    ../wildlife-monitor/.venv/bin/python -m pipeline.run_e2e
    ../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests -q

Default parameters: seed 42, strata `aw_with_gt=100, mo_with_gt=75,
mo_empty=25` (exactly 1/4 of the 800-image S2 strata), deploy conf 0.25,
decode floor 0.10, cv2.dnn graph engine `new`. Scale up with
`--sizes aw_with_gt=400,mo_with_gt=300,mo_empty=100` to cover the full
S2 sample (proportional wall time).

### Sample and outputs

- 200 images, 264 GT boxes, 0 decode failures, 15 images with zero
  deployed detections (the sample includes 25 GT-empty stratum images by
  construction).
- Deployed detections (conf ≥ 0.25): **350** — 339 animal / 10 person /
  1 vehicle. All 339 animal crops classified (none dropped by the 8 px
  minimum-edge rule).
- Outputs in `processed/` (prefix `e2e_union_v0`): `batches.jsonl` (200
  contract-valid batches, one per line), `records.parquet` (350 flat
  per-detection rows with species prob and GT match), `report.json`,
  `summary.md`.

### Sanity metrics

Detection recall vs GT (IoU ≥ 0.5, class-agnostic, S2 parity protocol —
computed by `wildlife_monitor.detect.parity.engine_metrics` and
cross-checked by this repo's independent matcher: **0.8068 vs 0.8068,
match**):

| conf | overall TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| 0.20 | 224 | 173 | 40 | 0.8485 | 0.5642 |
| 0.25 | 213 | 137 | 51 | **0.8068** | 0.6086 |
| 0.30 | 210 | 117 | 54 | 0.7955 | 0.6422 |

Consistent with S2's 0.7945 @ 0.25 over the full 800-image sample
(this is a different, smaller subsample). Per-dataset scopes live in
`processed/e2e_union_v0_summary.md`.

Classifier on GT-box crops (training-recipe-exact windows; accuracy by
split because train crops were seen in training):

| split | n | accuracy | note |
|---|---|---|---|
| test | 25 | 1.0000 | compare to classifier v0's 0.9867 (375 crops) — consistent, but only 25 evaluable crops here |
| train | 146 | 0.9932 | seen in training |
| val | 18 | 1.0000 | |

Species on deployed detections (animal dets matched to species-labelled
AW GT): **146/147 = 0.9932**; the single error is one buffalo predicted
elephant (confusion: buffalo 15/16, elephant 47/47, rhino 40/40,
zebra 44/44). Species counts over deployed animal records: elephant 162,
zebra 85, rhino 62, buffalo 30.

Latency (CPU, warmup 3 images excluded), p50 / p95 per image: decode
10.1 / 20.5 ms, letterbox+blob 8.3 / 18.8 ms, detector forward 124.2 /
151.7 ms, decode+NMS 1.1 / 1.6 ms — detector total 147.6 / 182.7 ms;
per crop: extract 0.17 / 0.29 ms, classify 9.6 / 12.3 ms. End-to-end
wall time **35.5 s** for 200 images (~0.18 s/image, CPU).

### Verification

- Offline unit tests: **23 passed** (crop-recipe equivalence against the
  trainer's verbatim arithmetic, window clamping/min-edge, sampling
  determinism and seed sensitivity, record assembly passing
  `infra.schema` validation incl. the species-only-on-animal guard,
  matcher agreement with `parity.match_image` over randomized cases,
  histogram/summary helpers).
- Fresh-process contract re-validation: all 200 jsonl lines and all 350
  detections pass `infra.schema` validate_batch + validate_detection.
- Determinism: a second run with identical parameters reproduced every
  metric and a byte-identical `batches.jsonl` (wall time differs — it is
  timing).

### Caveats

- CPU-only (`nvidia-smi` absent in this environment; `DNN_TARGET_CPU`,
  same as S2 and the classifier eval). No batching exists to degrade.
- `captured_at` is omitted: the manifest's `camera_ts` is timezone-naive
  and the contract requires a tz-aware ISO-8601 stamp. `camera_id` comes
  from the manifest `location_id` join; `source.key` follows the
  documented `raw/<camera>/<image_id><suffix>` layout.
- Crops are fed in-memory (BGR ndarray); training read from a JPEG
  quality-95 cache, so pixel values differ marginally from training
  crops. The measured GT-crop accuracy shows no visible impact.
- `confidence` in contract records is the detector's; the classifier
  probability travels only in the parquet rows (`species_prob`).
- Species accuracy is evaluable only on african_wildlife detections —
  the only dataset with species labels. The test-split GT-crop n is 25;
  treat 1.0000 as "consistent with v0", not an improvement claim.
- Recall is measured on the 200-image deterministic subsample, not the
  full S2 sample.

## Next

- Wire the producer side for real: run the union stage output through
  the LocalStack ingest path (`infra/`) end to end — jsonl batches are
  already the contract upload shape.
- Scale the union run to the full 800-image sample (or a regenerated
  larger sample) once AWS-side G1 work resumes.
