# STATUS — wildlife-monitor-ext

This repo holds the extension work around the sprints-1-2 pipeline that
lives in the sibling `wildlife-monitor` repo (its STATUS.md tracks S1/S2
and classifier v0): untracked datasets/models/artifacts, the sprint-3
`infra/` ingest slice, the `pipeline/` E2E union stage, and — as of
2026-09-16 — the `eval/` G3 perception evaluation documented below.

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

## G3 perception evaluation — metrics, day/night split, sequence logic, failure catalog (2026-09-16)

Sprint 3 / Gate G3: a reproducible evaluation harness over the S1/S2
datasets and artifacts, with held-out integrity enforced before any
number is produced. Full report with every table, split definition, and
caveat: `eval/G3-REPORT.md`; annotated failure cases:
`eval/failures/CATALOG.md` (7 cases with images). Day/night throughout
is the content-based IR flag (mean HSV saturation < 25); the manifest's
`is_night` is synthetic-random (agreement with content ≈ chance:
0.449 missouri / 0.516 african_wildlife) and is never used to split a
number.

### What ran

    ../wildlife-monitor/.venv/bin/python -m eval.run_eval
    ../wildlife-monitor/.venv/bin/python -m eval.run_failures
    ../wildlife-monitor/.venv/bin/python -m pytest eval/tests pipeline/tests -q

`run_eval` = 4 stages (motion gate, classifier, detection, sequence),
split-integrity gate first, deterministic (seeded, second full run
reproduced every number). Expensive passes cache under `processed/`
(`g3_motion.parquet`, `g3_seq_dets_*.json`); `--refresh` recomputes.
CPU-only (`nvidia-smi` absent — no VRAM to check, no batching to
degrade; documented in the report).

### Sample and outputs

- classifier v0: the trainer's own 375-crop AW **test** cache (227
  source images); 6-check integrity gate all-pass, crop windows
  recomputed from YOLO labels 375/375 integer-exact.
- detection: the S2 800-image sample (400 AW-with-GT / 300 MO-with-GT /
  100 MO-empty), MDv6, parity protocol IoU ≥ 0.5.
- sequence: 100 Missouri GT sequences (1,276 frames) + 115 AW test
  bursts (227 images; all pure-test).
- motion: full manifest 26,177 frames / 48 cameras, 0 decode failures.
- Outputs in `processed/` (prefix `g3_`): `report.json`, `metrics.csv`
  (tidy long), `metrics.md` (tables), `motion.parquet`, two seq-dets
  caches. Failure images are committed under `eval/failures/images/`.

### Sanity metrics

Classifier (AW test, n=375): top-1 **0.9867** micro / 0.9852 macro —
matches the trainer's torch reference exactly (375/375 top-1 agreement,
max |Δprob| 0.0001). Top-5 degenerate (4 classes). Per-class: rhino
weakest 0.9733 (n=75). Night n=5 — too small to read, stated not hidden.

Detection @ deploy conf 0.25 (parity protocol, own matcher ==
`detect.parity`: TP 812 / FP 676 / FN 210 identical): recall **0.7945**,
precision 0.5457, FP/img 0.845. Day/night: night recall HIGHER than day
(0.8491 vs 0.7754) — the cost is AW day close-up fragmentation, not IR.
missouri|night 0.8458 vs missouri|day 0.7843; AW|night nominal (12 GT
boxes). Sweep 0.20/0.30 in the report.

Sequence logic (MOG2 trigger + max-per-trigger, 215 GT sequences):
trigger observed **0.8884** (191/215; 18 AW no-gate sequences are
all-warmup bursts, 6 no-detection). Recall ladder union → gated →
selected: **0.9349 → 0.8093 → 0.7535**; selection loss 5.6%. Frame gate
rate 0.5963 → inference saving 0.4251. Species on selected frames
(pure-test bursts) **1.000 (n=75)** vs all test frames 0.993 (n=286) —
max-conf selection skews toward clean crops. Weak scope:
missouri|day selected 0.556 (n=27).

Motion gate full-manifest: overall gate rate 0.5636 (AW 1.0 — close-up
subjects always move; missouri day 0.4877 / night 0.5581).

Failure catalog (7 cases, images + numbers + reproducible keys): AW day
close-up fragmentation (best fragment IoU 0.414 → FN + 3 FP on one
frame); MO night edge-hugging blind spot (best floor IoU 0.000 — no
threshold recovers); tiny-crop upsample confusion (54×26 px → 224²,
rhino→elephant); frame-edge truncation (elephant→zebra at (0,0));
closed-set OOV (MO roe deer → "rhino" p=0.57); selection loss (0.81-conf
fragment outranked a 0.69-conf true match); MOG2 burst-start spike
(fg_ratio exactly 1.0, 21/22 full-fg frames on the camera are
first-frame-after-gap — stale background model).

### Verification

- Offline tests: **53 passed** (30 eval — micro/macro on synthetic
  vectors, split-integrity rejects (wrong split, dup ids, OOV class,
  count drift, train leak), parity no-fallback matcher, trigger/
  selection ladder, table emitters; 23 pipeline).
- Cross-checks inside the run: torch-vs-ONNX 375/375; own matcher vs
  `wildlife_monitor.detect.parity` identical TP/FP/FN; crop-window
  recipe 375/375 integer-exact vs the trainer's index.
- The gate-fp failure case replays MOG2 per-camera with the
  stage-motion parameters and asserts the replayed fg_ratio equals the
  cached parquet value (exact match).
- Determinism: second full `run_eval` reproduced every metric; sampling
  is seeded (seed 42, same as S2).

### Caveats

- AW night cells are tiny (5 classifier crops, 12 detection GT boxes) —
  reported, flagged, never aggregated into the night story, which is
  carried by Missouri (n=253 boxes / 73 GT sequences).
- Missouri carries no classifier metrics: species disjoint from v0's
  closed 4-species vocabulary (failure case 5 shows a roe deer scored
  "rhino" 0.57). CCT20 is metadata-only in this repo — excluded from
  every stage.
- Missouri sequence recall is box-level; GT annotates a subset of frames
  per sequence, so selected-recall there is a lower bound.
- Detection n is the S2 800-image sample, not full datasets; the AW det
  stratum spans train/val/test images (pretrained detector — no leak
  vector, documented).
- The motion pass uses CLAHE-on-content-IR (S1's bench keyed on the
  synthetic flag), so trigger rates are not comparable to S1 numbers.

## Next

- Detector post-step: fragment merge for close-up subjects (fixes the
  top AW day failure mode and its inherited selection loss) — measured
  as failure cases 1 and 6.
- Motion-gate hardening: two-consecutive-frames debounce or a
  global-change guard on fg ≳ 0.9 (failure case 7 — burst-start spikes).
- Classifier v0.1 candidates: min-resolution guard or aspect-preserving
  resize (case 3), and an OOV/none head or embedding threshold before
  any Missouri deployment (case 5).
- Wire the producer side for real: run the union stage output through
  the LocalStack ingest path (`infra/`) end to end — jsonl batches are
  already the contract upload shape.
- Scale the union run to the full 800-image sample (or a regenerated
  larger sample) once AWS-side G1 work resumes.
