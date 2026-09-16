# STATUS — wildlife-monitor-ext

This repo holds the extension work around the sprints-1-2 pipeline that
lives in the sibling `wildlife-monitor` repo (its STATUS.md tracks S1/S2
and classifier v0): untracked datasets/models/artifacts, the sprint-3
`infra/` ingest slice, the `pipeline/` E2E union stage, the `eval/` G3
perception evaluation and S5 final evaluation, the `agent/` S4 triage
agent, and the `dashboard/` S5 read-only ops endpoint, all documented
below.

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

## S4 triage agent — Gate G4, kill-switch B closed (2026-09-16)

Sprint 4: a Bedrock-style tool-calling agent over the perception output,
offline-first (deterministic reference backend behind a documented
adapter interface; the Bedrock adapter is a stub that refuses to run
without credentials). Full verdict with every table:
`agent/G4-REPORT.md`. Demo transcript + traces (committed):
`agent/demo/artifacts/`. Kill-switch B corpus comparison (committed):
`agent/artifacts/corpus_comparison.{json,md}`.

### What ran

    ../wildlife-monitor/.venv/bin/python -m pytest eval/tests pipeline/tests agent/tests -q
    ../wildlife-monitor/.venv/bin/python -m agent.run_corpus
    ../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo

### What it does

Per detection batch: consults four tools (`get_detection`, `query_zone`,
`query_hour`, `query_history` — specs carry the Bedrock toolConfig
shape) against an in-memory store, then one conditional action
(escalate / approval_queue / night_reanalysis / archive) with a
byte-deterministic trace (`wfm-triage-trace/v1`, `trace_id` =
sha256 of the trace content — no wall clock; uniform consult = exactly
9 steps). Loop safety: `MAX_STEPS` budget and any tool error both close
with the documented safe default (approval_queue) — a stuck agent can
never drop an event or auto-act on it. Person evidence always carries
`approval_required` + `face_blur_required` (the product approves, never
accuses).

### Gate G4: met — cv2.dnn output changes decisions, in trace

The demo's core sequence (s2, live pixels): pass-1 animal 0.3417 on a
content-IR night frame → `night_reanalysis` → a REAL second MDv6 pass
(grayscale → CLAHE(3.0, (8,8)) — the motion gate's recipe — decoded at
floor 0.10) finds four marginal animal boxes the deploy floor never
surfaced → the pass-2 event re-enters the loop → archive on the pass-2
output. Six further scenarios exhibit escalation (person ≥ 0.50 +
restricted zone + curfew), the approval queue (0.4102 below the bar;
0.5164 over the bar but open zone — the exact event the baseline
escalates blind), and a vehicle ranger check. Every fixture expectation
is asserted at run time; re-runs reproduce every artifact byte-identically.

### Kill-switch B: not triggered

`rules-only-v1` baseline (identical perception thresholds, NO context)
vs the agent over all 200 corpus events: 195 agree, the 5 divergences
are all confirmed persons in open zones midday that the baseline
escalates blind (5 highest-urgency pages for probable fragmentation
artifacts — G3 cases 1/3) while the agent queues with zone/clock
rationale. Zero person evidence is dropped in either lane. Night
re-analysis fires identically in both (9 events — shared perception
rule). The agent's measured value is urgency correctness + one
vigilance branch (vehicle + curfew, demo s6) — stated honestly in the
report: the winning lane is the deterministic backend; a future Bedrock
backend must beat it on this same corpus before its cost is justified.
Bedrock spend this sprint: $0 (§7 note in the report).

### Verification

- Offline tests: **132 passed** (30 eval + 23 pipeline + **79 agent** —
  tool dispatch/curfew math, every decision-table branch, tool-error
  and budget safe defaults, trace determinism, baseline head-to-head,
  scenario assembly validation).
- The demo asserts each scenario's expected decision at run time;
  corpus and demo artifacts verified byte-identical across re-runs
  (traces to `processed/`, gitignored).

### Caveats

- Zone policy (per-camera `zone_restricted` + fixed 18:00–06:00 window)
  and the clock (synthetic manifest `camera_ts` + fixed per-dataset
  offset) are SIMULATED context, tagged in every trace
  (`manifest-synthetic` / `pinned-demo`); demo scenarios s3/s6 pin
  documented contexts to exhibit branches absent from the corpus.
- Night is always the G3 content-IR flag, never the synthetic
  `is_night`. History on the corpus is the corpus itself (sparse — the
  repeat-signal branch never fired in the run; unit-tested, policy not
  measured behavior).
- Corpus events carry no timestamp — `camera_ts`/zone join via
  `image_id` → manifest `location_id` (verified consistent per camera
  across all 26,177 rows).

## S5 dashboard + final eval — Gate G5, met (2026-09-16)

Sprint 5: the read-only ops endpoint, the final evaluation tables, and
the latency/cost comparison. Full report with every table and the gate
checklist: `eval/S5-REPORT.md`; dashboard docs: `dashboard/README.md`.

### What ran

    ../wildlife-monitor/.venv/bin/python -m eval.run_final
    ../wildlife-monitor/.venv/bin/python -m eval.run_latency
    ../wildlife-monitor/.venv/bin/python -m dashboard            # 127.0.0.1:8787 (--host ::1 on WSL2 mirrored)
    ../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests eval/tests agent/tests dashboard/tests -q

### Dashboard

Stdlib-only `ThreadingHTTPServer` (no web dependency) over the local
artifacts: one page + five JSON routes + **one** image route. 48
cameras (schematic two-cluster map — no GPS exists in any metadata;
stated in the payload), deployed species counts (elephant 162 · zebra
85 · rhino 62 · buffalo 30), the 7 S4 approval-queue person events with
the committed G4 baseline cross-reference, and the 7 G3 failure cases
with images. Face blur is enforced on the single image-serving path:
stored person boxes ∪ a live MDv6 person pass at floor 0.10 at serve
time → destructive head-band blur (top 28% + 6%/side — no offline face
model exists; the cv2 wheel ships no Haar data); no-person images are
served byte-identical; the `X-Face-Blur` header always states the
outcome. Verified live (`applied=6;live` on a queue event, byte
identity + `none;live` on a no-person image, five 404 guards) and
locked by tests.

### Final eval

FP/camera/day over the S2 800-image sample (deploy 0.25, parity
protocol; TP/FP/FN cross-checked vs committed G3: exact match — 812/
676/210): 347 active camera×day cells, 99 at zero FP; FP per active
cell mean 1.95 / p95 6 / max 18. missouri 3.0989 vs african_wildlife
1.5391 FP/camera/day (day partition rides the synthetic ETL clock —
counts measured, partition simulated). **Wellington person P/R:
NOT_COMPUTABLE** — the dataset is absent and no local GT vocabulary
carries a `person` class (probe evidence in the report); substitute on
the available held-out: 48 person detections over 29 images / 18
cameras (load, not errors), all 7 corpus person events queued by the
agent.

### Latency/cost

Same-protocol three-lane table (800 images, warmup 3 excluded, CPU):
new engine (cv2.dnn new) e2e p50 **148.47 ms** → **$3.67/1M** on
c7i.large; legacy (PyTorch + Pytorch-Wildlife) 168.41 ms → $4.16/1M —
**+13%** legacy overhead; cv2 classic ablation in between. Fresh n=100
re-run confirms the new engine's lead (classic/PyTorch can swap at
n=100 — the 800-image table is the number of record). Prices are stated
assumptions (us-east-1
list, retrieved 2026-09-16), model = ms/3.6 sequential.
**Graviton: NOT MEASURED** (no AWS access; optional per the brief) —
at-parity arithmetic $2.99/1M + the documented on-box bench procedure
instead of a fabricated number.

### Verification

- Offline tests: **189 passed** (23 pipeline + 49 eval + 79 agent +
  **38 dashboard** — blur geometry/destructiveness, Store aggregation,
  image-resolution guards, live-HTTP endpoint tests on an ephemeral
  port incl. blur enforcement and 404s).
- Parity cross-check and the fresh-vs-stored latency deltas live inside
  the generated reports (`processed/s5_final_report.json`,
  `processed/s5_latency_report.json`).

### Gate G5: met

Endpoint deployed (local, read-only, blur always) + latency table (new
vs legacy, CPU; Graviton honest NOT MEASURED with procedure) + cost per
1M with assumptions + Wellington P/R reported as an honest
NOT_COMPUTABLE with the person-load substitute. Item-by-item checklist:
`eval/S5-REPORT.md` §6.

## Next

- Graviton latency for real: run the documented `--fresh` bench on a
  c7g.large (arm64 wheels, same first-100 images, warmup 3 excluded)
  once AWS access resumes — replaces the at-parity arithmetic in
  `processed/s5_latency.md`.
- Ingest a person-labelled held-out (Wellington or equivalent) to make
  person P/R computable — today's person numbers are load, not errors.
- Agent: exercise the Bedrock adapter lane once AWS credentials resume
  (offline stub documented in `agent/adapters.py`) — it must beat
  `rules-v1` on the committed corpus comparison before its cost is
  justified; revisit the person fragmentation band if the G3 fragment
  merge lands (it would collapse several s4/s5-style queue events).
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
