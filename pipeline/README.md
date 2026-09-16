# E2E union stage (detections → crops → species → contract records)

One reproducible pipeline stage that closes the gap left after S2 and
classifier v0: run the REAL detector over a deterministic sub-sample of
the ETL evaluation set, crop each animal detection with the exact recipe
classifier v0 was trained on, classify it, and emit normalized
detection-batch records that validate against the ingest contract
(`../infra/schema.py`, imported read-only — this stage never modifies
`infra/`).

Measured results (recall, accuracies, timings) live in `STATUS.md` and in
the generated `processed/e2e_union_v0_summary.md` / `_report.json`.

## Layout

- `e2e.py` — pure glue (numpy-only): deterministic sampling, crop-window
  math (training-recipe-exact), detection↔GT matching with per-detection
  mapping, AW species-label reading, record assembly + contract
  validation. No cv2 / models / datasets needed to import it.
- `run_e2e.py` — the runner: loads the sample, runs MDv6 (cv2.dnn) →
  crop → classifier v0, measures sanity metrics, writes outputs.
- `tests/` — offline unit tests for the glue (no models or images).

## Run

Needs the sprints-1-2 venv (OpenCV 5, numpy, pyarrow, `wildlife_monitor`
importable) and the untracked artifacts in this repo (`models/*.onnx`,
`data/`, `processed/eval_sample_s2.json`, `processed/manifest.parquet`).
From this repo's root:

```bash
../wildlife-monitor/.venv/bin/python -m pipeline.run_e2e
```

Options:

- `--sizes aw_with_gt=100,mo_with_gt=75,mo_empty=25` — stratum counts
  (default draws 1/4 of the 800-image S2 strata; scale up to
  `400,300,100` to cover the full sample, or beyond by regenerating the
  sample with the S2 tooling).
- `--seed 42` — sampling seed (fixed default = reproducible).
- `--deploy-conf 0.25` — confidence an animal detection must reach to be
  deployed (cropped, classified, written as a record). Detections are
  decoded down to 0.10 and stored in-memory so the recall sweep
  (0.20/0.25/0.30) needs no re-run.
- `--engine new|classic|auto` — cv2.dnn graph engine (default `new`).
- `--out-prefix e2e_union_v0` — output filename prefix.

Outputs, under `processed/`:

| file | contents |
|---|---|
| `<prefix>_batches.jsonl` | one contract-valid detection batch per line — the exact upload shape the ingest Lambda consumes |
| `<prefix>_records.parquet` | flat per-detection analytics rows (species prob, GT match, inherited GT species) |
| `<prefix>_report.json` | full measured report: timings, recall sweep, accuracies, counts, model hashes |
| `<prefix>_summary.md` | human-readable summary tables |

Every batch line is validated against `infra.schema` (batch + each
detection) before it is written; a violation raises and fails the run.

## Tests

```bash
../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests -q
```

Pure-numpy logic runs anywhere; the matcher cross-check and crop-resize
test need `wildlife_monitor` / `cv2` and skip otherwise. The
training-recipe test compares crop windows against the verbatim
arithmetic of `train_classifier_v0.py`, so recipe drift fails loudly.

## Caveats

- **CPU-only.** No GPU is exposed in this environment (`nvidia-smi`
  absent); the stage runs on `DNN_TARGET_CPU`, same as S2 and the
  classifier eval. There is no batching to degrade — detection is
  per-image, classification per-crop.
- **`captured_at` omitted.** The manifest's `camera_ts` is timezone-naive;
  the contract requires an ISO-8601 timestamp with timezone, so the field
  is left out rather than fabricated. `camera_id` comes from the
  manifest's `location_id` join and `source.key` follows the documented
  `raw/<camera>/<image_id><suffix>` bucket layout.
- **Crops are fed in-memory** (BGR ndarray) while training read from a
  JPEG-quality-95 cache; pixel values can differ marginally from the
  training crops. Measured GT-crop accuracy below tests whether that
  matters in practice.
- **Species only where GT knows it.** Species accuracy is evaluable only
  on african_wildlife detections (the only dataset with species
  labels); missouri detections carry `label=animal` without species.
- **`confidence` is detector confidence.** Schema v1 has one confidence
  field; the classifier probability travels in the parquet analytics
  rows (`species_prob`), not in the contract records.
