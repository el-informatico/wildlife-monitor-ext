# Person/vehicle P/R — held-out evaluation (closes the S5 §4 honest gap)

S5 left person precision/recall as NOT_COMPUTABLE and substituted
person-event *load* (48 detections / 29 images — counts, not errors).
This stage closes that gap with measured error rates on two public
held-outs that actually carry person/vehicle ground truth. The deeper
Wellington finding is documented below: the gap was not download
bandwidth — the public labels have no person class at all.

Generated tables land in `processed/person_pr_tables.md` (+ the full
JSON with per-image predictions in `processed/person_pr_report.json`);
the headline tables are embedded verbatim below.

## Commands (verbatim)

    ../wildlife-monitor/.venv/bin/python -m eval.run_person_pr            # both held-outs, full run
    ../wildlife-monitor/.venv/bin/python -m eval.run_person_pr --max-per-set 50   # smoke pass
    ../wildlife-monitor/.venv/bin/python -m eval.run_person_pr --remake-tables    # rebuild tables from saved JSON, no inference
    ../wildlife-monitor/.venv/bin/python -m pytest eval/tests -q          # 9 runner + 4 sampler tests

## Why Wellington cannot compute person P/R (enumerated, not assumed)

- **The public labels have no person class.** The published
  `wellington_camera_traps.json` enumerates **17 categories: 15 animals
  + `empty` + `unclassifiable`** — no person, no vehicle — and
  annotates at image level only (no boxes). Person P/R is uncomputable
  from Wellington even with all 189 GB of images on disk: there is no
  ground truth to match against. (S5 §4 said "dataset absent"; the
  truth is stronger — the vocabulary itself lacks the class.)
- **Person-labelled camera-trap GT is structurally scarce.** LILA sets
  that once carried human classes removed those images for privacy.
  Measured on SWG 2018-2020's public no-species box file: **781
  person-annotated images, all 781 under `private/`, 0 public** (the 6
  vehicle-annotated images are public — an anecdote, not a metric).
  Idaho/WCS/ENA24 document the same removal.

## Held-outs used (metrics + scripts committed; images never are)

| held-out | domain | GT level | license | role |
|---|---|---|---|---|
| Seattle(ish) Camera Traps (lila.science) | camera trap (one yard, Seattle) | class (dan / vehicle / empty / animal) | CDLA-Permissive-1.0 | **in-domain** person/vehicle P/R, image level |
| COCO 2017 val (5,000 images) | consumer photos (domain shift, stated) | box (person; car/motorcycle/bus/truck) | annotations CC-BY 4.0 (images: Flickr terms) | multi-identity/scene **ceiling**, box level |

Seattle(ish) sample: deterministic, seed 17 (`eval/sample_seattle.py`
regenerates the spec + fetch list) — 800 `dan` + all 60 still `vehicle`
+ 500 `empty` + 380 animal-only images = **1,740 still images**. The
metadata's 4,464 video entries (`.avi`/`.mp4`) are excluded: a video's
class label does not guarantee the subject is visible in any single
frame, and this evaluation runs the still-frame detector path (the 19
vehicle-labelled videos are why the vehicle stratum is 60, not 79).
`dan` is the dataset's one human label — the author himself; images
containing other people were removed by the dataset publisher. **That
single-identity design is the honest limit of the in-domain number**: it
bounds the deployed detector's behavior on the person it can see in this
yard, not on arbitrary people. COCO val2017 answers the multi-identity
question out-of-domain.

## Protocol

- Detector: the deployed `MDV6-yolov10-c.onnx` via `cv2.dnn` (OpenCV 5
  graph engine) — letterbox → blob → forward → decode, the same path as
  G2/G3/S5. Every detection ≥ 0.10 is stored once; thresholds are
  swept offline. Deploy conf **0.25** stays the headline.
- Seattle tier (image level): an image counts as predicted-positive for
  a class if any detection of that class meets the threshold; GT is the
  class label. Images positive for the other class stay in the
  denominator (a person det on a vehicle-only image is an honest
  person FP). FP attribution by stratum is in the JSON.
- COCO tier (box level): class-restricted greedy matching at IoU ≥ 0.5 —
  the G3 parity protocol (`eval/metrics.greedy_match`; GT in COCO-native
  xywh, detections xyxy, score-ordered, each GT consumed once). Vehicle
  GT folds car/motorcycle/bus/truck; bicycle/train/airplane/boat are
  excluded (not in MDv6's vehicle vocabulary — folding them in would
  manufacture FNs). `iscrowd` annotations excluded.
- 9 offline tests (`eval/tests/test_run_person_pr.py`) pin the
  accumulators on hand-computed fixtures (threshold ≥ semantics,
  cross-class never matches, greedy-argmax rule, FP attribution). The
  fixtures caught two real counting bugs before any inference ran.

## Results

## Person/vehicle P/R — held-out tables (generated; embed verbatim)

### Seattle(ish) — camera-trap domain, image level (1740 images, seed-17 sample)

**Person**

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 728 | 83 | 72 | 0.8977 | 0.91 | 0.9038 |
| 0.15 | 719 | 64 | 81 | 0.9183 | 0.8988 | 0.9084 |
| 0.20 | 712 | 42 | 88 | 0.9443 | 0.89 | 0.9163 |
| 0.25 **← deploy** | 703 | 30 | 97 | 0.9591 | 0.8788 | 0.9172 |
| 0.30 | 698 | 24 | 102 | 0.9668 | 0.8725 | 0.9172 |
| 0.40 | 676 | 12 | 124 | 0.9826 | 0.845 | 0.9086 |
| 0.50 | 640 | 5 | 160 | 0.9922 | 0.8 | 0.8858 |

FP-by-stratum at each threshold in the JSON; single-identity (“dan”) limit applies.

**Vehicle**

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 0 | 94 | 60 | 0.0 | 0.0 | — |
| 0.15 | 0 | 70 | 60 | 0.0 | 0.0 | — |
| 0.20 | 0 | 52 | 60 | 0.0 | 0.0 | — |
| 0.25 **← deploy** | 0 | 40 | 60 | 0.0 | 0.0 | — |
| 0.30 | 0 | 32 | 60 | 0.0 | 0.0 | — |
| 0.40 | 0 | 17 | 60 | 0.0 | 0.0 | — |
| 0.50 | 0 | 8 | 60 | 0.0 | 0.0 | — |

TP = 0 at every threshold incl. the 0.10 storage floor — no vehicle detection fired on any vehicle-GT still (see Reading).

### COCO 2017 val — box level, greedy IoU ≥ 0.5 (5000 images)

**Person**

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 5344 | 11692 | 5433 | 0.3137 | 0.4959 | 0.3843 |
| 0.15 | 4825 | 7181 | 5952 | 0.4019 | 0.4477 | 0.4236 |
| 0.20 | 4454 | 4752 | 6323 | 0.4838 | 0.4133 | 0.4458 |
| 0.25 **← deploy** | 4126 | 3404 | 6651 | 0.5479 | 0.3829 | 0.4508 |
| 0.30 | 3843 | 2477 | 6934 | 0.6081 | 0.3566 | 0.4496 |
| 0.40 | 3302 | 1378 | 7475 | 0.7056 | 0.3064 | 0.4272 |
| 0.50 | 2821 | 736 | 7956 | 0.7931 | 0.2618 | 0.3936 |

GT = person boxes, iscrowd excluded.

**Vehicle**

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 645 | 6735 | 2337 | 0.0874 | 0.2163 | 0.1245 |
| 0.15 | 564 | 4773 | 2418 | 0.1057 | 0.1891 | 0.1356 |
| 0.20 | 506 | 3527 | 2476 | 0.1255 | 0.1697 | 0.1443 |
| 0.25 **← deploy** | 449 | 2747 | 2533 | 0.1405 | 0.1506 | 0.1454 |
| 0.30 | 405 | 2165 | 2577 | 0.1576 | 0.1358 | 0.1459 |
| 0.40 | 328 | 1356 | 2654 | 0.1948 | 0.11 | 0.1406 |
| 0.50 | 263 | 856 | 2719 | 0.235 | 0.0882 | 0.1283 |

Vehicle-mapped GT: car/motorcycle/bus/truck; iscrowd excluded.

## Reading the numbers

**Person, in-domain (Seattle, image level, deploy conf 0.25): P 0.9591 /
R 0.8788** (TP 703 / FP 30 / FN 97 over 800 dan-GT images). The FP
attribution answers *where* it is wrong: 29 of the 30 FPs fire on
`empty` frames and 1 on an animal frame — the failure mode is "person
on empty", not cross-class confusion. The sweep says the deployed 0.25
sits on the F1 plateau (0.9172 at both 0.25 and 0.30); precision above
0.98 is buyable but costs recall fast (R 0.80 at conf 0.50). **Bound**:
one identity (the dataset's author), one yard — this measures the
deployed detector on the person it can see there, not a population
claim.

**Vehicle, in-domain: measured zero.** TP = 0 at *every* threshold from
the 0.10 storage floor up — not a single vehicle-class detection fired
on any of the 60 vehicle-GT stills — while vehicle FPs accumulated
elsewhere (40 at 0.25, 27 of them on dan frames). Precision 0 / recall
0 on this held-out. Caveats that bound it: n = 60 stills, one backyard,
vehicles possibly partial at frame edges. But zero at the floor across
the whole sweep is a strong negative signal, not a threshold artifact:
the vehicle class does not transfer to this yard's vehicles, and
vehicle alerting should not be trusted in this deployment.

**Person, out-of-domain ceiling (COCO val2017, box level, 0.25): P
0.5479 / R 0.3829** (TP 4,126 / FP 3,404 / FN 6,651 of 10,777 person
boxes). Splitting the FPs from the stored predictions with the same
matcher: **2,821 of 3,404 (83%) sit on person-GT images** —
localization errors at IoU < 0.5 — and 583 fire on no-person-GT
images. Read: precision survives the domain shift far better than
recall; small, distant, or occluded people in consumer photos are
where recall dies.

**Vehicle, out-of-domain: P 0.1405 / R 0.1506** — consistent with the
in-domain zero. MDv6's vehicle vocabulary (trail-cam vehicles) does not
cover COCO street scenes either; the vehicle conclusion is the same in
both tiers.

**What this changes in the story.** The S5 person-event load (48
detections / 29 images / 18 cameras) now sits on measured error rates:
at deploy conf, a person alert in the camera-trap domain is right ~96%
of the time it fires and catches ~88% of (single-identity) person
frames. The identity diversity limit is quantified rather than hand-
waved: one person in-domain; the multi-identity answer costs recall
out-of-domain. Vehicle detections were never the alerting story — now
they are *measured* as unreliable. Reproduce everything with the
commands above; per-image predictions for independent re-analysis are
in `processed/person_pr_report.json`.
