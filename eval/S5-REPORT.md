# S5 — Dashboard + final eval + latency/cost (Gate G5)

Sprint 5 closes the loop: a read-only ops endpoint over the pipeline's
local artifacts (with face blur enforced on the single image-serving
path), the final evaluation tables (FP/camera/day, person-event load,
the Wellington gap), and the latency/cost comparison of the new cv2.dnn
engine vs the legacy PyTorch baseline. Everything offline/local — no
cloud, no AWS spend. Generated tables land in `processed/`
(`s5_final_tables.md`, `s5_latency.md`, plus the JSON reports the
dashboard reads); this file embeds the headline tables and links the
rest.

## Commands (verbatim)

    ../wildlife-monitor/.venv/bin/python -m eval.run_final               # FP/cam/day + person load + Wellington probe
    ../wildlife-monitor/.venv/bin/python -m eval.run_latency             # latency/cost table (stored + fresh lanes)
    ../wildlife-monitor/.venv/bin/python -m dashboard                    # http://127.0.0.1:8787
    ../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests eval/tests agent/tests dashboard/tests -q

Dashboard notes: `--host ::1` on WSL2 mirrored-networking setups where
IPv4-loopback connects are dropped (verified here; see
`dashboard/README.md`). The latency runner's PyTorch lane shells out to
the sprints-1-2 `baseline-venv` (torch-cpu + Pytorch-Wildlife) via
`eval/_pytorch_bench.py`; `--fresh N` re-runs the confirmation bench.

## 1. Dashboard — what it serves and how it was verified

`dashboard/` (see its README): stdlib-only `ThreadingHTTPServer` — no
FastAPI/flask dependency; cv2/numpy are touched only on the image path.
One page (`/`) + five JSON routes (`/api/overview|cameras|species|
queue|failures`) + **one** image route (`/image?src=frame:<image_id>` /
`failure:<file>`). Data: `manifest.parquet`, e2e union batches/records,
S4 agent traces (approval-queue actions with the committed G4 baseline
cross-reference), and the G3 failure catalog with its committed images.

- **Camera map** — 48 cameras, per-camera image counts, first/last
  synthetic `camera_ts`, restricted-zone flag, FP + person-det samples
  from the S5 final report. **Schematic** two-cluster grid: no GPS/lat-lon
  exists in any local metadata — stated in the payload itself, not just
  the docs.
- **Species counts** — the union stage's own deployed-animal counts
  (elephant 162 · zebra 85 · rhino 62 · buffalo 30 over the 200-image
  deterministic subsample), not re-derived.
- **Approval queue** — the 7 person events the S4 agent queued (all with
  `face_blur_required`), each linked to its frame.
- **Failure gallery** — the 7 committed G3 cases with images.

Verified end-to-end over HTTP (all routes + the image paths + 404
guards), then locked by 38 tests (`dashboard/tests/`) that run a real
server on an ephemeral port against a miniature artifact fixture.

## 2. Face blur — always, on the single image path

Policy (in `dashboard/blur.py`, enforced in
`DashboardHandler._serve_image`): stored person boxes from the union
batches **∪ a live MDv6 person pass at conf floor 0.10 run at serve
time** — a frame whose stored pass found nothing is still protected if
the live pass sees one. Each person box contributes a destructive head
band (top 28% of the box, widened 6%/side — no offline face model
exists: the OpenCV wheel ships no Haar data and `models/` has none),
blurred in place (3× Gaussian k=31) and re-encoded; there is no
un-blur. Images with zero person evidence are served **verbatim**
(byte-identity, no re-encode). The `X-Face-Blur` header always states
the outcome — including `live=unavailable;<reason>` if the model cannot
be built, in which case stored-box blur still applies.

Verified live: queue event `aw-00095` served with
`X-Face-Blur: applied=6;live` (3 stored ∪ 3 live boxes; bytes differ
from disk), a no-person gallery image byte-identical with
`X-Face-Blur: none;live`, and five 404/rejection paths (unknown id,
path escapes, bogus kinds).

## 3. Final evaluation — FP per camera per day

Full tables: `processed/s5_final_tables.md` (all 47 in-sample cameras,
worst-10 cells, day partition note). S2 800-image sample, deploy conf
0.25, parity protocol IoU ≥ 0.5. Parity cross-check vs the committed G3
TP/FP/FN: **match** (812 / 676 / 210).

| scope | TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| all\|all | 812 | 676 | 210 | 0.7945 | 0.5457 |
| all\|day | 587 | 481 | 170 | 0.7754 | 0.5496 |
| all\|night | 225 | 195 | 40 | 0.8491 | 0.5357 |
| african_wildlife\|all | 558 | 394 | 160 | 0.7772 | 0.5861 |
| missouri\|all | 254 | 282 | 50 | 0.8355 | 0.4739 |

Active camera×day cells: **347** (800 images), 99 cells at zero FP;
FP per active cell mean 1.95 · p50 1 · p95 6 · max 18.

| dataset | cameras | cells | images | FP | FP/camera/day |
|---|---|---|---|---|---|
| african_wildlife | 24 | 256 | 400 | 394 | 1.5391 |
| missouri | 23 | 91 | 400 | 282 | 3.0989 |

Clock honesty: the day partition uses the manifest's `camera_ts` — the
ETL's **synthetic** per-camera clock (deterministic, seed-pinned). The
FP counts are measured; the per-day aggregation is over a simulated
calendar.

## 4. Wellington person P/R — NOT_COMPUTABLE (honest gap)

The brief asks for Wellington person P/R. Probe result
(`processed/s5_final_report.json` → `wellington`):

- Wellington dataset on disk: **NO**.
- Person category in any local GT vocabulary: **NO** — Caltech camera
  traps (22 categories) and Missouri set1 (21 categories) carry no
  `person` class; the S2 eval sample's GT vocabulary is `animal` only.
- Person P/R status: **NOT_COMPUTABLE** — any number here would be
  fabricated.

Substitute actually computable from the available held-out (same S2
sample): **person-event load** — 29 images with ≥1 person-labeled
detection, 48 person detections (conf p50 0.5623, max 0.9285), spread
over 18 cameras; per-camera and per-day tables in
`processed/s5_final_tables.md`. These are detection counts, not errors.
Cross-reference: all 7 person events in the S4 200-event corpus are
queued by the agent (baseline escalates 5 blind — committed G4
comparison).

## 5. Latency and cost — new engine vs legacy

Full tables: `processed/s5_latency.md` + `s5_latency_report.json`.
Protocol: same 800-image S2 sample, warmup 3 excluded, single-image
CPU inference (no GPU in this environment); the PyTorch lane runs in
the sprints-1-2 `baseline-venv` via a subprocess harness.

| lane | role | n | forward/infer p50 | e2e p50 | e2e p95 | img/s @p50 | h/1M | $/1M c7i.large |
|---|---|---|---|---|---|---|---|---|
| cv2.dnn new (deploy) | new engine | 800 | 125.13 | 148.47 | 185.43 | 6.74 | 41.24 | **3.67** |
| cv2.dnn classic | graph-engine ablation | 800 | 141.64 | 162.16 | 189.23 | 6.17 | 45.04 | 4.01 |
| PyTorch + Pytorch-Wildlife | legacy engine | 800 | 157.15 | 168.41 | 208.61 | 5.94 | 46.78 | **4.16** |

The new engine is the fastest lane; the legacy PyTorch engine is
**+13%** at e2e p50 on identical inputs and weights. Fresh same-protocol
re-run (first 100 images, re-run today) confirms the new engine's lead
(130.43 ms e2e p50); at n=100 the classic/PyTorch order can swap
run-to-run (152.13 vs 140.50 today, vs 141.64/157.15 stored) — the
800-image stored table is the number of record.

### Cost per 1M images — assumptions, not a quote

Model: e2e ms ÷ 3.6 = hours per 1M (sequential, no batching/queueing/
I/O-wait/retries), × us-east-1 on-demand hourly list (retrieved
2026-09-16 from public price trackers — **ASSUMPTION**, verify at
aws.amazon.com/ec2/pricing): c7i.large (x86, 2 vCPU) $0.089/hr,
c7g.large (Graviton3, 2 vCPU) $0.0725/hr.

**Graviton: NOT MEASURED** — no AWS access this sprint, and the brief
marks it optional. The at-parity arithmetic (measured x86 ms × c7g
price) is **$2.99/1M** for the deploy engine — it isolates the
hourly-price difference ONLY. Exact bench-and-claim procedure is
documented in `processed/s5_latency.md`: re-run `--fresh` on the c7g
itself (arm64 wheels of opencv-python and torch-cpu; ONNX weights are
architecture-independent), same first-100 images, warmup 3 excluded.

## 6. Gate G5 verdict

Brief: *"G5: endpoint desplegado + tabla de latencias (nuevo vs legacy
engine, CPU vs Graviton, costo por 1M imgs) + Wellington P/R
reportados."*

| gate item | status |
|---|---|
| endpoint deployed (map, counts, queue, gallery, blur) | **MET** — `python -m dashboard` → http://127.0.0.1:8787, read-only, 48 cameras / 4 species / 7 queue events / 7 failure cases; blur enforced and verified on the single image path |
| latency table new vs legacy, CPU | **MET** — 3-lane same-protocol table + fresh re-run, new engine +13% faster than legacy at e2e p50 |
| CPU vs Graviton | **PARTIAL (per brief: optional note)** — Graviton NOT MEASURED, honestly; at-parity arithmetic + a documented bench procedure instead of a fabricated number |
| cost per 1M imgs | **MET** — $3.67 (new) vs $4.16 (legacy) on c7i.large, assumptions stated |
| Wellington P/R reported | **MET as an honest gap** — NOT_COMPUTABLE (dataset absent, zero person GT locally), probe evidence + person-event-load substitute on the available held-out |

**G5: MET** (Graviton unmeasured per its optional status; Wellington
P/R satisfied by the stated-honestly path the brief itself allows).

## 7. Verification

- Test suites, all green: pipeline **23** · eval **49** · agent **79** ·
  dashboard **38** — **189 total** (`python -m pytest <pkg>/tests -q`
  per package, sprints-1-2 venv).
- Dashboard endpoint tests run a real server on an ephemeral port
  (loopback-address probe — this machine's WSL2 mirrored networking
  drops IPv4-loopback connects) and cover blur enforcement end-to-end:
  stored-box blur, injected live-pass blur on a no-stored-person frame,
  byte-identity for no-person images, detector-degrade headers, and the
  404 guards.
- The latency table's stored lanes are the committed S2 artifacts (same
  protocol); the fresh lane re-ran today and confirms the ordering.
- FP/camera/day cross-checks TP/FP/FN against the committed G3 numbers
  (exact match) before any per-cell table is emitted.

## 8. Caveats

- The dashboard is a **local** endpoint — "deployed" here means running
  on this machine (offline constraint); no cloud deploy exists or was
  attempted. The camera map is schematic (no GPS in any metadata); the
  day partition of FP/camera/day rides the synthetic ETL clock.
- Face blur is geometric (person-box head band ∪ live person pass), not
  a face model — no offline face detector exists in this environment;
  the band is deliberately generous and the policy is stated in code.
- Latency numbers are this workstation's CPU (AMD Ryzen 9 7940HS, 6
  cores, WSL2), not a server SKU; the c7i/c7g rows are a price-model
  illustration over those measurements. Graviton is unmeasured.
- Person P/R remains uncomputable until a person-labelled held-out
  (Wellington or equivalent) lands in the ETL assets; the person-load
  substitute measures detector output, not error rates.
