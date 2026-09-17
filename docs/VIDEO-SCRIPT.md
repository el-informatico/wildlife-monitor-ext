# Video script + storyboard — demo recording (S6 packaging)

**Target: ≤ 5:00 hard; this script times to ~4:45** with clip beats.
English narration, **696 words** at a comfortable 150 wpm (≈ 4:38 of
speech + beats). Record the clips first (list below), then narrate over
them — every clip is a terminal, a browser, or a rendered file; no
camera needed. All commands run from this repo's root and use the
sprints-1-2 venv verbatim (`../wildlife-monitor/.venv/bin/python`).

Honesty is load-bearing here: the dashboard is a **local** endpoint, the
camera map is **schematic**, the zone/hour context is **simulated**, the
camera clock is **synthetic**, person P/R is **not computable**, Graviton
is **not measured**, and the AWS lane is validated on **LocalStack only**.
The narration below already says these words where they are true —
**do not smooth them out while editing** (protected list at the end).

## Timing budget

| # | Time | Segment | Words |
|---|---|---|---|
| 1 | 0:00–0:20 | Hook — every frame costs | 49 |
| 2 | 0:20–0:50 | What it is — two repos, one loop | 69 |
| 3 | 0:50–1:35 | Perception honesty — the measured numbers | 99 |
| 4 | 1:35–2:05 | Failure gallery — seven committed cases | 63 |
| 5 | 2:05–3:05 | The agent moment — s2, live second pass | 142 |
| 6 | 3:05–3:40 | Human control — person events never dropped | 89 |
| 7 | 3:40–4:15 | Dashboard + privacy — blur always | 81 |
| 8 | 4:15–4:40 | Speed, cost, engineering | 60 |
| 9 | 4:40–5:00 | Close — honest limits + sign-off | 44 |

Total: 696 words. If a take runs long, cut inside segments 3 and 6 —
never segment 5 (the star) and never a honesty word.

## Narration + storyboard

### 1. Hook (0:00–0:20)

> Camera traps fire thousands of frames, and almost all of them are
> nothing. Every frame a human reviews costs. This monitor triages every
> frame itself: an agent decides what to re-analyze, escalate, or
> archive — humans only approve an evidence queue. Every camera-trap
> photo triages itself; the human only approves.

**On screen:** terminal, idle prompt → type the test suite (clip C1),
let the tail land: `189 passed`. Cut on the number.

### 2. What it is (0:20–0:50)

> Two repos, one loop. The sprints-one-two repo: ETL turns raw camera
> media into a manifest, a MOG2 motion gate drops static frames, CLAHE
> preps infrared nights, and OpenCV 5's DNN module runs MegaDetector v6
> as ONNX on the new graph engine. A species classifier reads
> training-recipe crops; everything normalizes into contract records.
> This extension repo adds the triage agent and the dashboard. The whole
> chain runs offline, on CPU.

**On screen:** `docs/ARCHITECTURE.md` rendered (GitHub or editor
preview, clip C2) — slow scroll of the system diagram: ETL → motion
gate (with the CLAHE night edge) → `cv2.dnn` MDv6 → crops → classifier
→ contract records → agent + dashboard; end on the AWS/LocalStack lane
diagram.

### 3. Perception honesty (0:50–1:35)

> Before the agent, the perception numbers — measured, on held-out data,
> on screen now. Detection recall at the deploy threshold: 0.7945 over
> an 800-image held-out sample. The surprise: night recall is higher
> than day — 0.8491 versus 0.7754 — because the real cost is daytime
> close-up fragmentation, not infrared. The classifier scores 0.9867
> top-one on its 375-crop held-out split, reproduced bit-tight against
> the trainer's own reference. False positives per camera per day: mean
> 1.95, and 99 of 347 active camera-days at zero. The motion gate ran
> the full 26,177-frame manifest with zero decode failures. All local,
> CPU-only, and reproducible from committed runners.

**On screen:**
- 0:50 `eval/G3-REPORT.md` (clip C6) — the detection table at deploy
  conf 0.25; **moment M4**: zoom the `all|day 0.7754` vs
  `all|night 0.8491` pair.
- 1:10 zoom the classifier table row `all 375 0.9867` and the
  "375/375 top-1 agreement, max |Δprob| 0.0001" line.
- 1:25 `eval/S5-REPORT.md` §3 — the line `FP per active cell mean 1.95
  · p50 1 · p95 6 · max 18` and `99 cells at zero FP`.

### 4. Failure gallery (1:35–2:05)

> And the failures are committed, not hidden: seven cases, each with its
> image, its measured numbers, and a diagnosis. Close-up fragmentation.
> Night edge-hugging. A tiny crop upsampled into the wrong species.
> Frame-edge truncation. A closed-set classifier calling a roe deer a
> rhino at 0.57. Sequence selection loss. A motion-gate burst-start
> spike. Every case names its fix — and which fixes threshold tuning
> cannot deliver.

**On screen:** dashboard failure gallery in the browser
(`http://localhost:8787`, clip C7) — scroll case 1 → case 7, images
visible; fallback if the server is down: `eval/failures/CATALOG.md`
rendered, same scroll.

### 5. The agent moment (2:05–3:05)

> This is the moment the project is built around. Pass one: an infrared
> night frame, one animal, confidence 0.3417 — marginal. The agent does
> not act on that number. Its action is to re-ask OpenCV a different
> question: grayscale plus CLAHE preprocessing — the motion gate's own
> night recipe — and a decode floor lowered to 0.10. That second cv2.dnn
> pass runs live on the pixels, and it returns four marginal animal
> boxes the deploy floor never surfaced. The pass-two event re-enters
> the loop, and the final action — archive — is decided on pass-two
> pixels, not pass-one. Nothing here is asserted after the fact: the
> full nine-step tool-call trace is content-hashed and committed. The
> zone and hour context it consulted is simulated — tagged so in the
> trace — but the perception passes are real pixels, twice. One marginal
> number changed the question; the answer changed the decision.

**On screen:**
- 2:05 terminal: `../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo`
  (clip C3) — seven scenarios print, each asserting its expected
  decision at run time.
- 2:15 open `agent/demo/artifacts/TRANSCRIPT.md` at the s2 section
  (clip C3). **Moment M1**: pass-1 `night_reanalysis` verdict beside the
  pass-2 `archive` verdict.
- 2:30 **moment M2**: zoom the pass-2 table — four animal rows
  `0.1979 / 0.1898 / 0.1612 / 0.1400` — then the archive rationale
  ("animal max conf 0.1979 after re-parameterized second pass").
- 2:50 open `agent/demo/artifacts/traces/s2.json` and `s2_pass2.json`
  (clip C4) — zoom `trace_id: sha256:…`, the 9 steps of tool calls and
  rationales in each, and `reanalysis_of: mo-04612` in the pass-2 event.

### 6. Human control (3:05–3:40)

> People are never an accusation. Every person event reaches a human —
> approval queue or escalation — and every queued person image is
> face-blurred before it is served. Over the 200-event corpus, agent
> versus rules-only baseline: 195 agree. All five divergences are the
> same shape — the baseline escalating, at highest urgency, open-zone
> midday person artifacts; the agent queues the same evidence at normal
> priority, with the zone and clock rationale written into the trace.
> Zero person evidence dropped, in either lane. The context layer is
> simulated; the routing discipline is not.

**On screen:** `agent/artifacts/corpus_comparison.md` (clip C5) — the
action-counts table (`escalate:high 0 vs 5`), then the every-person-event
table; drag a highlight down the `agent` / `baseline` columns of the five
divergent rows in lockstep.

### 7. Dashboard + privacy (3:40–4:15)

> The ops dashboard is a local, read-only endpoint. A 48-camera
> schematic map — no GPS exists in any of the data, and the payload says
> so itself. Deployed species counts. The seven-event approval queue.
> The failure gallery. And one privacy rule with no exception: exactly
> one image-serving path, and face blur is always on. Stored person
> boxes united with a live person-detection pass at serve time, a
> destructive head-band blur, and a response header that states the
> outcome on every single image.

**On screen:**
- 3:40 browser at `http://localhost:8787` (clip C8; server started with
  `../wildlife-monitor/.venv/bin/python -m dashboard --host ::1` — WSL2
  note in C8). Slow pan: schematic map (the no-GPS note is in the
  payload) → species counts elephant 162 · zebra 85 · rhino 62 ·
  buffalo 30 → the 7-event approval queue → the failure gallery.
- 4:03 terminal: `curl -sI 'http://localhost:8787/image?src=frame:aw-00095'`
  — **moment M3**: zoom the response line
  `X-Face-Blur: applied=6;live`.

### 8. Speed, cost, engineering (4:15–4:40)

> Speed and cost. The new engine: 148 milliseconds end-to-end, median
> per image, on CPU — thirteen percent faster than the legacy PyTorch
> lane, identical inputs, identical weights. That is three dollars
> sixty-seven per million images, against four sixteen. Graviton:
> honestly not measured — the bench procedure is documented instead.
> And the engineering floor: nearly 300 offline tests, a stdlib-first
> core, byte-deterministic artifacts.

**On screen:**
- 4:15 `processed/s5_latency.md` (or `eval/S5-REPORT.md` §5, clip C9) —
  **moment M5**: zoom the `$/1M` pair `3.67` (cv2.dnn new) vs `4.16`
  (PyTorch legacy), then the `Graviton … NOT MEASURED` row.
- 4:30 terminal: `cd infra && uv run pytest -q` tail — `90 passed`
  (**moment M6**, second half; the `189 passed` tail opened the video).

### 9. Close (4:40–5:00)

> The honest limits, in one breath. Person precision and recall: not
> computable — no person-labelled held-out exists locally, so we state
> it instead of fabricating it. The camera clock is synthetic.
> Measurements are CPU-only. The AWS lane is validated on LocalStack
> only. Thanks for watching.

**On screen:** `eval/S5-REPORT.md` §4 — the `NOT_COMPUTABLE` verdict
block — for one beat, then title card: repo name + Apache-2.0 (repo URL
only if the publication gate has passed by recording day; otherwise name
only).

## Clips to record

Record each clip separately (retakes stay cheap). Terminal: large font
(18 pt+), consistent dark theme, 1920×1080, `clear` before every
command, cursor noise off.

| Clip | What | Command / file | Expected on screen |
|---|---|---|---|
| C1 | test suite | `../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests eval/tests agent/tests dashboard/tests -q` | tail `189 passed` (speak the number from the run you record) |
| C1b | infra suite | `cd infra && uv run pytest -q` | tail `90 passed` (same rule) |
| C2 | architecture | `docs/ARCHITECTURE.md` rendered (fallback: `README.md`) | system mermaid scroll (ETL → gate → cv2.dnn → classifier → records → agent/dashboard), then the AWS/LocalStack lane diagram |
| C3 | agent demo | `../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo`, then open `agent/demo/artifacts/TRANSCRIPT.md` | 7 scenarios, expectations asserted; TRANSCRIPT s2: pass-1 0.3417, pass-2 four-box table, archive rationale |
| C4 | traces | open `agent/demo/artifacts/traces/s2.json` + `s2_pass2.json` | 9 steps each, `trace_id: sha256:…`, actions `night_reanalysis` → `archive` |
| C5 | corpus comparison | open `agent/artifacts/corpus_comparison.md` | counts table (0 vs 5 escalate:high), every-person-event table, 5 divergent rows |
| C6 | perception numbers | open `eval/G3-REPORT.md` (+ `eval/S5-REPORT.md` §3) | detection day/night rows, classifier 0.9867 row, FP/cell mean 1.95 / 99-at-zero line |
| C7 | failure gallery | browser `http://localhost:8787` (or `eval/failures/CATALOG.md`) | 7 cases with images and diagnoses |
| C8 | dashboard + blur | server: `../wildlife-monitor/.venv/bin/python -m dashboard --host ::1`, then browser `http://localhost:8787`; curl `curl -sI 'http://localhost:8787/image?src=frame:aw-00095'` | map / species / queue / gallery; header `X-Face-Blur: applied=6;live`. WSL2: `--host ::1` because mirrored networking drops IPv4-loopback connects; if `localhost` stalls in curl, use `http://[::1]:8787/...`; a Windows-side browser at `127.0.0.1:8787` works |
| C9 | latency/cost | open `processed/s5_latency.md` (or `eval/S5-REPORT.md` §5) | 3-lane table, e2e p50 148.47 vs 168.41 ms, `$/1M` 3.67 vs 4.16, Graviton NOT MEASURED |

**Before recording, regenerate everything fresh** (deterministic — the
demo and corpus reproduce byte-identical artifacts):

```bash
git status                                                        # clean tree, nothing distracting
../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests eval/tests agent/tests dashboard/tests -q   # 189
cd infra && uv run pytest -q                                      # 90
../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo       # transcript + traces, byte-identical
../wildlife-monitor/.venv/bin/python -m agent.run_corpus          # only if processed/ corpus traces are
                                                                  # missing — the dashboard queue reads them
```

Do NOT re-run the eval runners for the video (`eval.run_eval` and
`eval.run_latency` are slow; the committed reports and
`processed/s5_latency.md` are the numbers of record — speak whatever the
artifact you show has printed). Disable OS notifications; close personal
apps/tabs (nothing personal should appear in the recording).

## Moments (the frames that must land)

| Moment | Where | What the viewer sees |
|---|---|---|
| M1 | C3/C4, 2:15–2:30 | s2 pass-1 verdict `night_reanalysis` vs pass-2 verdict `archive`, side by side — the decision change |
| M2 | C3, 2:30–2:40 | pass-2 four-box table zoomed: `0.1979 / 0.1898 / 0.1612 / 0.1400` |
| M3 | C8, 4:03–4:10 | the `X-Face-Blur: applied=6;live` response header line |
| M4 | C6, 0:50–1:05 | day vs night recall pair: `0.7754` vs `0.8491` |
| M5 | C9, 4:15–4:25 | `$/1M` pair: `3.67` (new) vs `4.16` (legacy) |
| M6 | C1/C1b, 0:15 + 4:30 | test-suite pass tails: `189 passed` and `90 passed` |

## Editing notes

- Zooms: 150–200 % on table rows and header lines; keep each zoom ≥ 2 s.
- Burned-in captions welcome (English) — keep the honesty words intact.
- No background music with vocals; low ambient only, narration front.
- Export: 1080p, H.264, **≤ 5:00 hard**; upload unlisted to YouTube
  first, check the end-to-end time, then paste the URL into Devpost.
- Filename: `wildlife-monitor-demo-<date>.mp4`.

## Words that must survive editing

The narration says these where they are true; editing must not remove
or soften them (captions included):

**simulated · schematic · local · offline · not computable**

and the supporting honesty terms: **synthetic · CPU-only · not measured
(Graviton) · LocalStack · NOT_COMPUTABLE (on screen) · read-only**.
