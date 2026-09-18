# AWS Compute Grant Proposal — Wildlife Agentic Monitor

**Competition:** OpenCV AI Competition 2026, powered by AWS · **Track:** Agentic Vision · **Grant:** $150 compute credits · 2026-09-17

**Team name — pick one for the JotForm field:**
**WildGuard Vision** (recommended: serious, says both domain and method) · **TrapTriage** (names the core mechanism) · **SilentWatch** (evokes field monitoring)

## 1. Team name

Solo team. Candidates above; **WildGuard Vision** is the recommendation — short, serious, and it names both the domain (wildlife) and the function (vision triage).

## 2. Problem statement + real-world impact

Camera traps produce far more frames than any field team can review: ecological monitoring programs routinely accumulate backlogs of weeks, and false positives bury the handful of frames that matter. The consequence is twofold — missed evidence (rare species, human presence in restricted zones) and wasted expert hours.

This project triages every frame autonomously. An OpenCV 5 perception pipeline gates, detects, and classifies; a tool-calling triage agent then decides per event what to re-analyze, escalate, or archive. Humans only approve an evidence queue. People are never an accusation: every person action carries `approval_required` + `face_blur_required`, and the single image-serving path blurs faces always, at serve time.

Real-world impact: conservation teams and park rangers review a queue instead of a firehose. Measured on our corpus, the agent removes 100% of blind high-urgency escalations that a rules-only baseline fires on probable detector artifacts (5/5), while zero person evidence is dropped — correct urgency, same evidence to humans.

## 3. Planned/built OpenCV 5 image analysis

OpenCV 5.0.0 is built from source (git tag 5.0.0, no wheel exists) into every environment, and it is the substantive analysis layer in three roles:

- **Motion gating** — MOG2 (history 50, varThreshold 24) with CLAHE(3.0, (8,8)) night preprocessing; 42.5% of frames never reach the detector (measured inference saving).
- **Detection** — MegaDetector v6 (MDv6-yolov10-c) ONNX via `cv2.dnn`, the **new graph engine**; animal/person/vehicle boxes per frame at deploy conf 0.25.
- **Classification** — 4-species classifier v0 (ONNX, also on `cv2.dnn`), top-1 **0.9867** held-out.

Depth evidence, not wrappers: an engine ablation inside OpenCV itself (new graph engine vs classic) measured on 800 images; the deployed lane runs **$3.67/1M** images vs the PyTorch + Pytorch-Wildlife legacy lane's $4.16/1M (+13%).

**Agentic qualification (the lane's own bar):** image results must change subsequent actions — demonstrated in committed traces. Scenario s2: a marginal pass-1 detection (animal 0.3417 on a content-IR night frame) selects `night_reanalysis` as the next action; a **live second `cv2.dnn` pass** runs the CLAHE recipe at decode floor 0.10 and surfaces four marginal boxes; the pass-2 event re-enters the loop and the final action (archive) follows the pass-2 output. Trace: `agent/demo/artifacts/`.

## 4. Planned AWS architecture and services

Terraform-pinned and validated end-to-end on LocalStack; real deploy is what these credits fund:

- **S3** (versioned) — raw media, detection-batch sidecars (contract schema v1), motion jobs.
- **Lambda** — `wfm-detection-ingest` (validation → normalized items, idempotent `TransactWriteItems`) and `wfm-motion-gate` (Python 3.12 container image, ECR-hosted, vendoring the same source-built OpenCV 5; MOG2 runs inside the function — boot verified on the real Runtime Interface Client path).
- **SQS DLQ** (14 d) — poison messages, maximum_retry_attempts 0.
- **DynamoDB** — single table, on-demand.
- Events: S3 `ObjectCreated` notifications per prefix.

## 5. Architecture diagram

```
cameras → S3 (raw + sidecars + motion jobs)
            │ ObjectCreated
            ├─ detections/*.json ─→ Lambda ingest ── validate ──┬─→ DynamoDB
            │                                    └─ poison ──→ SQS DLQ
            └─ motion-jobs/ ─────→ Lambda motion (OpenCV 5 · MOG2) ─→ DynamoDB
DynamoDB events ─→ triage agent: 4 context tools (zone/hour/history/detection)
                      → decision table → ONE action per event:
                         escalate | approval_queue | night_reanalysis | archive
                         (night_reanalysis re-asks cv2.dnn live, max 2 passes)
approval queue ─→ read-only dashboard (map · species · queue · failure gallery · face blur always)
```

Full canonical diagrams (system, AWS lane, agentic flow): `docs/ARCHITECTURE.md`.

## 6. Target users / beneficiaries

Ecologists and conservation teams running camera-trap arrays (our corpus mirrors theirs: 26,177 frames, 48 cameras across three public datasets); park rangers who need human-presence evidence with correct urgency and privacy protections; reserve managers accountable for auditable evidence trails.

## 7. Evaluation method + judge demonstration

Everything below is measured, offline, CPU-only — each number traces to a committed report (`eval/`, `agent/`, `infra/`):

| measure | result |
|---|---|
| Detection (800-image sample, IoU ≥ 0.5, conf 0.25) | TP/FP/FN **812/676/210**, recall 0.7945; night recall **higher** than day (0.849 vs 0.775) |
| FP load | **Missouri 3.10 vs African Wildlife 1.54** FP/camera/day (mean 1.95) |
| Person P/R on person-labelled held-outs | in-domain (Seattle, 1,740 stills) **P 0.959 / R 0.879**; out-of-domain ceiling (COCO val2017, box) P 0.548 / R 0.383 |
| Agent vs rules-only baseline (200 events) | 195/200 agree; all 5 divergences = baseline escalates blind, agent queues for review |
| Test suites | **189** offline (pipeline 23 · eval 49 · agent 79 · dashboard 38) + 90 infra |
| G1 ingest | **1,504/1,504** images end-to-end S3 → Lambda → DynamoDB, 0 missing / 0 DLQ |

Honest limits, stated: vehicle detection measured unreliable (0/60 in-domain) — not claimed; a 7-case failure catalog with annotated images is committed (`eval/failures/`); the day/night clock partition is synthetic (seed-pinned) over measured counts.

**Judge demo:** the 7-scenario agent demo (`agent.demo.run_demo`) with byte-deterministic traces where every expectation is asserted at run time; plus the read-only ops dashboard (camera map, species counts, approval queue, failure gallery, `X-Face-Blur` verified live). The dashboard currently runs locally (offline constraint); a scheduled live demo in the submission window is the plan, and a real-AWS endpoint becomes possible with this grant.

## 8. Path

**Agentic Vision** (the $1K special) — deliberately our lane: the perception → decision → action loop with committed traces where `cv2.dnn` output changes the next action is exactly the qualification bar. Not COOL; not both.

## 9. Team bio

**Juan Zavala Quijano** — software engineer, Peru; solo builder. Currently running four concurrent hackathon tracks, including Ring/Amazon and AssemblyAI finalists-track work; builds end-to-end with measured-evidence discipline (every claim above traces to a committed report; gaps are stated as NOT MEASURED rather than papered over). Public work: YouTube **@jzavalaq**. Solo by choice — the repo, tests, and traces are all one person's verifiable output.

## Use of the $150 credits

The deploy is prepared, not deployed: `terraform plan` against real STS already computes the full 14-resource graph, and the Lambda container image boots OpenCV 5.0.0 on the real runtime path — what blocked is that this machine's only AWS identity has zero service permissions (`infra/DEPLOY-AWS.md` runbook). Credits fund: (1) the real-AWS replication of the 1,504-image G1 run; (2) compute + S3 for the full-corpus evaluation pass; (3) optionally the c7g Graviton latency bench that replaces our honestly-labeled NOT MEASURED cell.
