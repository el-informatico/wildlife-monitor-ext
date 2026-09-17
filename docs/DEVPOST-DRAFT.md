# Devpost submission draft — description + entry fields (UNSUBMITTED)

Draft-only. Nothing here is submitted; the entry is created and pressed
by the owner per `docs/SUBMISSION-CHECKLIST.md` §3 (submit by
**10-24**, hard deadline 10-26 23:59 PT). Every number below cites
committed evidence in the two repos — nothing claims an unmeasured
result. Sections map onto the Devpost editor's standard fields
(Inspiration / What it does / How we built it / Challenges /
Accomplishments / What we learned / What's next / Built with).

## Entry fields (owner-gated)

- **Title:** `[JUAN: pick — working title "Wildlife Agentic Monitor"
  (matches TECH-REPORT.md); alternative "Camera-Trap Triage Agent"]`
- **One-line pitch:** Camera-trap footage triaged by an agent whose
  vision output decides its next move — marginal night detections
  trigger a live second OpenCV 5 pass, people always route to a human
  approval queue, and every decision leaves a byte-deterministic trace.
- **Track:** Agentic Vision (special award path; also entered for the
  overall awards — the rules state COOL/agentic methods earn overall
  credit only to the extent they improve the project).
- **Video URL:** `[JUAN: record by 10-20 per SUBMISSION-CHECKLIST §3.2,
  upload unlisted]`
- **Repo URLs:** `[JUAN: after publication YES + one-repo-vs-two
  decision, SUBMISSION-CHECKLIST §3.1]`
- **Team:** `[JUAN: name + 2–3 line bio]`

## Inspiration

Camera-trap conservation programs drown in footage: thousands of
motion-triggered frames per station per week, reviewed by hand, with
night imagery — the majority of captures — effectively discarded
because classical pipelines miss animals in low-light IR frames. The
real problem is not "detect animals"; it is "what should happen next
for each frame, and who has to look at it." That is a triage decision,
and it is exactly where a perception → decision → action loop belongs.

## What it does

Images land in S3; an AWS Lambda runtime built on OpenCV 5 runs the
vision core — MOG2 motion gating, MobileDet-v6 (MDv6) detection, a
ResNet18 species classifier, and privacy blur on any person before
storage or display; records land in DynamoDB with an SQS dead-letter
queue. A triage agent then assigns **exactly one action per event**,
chosen from `escalate` / `approval_queue` / `night_reanalysis` /
`archive`, after consulting four context tools (current detection,
zone rules, hour/curfew, 72-hour history). People are never
auto-accused: the only path from a person detection to action is the
human approval queue. A read-only dashboard (map, species, queue,
gallery — blur enforced server-side) serves the monitoring UX.

## How we built it

- **OpenCV 5 is the substantive vision core**, not a wrapper: MOG2
  background subtraction, MDv6 object detection via `cv2.dnn`, ResNet18
  classification, CLAHE-based night reanalysis, and privacy blur — all
  OpenCV APIs in the runtime Lambda path.
- **AWS lane:** S3 → Lambda → DynamoDB → SQS DLQ, provisioned with
  pinned Terraform and validated end-to-end on LocalStack: **1,504 of
  1,504** real African Wildlife images processed through the full path,
  **0 failures, 0 DLQ entries**.
- **Deterministic-first agent:** the decision layer is a deterministic
  backend (`rules-v1`) with a documented Bedrock adapter stub — policy:
  a model backend must beat `rules-v1` on the committed 200-event
  corpus before its cost is justified. Bedrock spend: **$0**. The
  agentic claim rests on the closed loop and its traces, not on model
  spend.
- **Byte-deterministic traces:** every decision emits a
  `wfm-triage-trace/v1` record whose id is derived from the trace
  content itself — re-runs are byte-identical, so every claim below is
  reproducible from the repo.

### The agent moment (the strongest 30 seconds)

The Agentic Vision track sets one qualification bar: "To qualify,
image or video results must influence a subsequent plan, tool call,
action, or request for human approval. A chatbot that only explains a
fixed vision result is not enough—the visual evidence must change what
the system does next." Scenario s2 is that bar, executed: pass 1
returns one animal at **0.3417** on a content-IR night frame
(restricted zone, 04:01). That marginal number does not get archived
or narrated — it selects the next tool call, `night_reanalysis`, which
runs a **live second MDv6 pass** on re-parameterized pixels
(grayscale → CLAHE(3.0, (8,8)), decode floor 0.10). The second pass
surfaces **four marginal boxes (0.1979 / 0.1898 / 0.1612 / 0.1400)**
the pass-1 deploy floor never showed; the pass-2 event re-enters the
loop, the guard refuses a third pass, and the final action —
**archive** — follows the pass-2 output. The visual evidence changed
what the system did next.

## Challenges we ran into

- **Night IR was the failure mode, so it became a tool.** The
  day/night evaluation pinned small edge-hugging night targets as the
  top failure case; the answer was not a bigger model but a second
  question — the CLAHE reanalysis pass. Night recall ends **higher**
  than day (0.8491 vs 0.7754).
- **Parity discipline on the detector port.** The `cv2.dnn` MDv6 lane
  reproduces the PyTorch reference matcher **identically — 812 / 676 /
  210** TP/FP/FN at deploy confidence 0.25, a 0.0-point recall gap on
  1,488 boxes; the species head matches PyTorch **375/375 top-1**.
- **Saying "not computable" out loud.** Where person P/R cannot be
  honestly computed (Wellington corpus has no person ground truth), the
  report says NOT_COMPUTABLE instead of guessing a number.

## Accomplishments we're proud of

- **195/200 events agree** between agent and rules-only baseline on the
  committed corpus (200 events, 48 cameras) — and all 5 divergences are
  the baseline escalating blind on confirmed people in open zones at
  midday (agent: 0 blind escalations vs 5). Zero person evidence
  dropped in either lane.
- **189 offline tests** (pipeline 23 · eval 49 · agent 79 · dashboard
  38, incl. the endpoint HTTP suite) — the repo re-runs green from a
  clean clone.
- **$3.67/1M frames** for the new engine vs $4.16/1M legacy (+13%) at
  e2e p50 **148.47 ms** — with the honest label: Graviton NOT MEASURED
  (no real-AWS session yet; prices are stated assumptions).
- **$0 LLM spend** with the agentic loop still closed and traced.

## What we learned

The loop is the agentic part, not the model. A deterministic decision
table with the right tools closes the perception → decision → action
loop today, at zero marginal cost, and forces any future model backend
to beat a committed baseline instead of a vibe. And marginal-confidence
frames are signal, not noise: 0.3417 at 04:01 is exactly the number
that should select the next tool call.

## What's next

- Re-target the validated Terraform to real AWS (one `terraform apply`;
  the LocalStack run is the rehearsal) and re-run the 1,504-image
  corpus against it.
- Earn the Bedrock lane the policy demands: beat `rules-v1` on the same
  200-event corpus, or it does not ship.
- Extend the corpus beyond African Wildlife / Missouri — the evaluation
  harness is committed and reproducible.

## How this maps to the overall judging rubric

| Criterion (weight) | Where the evidence lives |
|---|---|
| Technical execution (30%) | OpenCV 5 vision core in the runtime path; detector parity 812/676/210 identical, 0.0-pt gap; species head 375/375; 189 offline tests; `docs/TECH-REPORT.md` |
| Innovation (20%) | Marginal-confidence-as-trigger: the 0.3417 night detection selects the next tool call; deterministic-first agent lane with a beat-the-baseline policy |
| Real-world impact (20%) | Night imagery recovered (night recall 0.8491 > day 0.7754); ranger time triaged to what matters; people handled by approval queue only |
| UX (10%) | Read-only dashboard, server-enforced blur, approval queue as the single human touchpoint |
| Documentation & presentation (10%) | `docs/TECH-REPORT.md`, `docs/ARCHITECTURE.md`, reproduction commands, the under-5-minute video |
| Cloud delivery, reproducibility, responsible operation (10%) | S3/Lambda/DynamoDB/SQS on pinned Terraform; 1,504/1,504 E2E with 0 DLQ; latency/cost table with honest LocalStack + NOT_MEASURED labels |

## How this maps to the Agentic Vision award rubric

| Criterion (weight) | Evidence |
|---|---|
| Substantive OpenCV 5 + agent integration (30%) | MDv6 + CLAHE reanalysis run inside the agent's tool call, live, on the pixels |
| Orchestration & appropriate autonomy (25%) | 4 context tools, exactly-one-action per event, MAX_STEPS guard, safe default (`approval_queue`) on tool error |
| Task effectiveness & evaluation (20%) | 200-event / 48-camera corpus, 195/200 agreement, 0 blind escalations vs 5, divergence table committed |
| Failure handling, observability, security, human control (15%) | SQS DLQ; byte-deterministic traces; face blur before storage; approval-only path to action on people |
| UX, documentation, demonstration (10%) | Dashboard, this description, TECH-REPORT, the s2 walkthrough in the video |

## Competition requirements checklist

- **OpenCV 5 for substantive image/video analysis** — yes: MOG2, MDv6
  via `cv2.dnn`, ResNet18, CLAHE, blur are the runtime path.
- **Meaningful component on AWS** — yes: detection/classification/blur
  run in Lambda over S3/DynamoDB/SQS. Honest label for judges: the E2E
  validation to date is on LocalStack; the real-AWS re-run is queued
  (see What's next).
- **Agentic additional evidence** — workflow diagram
  (`docs/ARCHITECTURE.md`), committed trace where OpenCV output changes
  a later decision (`agent/demo/artifacts/traces/s2.json` +
  `s2_pass2.json`), evaluation incl. failure handling and human control
  (`docs/TECH-REPORT.md` §9, `agent/G4-REPORT.md`).

## Built with

Python · OpenCV 5 · MobileDet-v6 (MDv6) · ResNet18 · AWS S3 / Lambda /
DynamoDB / SQS · Terraform · LocalStack
