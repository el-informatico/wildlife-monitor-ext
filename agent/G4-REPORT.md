# G4 report — triage agent, Gate G4, kill-switch B

Sprint 4 / Gate G4. Gate text: *"Demo grabable donde la salida de cv2.dnn
CAMBIA una decisión posterior documentada en trace (mín. 2 acciones
condicionales distintas: re-análisis nocturno y escalada/aprobación)."*

**Verdict: GATE MET.** The demo is
`agent/demo/artifacts/TRANSCRIPT.md` + `traces/` (committed,
regenerable with `python -m agent.demo.run_demo`, byte-identical): cv2.dnn
output demonstrably changes downstream decisions — the marginal pass-1
animal of s2 changes the agent's own next question and the final action
follows the pass-2 pixels (live second MDv6 pass); person/vehicle
confidences route escalation vs approval vs a ranger check, each with the
full 9-step trace. **Kill-switch B: NOT triggered** — over the whole
200-event corpus the context agent beats the rules-only baseline where
they differ (5/5 divergences are blind baseline escalations of open-zone
daytime person artifacts), and never loses evidence the baseline would
have kept. The honest boundary is stated in §4: the measured value comes
from the tool loop + decision table, not from LLM reasoning — the
default backend is deterministic by design (offline-first), and a future
Bedrock backend must beat `rules-v1` on this same corpus to justify its
spend.

## 1. What was built

A Bedrock-style agent loop, offline-first:

- **Tools** (`tools.py`): `get_detection`, `query_zone`, `query_hour`,
  `query_history` against an in-memory `ContextStore` (the corpus
  itself is the history; same camera, strictly earlier, 72 h lookback).
  Tool specs carry the Bedrock `toolConfig` shape.
- **Adapter interface** (`adapters.py`): `RulesAdapter` (`rules-v1`,
  default, deterministic — the reference implementation) and
  `BedrockAdapter` (documented Converse-API mapping; raises
  `NotImplementedError` offline, no credentials required to test that
  it does).
- **Loop** (`loop.py`): consult → decide, every step recorded;
  `MAX_STEPS` budget closes with the documented safe default
  (approval_queue) rather than dropping or auto-acting on an event; a
  tool error routes to the same safe default and is never re-consulted.
- **Actions** (`actions.py`): `escalate`, `approval_queue`,
  `night_reanalysis`, `archive` — person evidence always carries
  `approval_required` + `face_blur_required`; nothing can override the
  approval flag; unknown action types are rejected.
- **Traces** (`trace.py`): `wfm-triage-trace/v1`, canonical JSON,
  `trace_id = sha256(trace minus id)[:16]` — byte-deterministic, no
  wall clock. The uniform consult is exactly 9 steps; every scenario
  trace is committed.

Decision table (provenance per constant in `policy.py`):

| # | condition (in order) | action |
|---|---|---|
| 0 | any tool errored | approval_queue (normal) — safe default |
| 1 | no detections | archive |
| 2 | person ≥ 0.50 | restricted zone AND curfew active → **escalate (high)**; else approval_queue (normal) |
| 3 | person < 0.50 present | approval_queue — high iff ≥ 2 person events in the 72 h history (repeat signal), else normal |
| 4 | vehicle (no person) | restricted + curfew → approval_queue (ranger check); else archive |
| 5 | animals, pass 1, content-IR night, max conf < 0.45 | **night_reanalysis** (pass 2 re-enters the loop) |
| 6 | animals otherwise / pass 2 | archive (species annotates only) |

## 2. The demo — perception output changing decisions

Seven scenarios, real corpus payloads (`agent/demo/showcase_scenarios.json`;
expectations asserted at run time):

| id | image | perception (cv2.dnn, pass 1) | context | agent | baseline |
|---|---|---|---|---|---|
| s1 | mo-04611 | animal 0.9373 | L00 restricted, 04:00, curfew | archive | archive |
| s2 | mo-04612 | animal 0.3417 | L00 restricted, 04:01, content-IR night | **night_reanalysis → pass 2 → archive** | night_reanalysis |
| s3 | aw-00095 | person 0.9038 (+4 dets) | pinned: L00 restricted, 23:00 | **escalate (high)** | escalate (high) |
| s4 | mo-10786 | person 0.4102 | L05 open, 11:03 | approval_queue (normal) | approval_queue (normal) |
| s5 | mo-06604 | person 0.5164 | L17 open, 11:02 | approval_queue (normal) | **escalate (high)** |
| s6 | aw-00587 | vehicle 0.476 | pinned: L00 restricted, 23:00 | approval_queue (ranger check) | **archive** |
| s7 | mo-08536 | empty | L00 restricted, 17:00 | archive | archive |

s2 is the gate's core sequence, executed live:

1. pass 1 (corpus, deploy floor 0.25): one animal at 0.3417 — inside the
   marginal band where G3 pinned the night failure mode (small
   edge-hugging targets, failure case 2).
2. The agent's action is not archive/escalate — it re-asks cv2.dnn with
   a different question: grayscale → CLAHE(clip 3.0, tile (8,8)) — the
   motion gate's exact night recipe — decoded at floor 0.10
   (`REANALYSIS_PREPROCESS`, `REANALYSIS_CONF_FLOOR`).
3. Live pass 2 finds **four** animal boxes (0.1979 / 0.1898 / 0.1612 /
   0.1400) — evidence the pass-1 deploy floor never surfaced, none of it
   confident.
4. The pass-2 event (`mo-04612#p2`, `reanalysis_of: mo-04612`) re-enters
   the loop; the guard refuses a third pass and the final action
   (archive, "animal max conf 0.1979 after re-parameterized second
   pass") follows the pass-2 output, not the pass-1 one.

The two required conditional action classes are therefore both driven by
cv2.dnn output, in trace: **night re-analysis** (s2) and
**escalation/approval** (s3–s6). Pinned scenarios (s3, s6) mount real
payloads on a documented simulated context to exhibit branches the
corpus happens not to contain; their `clock_source` is `pinned-demo`
and the original corpus context is recorded beside them.

## 3. Kill-switch B — rules-only baseline vs context agent

Baseline `rules-only-v1` (`baseline.py`): IDENTICAL perception
thresholds, NO context tools — a confirmed person always escalates
(cannot see zone/clock), a vehicle always archives, no history check.
Anything the two share is agreement by construction; only the context
loop's value can create a divergence.

Corpus run (200 events / 48 cameras; committed artifact
`agent/artifacts/corpus_comparison.{json,md}`; per-event traces
regenerated to `processed/s4_corpus_traces_*.jsonl`):

| action (priority) | agent | baseline |
|---|---|---|
| approval_queue:normal | 7 | 2 |
| archive | 184 | 184 |
| escalate:high | **0** | **5** |
| night_reanalysis | 9 | 9 |

**195/200 agree; all 5 divergences are the same shape**: confirmed
person (≥ 0.50) in an OPEN zone, midday — the baseline escalates blind
(5/5 highest-urgency pages), the agent queues the same evidence for
human review at normal priority with the zone/clock rationale in the
trace (aw-00095 0.9038, aw-00494 0.7985, aw-00551 0.7691, mo-06604
0.5164, mo-10782 0.6448 — per G3's catalog, AW person boxes are
close-up fragmentation artifacts; missouri's are edge slivers). The two
sub-0.50 person events agree (both queue at normal). The corpus's one
vehicle event sits in an open zone (both archive); the
restricted+curfew branch is exhibited by demo s6, where the baseline's
blind archive drops a check the agent performs.

What the comparison does NOT show: any case where the agent escalates
and the baseline does not — impossible by construction (the baseline
escalates maximally), and the corpus contains no restricted-zone curfew
person event. The agent's measured value is exactly: same evidence to
humans, correct urgency, plus one vigilance branch (s6) the baseline
cannot express.

## 4. Honest boundaries

- The 5 "wins" de-escalate probable detector artifacts, not real
  intruders — every one is still human-reviewed. The claim is urgency
  correctness, never fewer reviews.
- The winning lane is `rules-v1` — a deterministic backend. The agentic
  deliverable is the TOOL LOOP + traces (what the offline-first brief
  prescribed); no LLM inference ran in this sprint. Any future Bedrock
  backend must beat `rules-v1` on this same corpus (same traces, same
  comparison) before its cost is justified — that experiment, not this
  one, decides the LLM question.
- Zone/hour/history are simulated context (ETL flags + fixed window +
  synthetic manifest clock, tagged `manifest-synthetic` /
  `pinned-demo` in every trace). The decision VALUE measured here is
  conditional on that policy being the real one.
- History on the corpus is the corpus itself (200 events, sparse: ~4.4
  events/camera) — the repeat-signal branch never fired in the run
  (0 priority divergences); it is exercised by unit tests and remains
  policy, not measured behavior.

## 5. Cost note (brief §7 — "Bedrock para el agente: medir en S4")

Measured: **$0** — the default backend is deterministic and offline; no
AWS credential was touched. The Bedrock lane's cost model (Converse
API, one tool-use round-trip per consult step, 9 steps/event ≈ 200
events × ~4 tool round-trips) is documented in `adapters.py` for the
moment credentials resume; the corpus comparison above is the
evaluation it must beat.

## 6. Reproduction

    ../wildlife-monitor/.venv/bin/python -m pytest agent/tests -q          # 79
    ../wildlife-monitor/.venv/bin/python -m pytest eval/tests pipeline/tests agent/tests -q   # 132
    ../wildlife-monitor/.venv/bin/python -m agent.run_corpus
    ../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo

Determinism: both runners were executed twice — every committed
artifact (comparison JSON/MD, all traces, transcript) reproduced
byte-identically; `trace_id`s are content hashes (no wall clock).
