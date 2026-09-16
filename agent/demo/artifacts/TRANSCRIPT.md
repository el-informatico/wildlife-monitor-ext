# G4 demo transcript — perception output changing triage decisions

Seven scenarios over real cv2.dnn corpus payloads (S2/E2E union
run). Reproduce with:

    ../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo

Every decision below is asserted against the fixture's
`expect` at run time; full step-by-step traces (tool calls, tool
results, rationale) sit next to this transcript in `traces/`.

## s1 — mo-04611: confident night animal inside a restricted zone during curfew — archive

the curfew governs human activity, not wildlife: a 0.94 animal at 04:00 in zone Z-missouri-L00 archives without a human.

- perception (pass 1, corpus): animal 0.9373 (rhino)
- context: camera missouri-L00 (restricted zone), 2026-03-10T04:00:00-06:00 — clock `manifest-synthetic`
- rules-only baseline: **archive**

decision: **archive**
  - animal max conf 0.9373 ≥ 0.45
  - top detection animal sp.rhino; species annotates only (closed-set:v0)
  - history: 0 events, species seen []
  - policy refs: species-annotate-only, G3-failure-case-5

## s2 — mo-04612: marginal night animal — the live second perception pass

pass 1 returns animal 0.3417 on a content-IR night frame; the agent does not act on it, it re-asks cv2.dnn with the motion-gate CLAHE recipe and decides on the pass-2 output.

- perception (pass 1, corpus): animal 0.3417 (zebra)
- context: camera missouri-L00 (restricted zone), 2026-03-10T04:01:11-06:00 — clock `manifest-synthetic`
- rules-only baseline: **night_reanalysis**

decision: **night_reanalysis**
  - content-IR night frame (content-ir (mean HSV saturation < 25, G3 definition))
  - animal max conf 0.3417 < 0.45 — inside the marginal band where G3 pinned the night failure mode (small edge-hugging targets, failure case 2)
  - re-analysis changes the QUESTION (CLAHE preprocessing + floor 0.1), then the pass-2 output re-enters this loop
  - policy refs: NIGHT_REANALYSIS_MAX_CONF, REANALYSIS_PREPROCESS, G3-failure-case-2, G3-night-recall-0.8491

### pass 2 — live second cv2.dnn pass on `missouri/Set1/1.58-Roe_Deer/SEQ75631/SEQ75631_IMG_0010.JPG`

preprocess `clahe-gray(clip=3.0, tile=(8,8))`, decode floor 0.1 (pass-1 floor context: the deploy stage ran at 0.25):

| label | confidence | species |
|---|---|---|
| animal | 0.1979 | — |
| animal | 0.1898 | — |
| animal | 0.1612 | — |
| animal | 0.1400 | — |

The pass-2 output re-enters the loop as event `mo-04612#p2` (`reanalysis_of: mo-04612`, pass 2 — the re-analysis never re-triggers):

decision: **archive**
  - pass 2 event: re-analysis already ran
  - animal max conf 0.1979 after re-parameterized second pass
  - policy refs: REANALYSIS_ONLY_PASS

## s3 — aw-00095: confirmed person, restricted zone, curfew active — escalate

the strongest person artifact in the corpus (0.9038, an AW close-up fragmentation per G3 case 1) mounted on missouri-L00 at 23:00: same pixels, context changed, action becomes escalation with blur + evidence package.

- perception (pass 1, corpus): person 0.9038, animal 0.8097 (elephant), person 0.7225, person 0.7168, animal 0.3581 (elephant)
- context: camera missouri-L00 (restricted zone), 2026-03-10T23:00:00-06:00 — clock `pinned-demo`
- pinned recontextualization — original corpus context: african_wildlife-L23, 2026-03-09T15:00:00+02:00, open zone — where the agent queues it (see corpus comparison)
- rules-only baseline: **escalate (high)**

decision: **escalate (high)**
  - person max conf 0.9038 ≥ 0.5
  - zone Z-missouri-L00 restricted, curfew window ['18:00', '06:00'] active at local hour 23
  - history: 0 person events in 72h lookback
  - approval is REQUIRED before any action on people — the system approves, never accuses
  - policy refs: PERSON_CONFIRM_CONF, RESTRICTED_WINDOW, approval-policy

## s4 — mo-10786: person at 0.4102 — below the confirm bar, single occurrence

G3 catalog person cases live under 0.5; without repeat history the agent reviews at normal priority and never escalates on a fragmentation band.

- perception (pass 1, corpus): person 0.4102
- context: camera missouri-L05 (open zone), 2026-03-21T11:03:49-06:00 — clock `manifest-synthetic`
- rules-only baseline: **approval_queue (normal)**

decision: **approval_queue (normal)**
  - person max conf 0.4102 < 0.5 (fragmentation band — G3 failure case 1: low-conf person boxes on close-up subjects are mostly fragments)
  - history: 0 person events in 72h — no repeat signal
  - policy refs: PERSON_CONFIRM_CONF, G3-failure-case-1, approval-policy

## s5 — mo-06604: person at 0.5164 — 0.0164 over the bar, open zone, midday

crossing PERSON_CONFIRM_CONF makes the agent treat it as evidence, but the open zone and inactive clock keep it in the queue — this is the event the rules-only baseline escalates blind (corpus comparison divergence).

- perception (pass 1, corpus): person 0.5164
- context: camera missouri-L17 (open zone), 2026-03-14T11:02:15-06:00 — clock `manifest-synthetic`
- rules-only baseline: **escalate (high)**

decision: **approval_queue (normal)**
  - person max conf 0.5164 ≥ 0.5
  - zone Z-missouri-L17 NOT restricted
  - person evidence always reaches a human; escalation requires restricted zone AND active curfew
  - policy refs: PERSON_CONFIRM_CONF, approval-policy

## s6 — aw-00587: vehicle in a restricted zone during curfew — ranger check

the corpus's only vehicle detection (0.476) mounted on missouri-L00 at 23:00: not an escalation, a queued check that rangers are expected on the road at night.

- perception (pass 1, corpus): vehicle 0.4760
- context: camera missouri-L00 (restricted zone), 2026-03-10T23:00:00-06:00 — clock `pinned-demo`
- pinned recontextualization — original corpus context: african_wildlife-L22, 2026-03-09T08:00:00+02:00, open zone — where both lanes archive it
- rules-only baseline: **archive**

decision: **approval_queue (normal)**
  - vehicle max conf 0.4760
  - zone Z-missouri-L00 restricted, curfew active at local hour 23
  - policy refs: vehicle-policy

## s7 — mo-08536: empty batch — archive

cv2.dnn found nothing; note the content-IR night flag is true while the manifest's synthetic is_night is false — night travels with the content flag and nothing else happens.

- perception (pass 1, corpus): no detections (empty batch)
- context: camera missouri-L00 (restricted zone), 2026-03-16T17:00:24-06:00 — clock `manifest-synthetic`
- rules-only baseline: **archive**

decision: **archive**
  - pass 1: 0 detections deployed at conf ≥ 0.25
  - policy refs: empty-batch policy

---

Gate G4 asks for at least two distinct conditional actions driven
by cv2.dnn output, documented in trace:

- **night re-analysis** (s2): a marginal pass-1 animal changes the
  agent's own next question — pass 2 re-runs cv2.dnn with the
  motion-gate CLAHE recipe and the final action follows the pass-2
  output, not the pass-1 one;
- **escalation / approval** (s3–s6): person and vehicle confidences
  from cv2.dnn route to escalation (restricted zone + curfew),
  normal-priority human review (open zone, or below the confirm
  bar), and a ranger check (vehicle + curfew) — each with its full
  tool-call trace in `traces/`.

Framing held throughout: every person event reaches a human
(queue or escalation — never a drop, never an accusation); the
simulated zone/hour layer is tagged in every trace
(`pinned-demo` / `manifest-synthetic`).
