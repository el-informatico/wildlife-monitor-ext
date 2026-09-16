"""Triage policy constants — every threshold with its provenance.

The decision table these constants feed lives in
:mod:`agent.adapters` (the reference backend) and is mirrored — with
NO context lookups — in :mod:`agent.baseline` for the kill-switch B
comparison. The two share thresholds wherever they overlap so the
comparison isolates what the tool/context loop adds, not tuning
differences (documented in the G4 report).

Provenance discipline (same as the G3 report): each constant cites the
sprint/measurement that justifies it. The zone/hour policy layer is
SIMULATED context — the ETL assigns a per-camera ``zone_restricted``
flag (consistent per camera across all 26,177 manifest rows) and this
sprint fixes a restricted-hours window on top of it; no real reserve
operates behind these numbers. The perception side is real cv2.dnn
output from the E2E union stage.
"""

from __future__ import annotations

# --- perception-side thresholds (G3/E2E-informed) ---------------------------

#: A person detection must reach this confidence before the agent treats
#: it as more than a probable fragmentation artifact. The E2E union run
#: deployed 10 person detections; on african_wildlife (a dataset with no
#: people in frame) they are all detector artifacts, and G3's failure
#: catalog pins the mechanism (close-up fragmentation, case 1). Below
#: this bar a person event routes to the human approval queue at normal
#: priority — never to escalation — because "the product approves, never
#: accuses" (project framing decision).
PERSON_CONFIRM_CONF = 0.50

#: Night animal detections below this confidence trigger a second,
#: re-parameterized perception pass instead of an immediate action.
#: G3 measured night recall HIGHER than day (0.8491 vs 0.7754) but the
#: night failure mode is small edge-hugging targets (failure case 2) —
#: exactly the boxes whose confidence sits in the marginal band. The
#: re-analysis pass re-runs MDv6 with CLAHE night preprocessing at a
#: lower decode floor, i.e. it asks cv2.dnn a different question rather
#: than re-asking the same one.
NIGHT_REANALYSIS_MAX_CONF = 0.45

#: Decode floor for the re-analysis pass: the E2E stage's CONF_FLOOR
#: (detections decoded down to 0.10 so sweeps need no re-run — see
#: pipeline/e2e.py). The re-analysis deploys everything ≥ this floor:
#: its purpose is to surface marginal evidence, not to re-filter it.
REANALYSIS_CONF_FLOOR = 0.10

#: Night preprocessing applied before the re-analysis forward pass — the
#: same CLAHE recipe the motion gate uses on content-IR frames since S1:
#: applied to the GRAYSCALE plane (clip 3.0, tile (8,8)) and replicated
#: back to 3 channels for the detector. infra/handler/motion_app.py pins
#: the parity; the string names the plane so traces cannot be read as
#: per-channel CLAHE on BGR.
REANALYSIS_PREPROCESS = "clahe-gray(clip=3.0, tile=(8,8))"

#: The re-analysis action exists to close a marginal question once: it
#: fires only on pass-1 events. A pass-2 event never re-triggers it
#: (loop guard — the agent does not chase its own tail).
REANALYSIS_ONLY_PASS = 1

# --- context-side policy (SIMULATED — see module docstring) -----------------

#: Restricted-zone curfew window, local clock. Crosses midnight.
#: Simulated policy: "core wildlife zone — no human activity 18:00–06:00".
RESTRICTED_WINDOW: tuple[str, str] = ("18:00", "06:00")

#: History lookback for the repeat-signal check (simulated policy clock).
HISTORY_LOOKBACK_HOURS = 72

#: Repeat low-confidence person signals inside the lookback that raise
#: the approval-queue priority (still approval, still never accusation).
PERSON_REPEAT_EVENTS = 2

# --- loop safety --------------------------------------------------------------


#: Hard ceiling on recorded steps per event (tool calls + results + the
#: final decision). If an adapter cannot decide within it, the loop
#: closes with the documented safe default — approval_queue — so a stuck
#: agent can never silently drop an event or auto-act on it. Generous
#: for the reference backend, whose uniform consult + decision is
#: exactly 9 steps.
MAX_STEPS = 16

#: Fixed per-dataset UTC offsets attached to the manifest's timezone-
#: naive camera_ts so the hour tool has a local clock. The manifest
#: clock itself is a SYNTHETIC uniform assignment (ETL synthetic layer;
#: G3 documented the same for is_night — agreement with content ≈
#: chance), so corpus hour numbers are simulated context; showcase
#: scenarios pin explicit clock values instead. Offsets are the reserves'
#: real nominal zones: South Africa (AW) and Missouri (CST, no DST — a
#: documented simplification).
DATASET_TZ_OFFSET = {
    "african_wildlife": "+02:00",
    "missouri": "-06:00",
}

#: Night is decided by the CONTENT-based IR flag (mean HSV saturation
#: < 25 — G3's split definition, cached in processed/g3_ir_flags.json),
#: never by the manifest's synthetic is_night. Both travel in the event
#: so traces can show the disagreement when it occurs.
NIGHT_SOURCE = "content-ir (mean HSV saturation < 25, G3 definition)"

#: Clock provenance tags — every event carries one so no downstream
#: number can pretend a synthetic clock was measured.
CLOCK_PINNED = "pinned-demo"            # showcase: explicit, documented
CLOCK_MANIFEST_SYNTHETIC = "manifest-synthetic"  # ETL uniform assignment
