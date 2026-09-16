"""Kill-switch B baseline — the rules-only lane (no tools, no context).

The brief's kill-switch B (§5) demands an honest comparison: if the
agentic loop adds no decision value over a rules-only path, the rules
path stays and the Agentic Vision lane dies. This module IS that
rules-only path, built to make the comparison fair rather than
strawmanned:

- the perception-side policy is IDENTICAL to the reference backend
  (:class:`agent.adapters.RulesAdapter`) — the same thresholds from
  :mod:`agent.policy`, the same action vocabulary, the same night rule;
- the baseline simply CANNOT see what the tools provide — no zone, no
  clock/curfew, no history — and must fall back to event-only defaults
  wherever the agent uses context.

Event-only defaults where context is unavailable:

- confirmed person (≥ PERSON_CONFIRM_CONF) → escalate ALWAYS — the
  agent escalates only in restricted+curfew and otherwise routes to
  the approval queue at normal priority;
- low-confidence person → approval_queue normal (the fragmentation
  guard is perception-side and shared; the repeat-priority bump needs
  history and cannot fire);
- vehicle → archive — the restricted+curfew ranger check is invisible;
- animals → the identical night-reanalysis / archive split.

The corpus comparison (``agent/run_corpus.py``, artifacts under
``agent/demo/artifacts/``) therefore isolates exactly one variable:
what did the context loop change? Verdict in agent/G4-REPORT.md.
"""

from __future__ import annotations

from typing import Any

from . import actions, policy
from .adapters import Decision
from .trace import SCHEMA, finalize

BASELINE_NAME = "rules-only-v1"


def _max_conf_by_label(detections: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in detections:
        out[d["label"]] = max(out.get(d["label"], 0.0), d["confidence"])
    return out


def decide_baseline(event: dict[str, Any]) -> Decision:
    """The event-only decision — mirrors the agent's perception-side
    policy, minus everything the tools would have added."""
    dets = event["detections"]
    p_flags = actions.person_flags(event)
    max_conf = _max_conf_by_label(dets)
    night = event["night"]["content_ir"]

    if not dets:
        return Decision(
            "archive",
            {"reason": "no deployed detections — nothing to triage",
             **p_flags},
            (f"pass {event['pass']}: 0 detections deployed at conf ≥ 0.25",),
            ("empty-batch policy",),
        )

    person_conf = max_conf.get("person")
    if person_conf is not None:
        if person_conf >= policy.PERSON_CONFIRM_CONF:
            return Decision(
                "escalate",
                {"priority": "high",
                 "reason": "confirmed person — the rules-only lane has no "
                           "zone/hour context, so every confirmed person "
                           "event escalates with an evidence package",
                 "evidence_package": True, **p_flags},
                (f"person max conf {person_conf:.4f} ≥ "
                 f"{policy.PERSON_CONFIRM_CONF}",
                 "no zone or clock context available (rules-only baseline): "
                 "the restricted+curfew precondition for escalation cannot "
                 "be evaluated, so the lane treats every confirmed person "
                 "as the worst case"),
                ("PERSON_CONFIRM_CONF", "approval-policy"),
            )
        return Decision(
            "approval_queue",
            {"priority": "normal",
             "reason": "low-confidence person — human review as a probable "
                       "fragmentation artifact (no history context for a "
                       "repeat check)", **p_flags},
            (f"person max conf {person_conf:.4f} < "
             f"{policy.PERSON_CONFIRM_CONF} (fragmentation band — G3 "
             "failure case 1)",
             "no history context available (rules-only baseline): the "
             "repeat-signal priority bump cannot fire"),
            ("PERSON_CONFIRM_CONF", "G3-failure-case-1"),
        )

    vehicle_conf = max_conf.get("vehicle")
    if vehicle_conf is not None:
        return Decision(
            "archive",
            {"reason": "vehicle — no zone/hour context in the rules-only "
                       "lane; the restricted-curfew ranger check is "
                       "invisible here, so routine log", **p_flags},
            (f"vehicle max conf {vehicle_conf:.4f}",
             "no zone or clock context available (rules-only baseline)"),
            ("vehicle-policy",),
        )

    animal_conf = max_conf.get("animal", 0.0)
    if (night and event["pass"] == policy.REANALYSIS_ONLY_PASS
            and animal_conf < policy.NIGHT_REANALYSIS_MAX_CONF):
        return Decision(
            "night_reanalysis",
            actions.night_reanalysis_params(
                "marginal night detection — re-run perception with night "
                "preprocessing at the lower decode floor before acting"),
            (f"content-IR night frame ({policy.NIGHT_SOURCE})",
             f"animal max conf {animal_conf:.4f} < "
             f"{policy.NIGHT_REANALYSIS_MAX_CONF} — marginal band (G3 "
             "failure case 2: small edge-hugging night targets)"),
            ("NIGHT_REANALYSIS_MAX_CONF", "G3-failure-case-2"),
        )
    if event["pass"] > policy.REANALYSIS_ONLY_PASS:
        return Decision(
            "archive",
            {"reason": "re-analysis pass concluded — evidence recorded; no "
                       "further automatic passes", **p_flags},
            (f"pass {event['pass']} event: re-analysis already ran",),
            ("REANALYSIS_ONLY_PASS",),
        )
    return Decision(
        "archive",
        {"reason": "confident wildlife detection — routine log into the "
                   "species timeline", **p_flags},
        (f"animal max conf {animal_conf:.4f}" +
         (f" ≥ {policy.NIGHT_REANALYSIS_MAX_CONF}" if night else
          " (day frame)"),
         "species annotates only (closed-set:v0)"),
        ("species-annotate-only",),
    )


def run_baseline(event: dict[str, Any]) -> dict[str, Any]:
    """Single-step trace: the decision and nothing else — no tools, no
    context, no loop. Same trace schema as the agent so the comparison
    harness reads both identically."""
    decision = decide_baseline(event)
    action = decision.to_action()
    return finalize({
        "schema": SCHEMA,
        "backend": BASELINE_NAME,
        "event_id": event["event_id"],
        "camera_id": event["camera_id"],
        "captured_at": event["captured_at"],
        "action": action,
        "steps": [{
            "step": 1,
            "kind": "decision",
            "action": action,
            "rationale": list(decision.rationale),
            "policy_refs": list(decision.policy_refs),
        }],
    })
