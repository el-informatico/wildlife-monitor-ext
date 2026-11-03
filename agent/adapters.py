"""Decision adapters — the swappable "brain" behind the tool loop.

The loop in :mod:`agent.loop` owns NOTHING about how decisions are made.
It repeatedly hands a :class:`LoopState` (the event + everything gathered
so far) to an adapter and gets back either a :class:`ToolCall` (keep
gathering) or a :class:`Decision` (act now). Two adapters ship:

- :class:`RulesAdapter` — the deterministic reference backend. A fixed
  consult order (all four tools), then a decision table whose thresholds
  and their provenance live in :mod:`agent.policy`. This is the default
  everywhere: tests, corpus runs, the demo. No cloud calls, byte-
  deterministic traces.
- :class:`BedrockAdapter` — the documented swap point for a real LLM
  (Amazon Bedrock Converse API with the same tool specs). It maps the
  loop onto Converse semantics and STOPS there: offline, constructing a
  client raises. It exists so the exact production contract is reviewable
  without any network dependency in this repo.

Why a rules reference backend is honest for Gate G4: the gate asks
whether the loop's DECISIONS are changed by perception output and
context, with traces proving it. That is a property of the loop + tools
+ policy, measurable with a deterministic backend — and the measurement
is repeatable. What the rules backend cannot demonstrate (LLM judgment
on out-of-policy situations) is exactly what the Bedrock adapter
documents as the upgrade path; the G4 report states this plainly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from . import actions, policy
from .tools import TOOL_ORDER, ToolError


# ---------------------------------------------------------------------------
# The adapter contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    action: str                       # one of actions.ACTION_TYPES
    params: dict[str, Any]
    rationale: tuple[str, ...]        # every line cites a fact that fired
    policy_refs: tuple[str, ...] = ()

    def to_action(self) -> dict[str, Any]:
        act = actions.build_action(self.action, {**self.params,
                                                 "rationale": list(self.rationale)})
        if self.policy_refs:
            act["policy_refs"] = list(self.policy_refs)
        return act


@runtime_checkable
class DecisionAdapter(Protocol):
    name: str

    def next_step(self, state: "LoopState") -> ToolCall | Decision:
        """One turn: consult another tool, or decide."""
        ...


@dataclass
class LoopState:
    """What the adapter sees each turn: the event + gathered results."""

    event: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)

    def result(self, tool: str) -> dict[str, Any] | None:
        """Most recent successful result for ``tool`` (None if absent)."""
        out = None
        for step in self.steps:
            if step.get("kind") == "tool_result" and step.get("tool") == tool:
                out = step["result"]
        return out

    def error(self, tool: str) -> str | None:
        """Most recent error string for ``tool`` (None if it never errored)."""
        out = None
        for step in self.steps:
            if step.get("kind") == "tool_error" and step.get("tool") == tool:
                out = step["error"]
        return out


# ---------------------------------------------------------------------------
# Reference backend: deterministic rules
# ---------------------------------------------------------------------------


class RulesAdapter:
    """Deterministic reference backend (default; no cloud calls).

    Turn structure: consult every tool in :data:`TOOL_ORDER` once —
    ground on perception, then zone, hour, history — then decide from
    the gathered facts. The uniform consult order is deliberate: every
    trace shows the full context set the loop had access to, which is
    what the G4 evidence needs (a dynamic backend may skip tools; the
    trace then shows what it chose to skip).

    Decision table (evaluation order; every branch cites its facts in
    the rationale and its constants' provenance in policy_refs):

    0. any tool errored         → approval_queue (safe default — the
                                  loop does not act autonomously on a
                                  broken context layer; a tool is asked
                                  at most once, errors are not retried)
    1. no detections            → archive (nothing to triage)
    2. person, conf ≥ .50       → restricted zone AND curfew active
                                  → escalate (high, approval required, blur)
                                  else → approval_queue (normal, blur)
       person, conf < .50       → repeat signal in history (≥ 2 person
                                  events in lookback) → approval_queue HIGH
                                  else → approval_queue normal (fragmentation
                                  guard: G3 case 1 — low-conf person boxes
                                  on this stack are mostly fragments)
    3. vehicle (no person)      → restricted AND curfew → approval_queue
                                  (ranger-patrol check); else archive
    4. animals only, pass 1     → night AND max conf < .45
                                  → night_reanalysis (CLAHE + floor .10;
                                  the pass-2 output re-enters the loop)
       animals, otherwise       → archive (routine wildlife log; species
                                  annotates only — closed-set caveat)
    """

    name = "rules-v1"

    def next_step(self, state: LoopState) -> ToolCall | Decision:
        # A tool that already errored is not asked again — the store is
        # deterministic, so a retry would repeat the error and burn the
        # step budget; the decision proceeds with the error branch (0).
        missing = [t for t in TOOL_ORDER
                   if state.result(t) is None and state.error(t) is None]
        if missing:
            return ToolCall(missing[0], self._args_for(missing[0], state))
        return self._decide(state)

    # -- consult args, derived from the event --------------------------------

    @staticmethod
    def _args_for(tool: str, state: LoopState) -> dict[str, Any]:
        ev = state.event
        if tool == "get_detection":
            return {"event_id": ev["event_id"]}
        if tool == "query_zone":
            return {"camera_id": ev["camera_id"]}
        if tool == "query_hour":
            return {"camera_id": ev["camera_id"], "at": ev["captured_at"],
                    "clock_source": ev["clock_source"]}
        if tool == "query_history":
            return {"camera_id": ev["camera_id"],
                    "lookback_hours": policy.HISTORY_LOOKBACK_HOURS,
                    "exclude_event_id": ev["event_id"]}
        raise ToolError(f"no consult args defined for {tool!r}")

    # -- the decision table ---------------------------------------------------

    def _decide(self, state: LoopState) -> Decision:
        errored = [t for t in TOOL_ORDER if state.error(t) is not None]
        if errored:
            return self._tool_error_default(state, errored)
        ev = state.event
        det = state.result("get_detection") or {}
        zone = state.result("query_zone") or {}
        hour = state.result("query_hour") or {}
        hist = state.result("query_history") or {}
        return self._decide_from(ev, det, zone, hour, hist)

    @staticmethod
    def _tool_error_default(state: LoopState, errored: list[str]) -> Decision:
        return Decision(
            "approval_queue",
            {"priority": "normal",
             "reason": "context layer error — the loop will not act "
                       "autonomously with a broken tool backend; human "
                       "review is the safe default",
             **actions.person_flags(state.event)},
            tuple(f"{t} errored: {state.error(t)}" for t in errored)
            + ("safe default policy: an event whose context cannot be "
               "established goes to the approval queue, never to an "
               "automatic action",),
            ("tool-error-safe-default", "approval-policy"),
        )

    def _decide_from(self, ev, det, zone, hour, hist) -> Decision:
        p_flags = actions.person_flags(ev)
        max_conf = det.get("max_conf_by_label", {})
        night = ev["night"]["content_ir"]

        if not ev["detections"]:
            return Decision(
                "archive",
                {"reason": "no deployed detections — nothing to triage",
                 **p_flags},
                ("pass %d: 0 detections deployed at conf ≥ 0.25" % ev["pass"],),
                ("empty-batch policy",),
            )

        # --- 1. person -----------------------------------------------------
        person_conf = max_conf.get("person")
        if person_conf is not None:
            restricted = zone.get("restricted", False)
            curfew = hour.get("curfew_active")
            if person_conf >= policy.PERSON_CONFIRM_CONF:
                if restricted and curfew:
                    return Decision(
                        "escalate",
                        {"priority": "high", "reason": "person detected in "
                         "restricted zone during curfew — routed to the "
                         "human approval queue with an evidence package",
                         "evidence_package": True, **p_flags},
                        (f"person max conf {person_conf:.4f} ≥ "
                         f"{policy.PERSON_CONFIRM_CONF}",
                         f"zone {zone.get('zone_id')} restricted, curfew "
                         f"window {zone.get('restricted_window')} active at "
                         f"local hour {hour.get('hour')}",
                         f"history: {hist.get('person_events', 0)} person "
                         f"events in {hist.get('lookback_hours')}h lookback",
                         "approval is REQUIRED before any action on people — "
                         "the system approves, never accuses"),
                        ("PERSON_CONFIRM_CONF", "RESTRICTED_WINDOW",
                         "approval-policy"),
                    )
                return Decision(
                    "approval_queue",
                    {"priority": "normal", "reason": "person detected in an "
                     "open zone / open hours — human review, no escalation",
                     **p_flags},
                    (f"person max conf {person_conf:.4f} ≥ "
                     f"{policy.PERSON_CONFIRM_CONF}",
                     f"zone {zone.get('zone_id')} NOT restricted"
                     if not restricted else
                     f"zone {zone.get('zone_id')} restricted but curfew "
                     f"inactive at local hour {hour.get('hour')}",
                     "person evidence always reaches a human; escalation "
                     "requires restricted zone AND active curfew"),
                    ("PERSON_CONFIRM_CONF", "approval-policy"),
                )
            # low-confidence person: probable fragmentation artifact
            repeat = hist.get("person_events", 0) >= policy.PERSON_REPEAT_EVENTS
            if repeat:
                return Decision(
                    "approval_queue",
                    {"priority": "high", "reason": "repeat low-confidence "
                     "person signal on this camera inside the lookback — "
                     "human review at high priority (still approval, not "
                     "accusation)", **p_flags},
                    (f"person max conf {person_conf:.4f} < "
                     f"{policy.PERSON_CONFIRM_CONF} (fragmentation band)",
                     f"history: {hist.get('person_events')} person events ≥ "
                     f"{policy.PERSON_REPEAT_EVENTS} in "
                     f"{hist.get('lookback_hours')}h — repeat signal raises "
                     "priority"),
                    ("PERSON_CONFIRM_CONF", "PERSON_REPEAT_EVENTS",
                     "G3-failure-case-1"),
                )
            return Decision(
                "approval_queue",
                {"priority": "normal", "reason": "single low-confidence "
                 "person detection — routed to human review as a probable "
                 "fragmentation artifact; never auto-escalated", **p_flags},
                (f"person max conf {person_conf:.4f} < "
                 f"{policy.PERSON_CONFIRM_CONF} (fragmentation band — G3 "
                 "failure case 1: low-conf person boxes on close-up "
                 "subjects are mostly fragments)",
                 f"history: {hist.get('person_events', 0)} person events in "
                 f"{hist.get('lookback_hours')}h — no repeat signal"),
                ("PERSON_CONFIRM_CONF", "G3-failure-case-1", "approval-policy"),
            )

        # --- 2. vehicle (person absent) --------------------------------------
        vehicle_conf = max_conf.get("vehicle")
        if vehicle_conf is not None:
            if zone.get("restricted") and hour.get("curfew_active"):
                return Decision(
                    "approval_queue",
                    {"priority": "normal", "reason": "vehicle in restricted "
                     "zone during curfew — ranger-patrol check by a human",
                     **p_flags},
                    (f"vehicle max conf {vehicle_conf:.4f}",
                     f"zone {zone.get('zone_id')} restricted, curfew active "
                     f"at local hour {hour.get('hour')}"),
                    ("vehicle-policy",),
                )
            return Decision(
                "archive",
                {"reason": "vehicle in open zone / open hours — routine "
                           "log", **p_flags},
                (f"vehicle max conf {vehicle_conf:.4f}",
                 "zone open or curfew inactive — no policy trigger"),
                ("vehicle-policy",),
            )

        # --- 3. animals only --------------------------------------------------
        animal_conf = max_conf.get("animal", 0.0)
        top = max(ev["detections"], key=lambda d: d["confidence"])
        species_note = (f"top detection {top['label']}"
                        + (f" sp.{top['species']}" if top.get("species") else ""))
        if (night and ev["pass"] == policy.REANALYSIS_ONLY_PASS
                and animal_conf < policy.NIGHT_REANALYSIS_MAX_CONF):
            return Decision(
                "night_reanalysis",
                actions.night_reanalysis_params(
                    "marginal night detection — re-run perception with "
                    "night preprocessing at the lower decode floor before "
                    "acting"),
                (f"content-IR night frame ({policy.NIGHT_SOURCE})",
                 f"animal max conf {animal_conf:.4f} < "
                 f"{policy.NIGHT_REANALYSIS_MAX_CONF} — inside the marginal "
                 "band where G3 pinned the night failure mode (small "
                 "edge-hugging targets, failure case 2)",
                 "re-analysis changes the QUESTION (CLAHE preprocessing + "
                 f"floor {policy.REANALYSIS_CONF_FLOOR}), then the pass-2 "
                 "output re-enters this loop"),
                ("NIGHT_REANALYSIS_MAX_CONF", "REANALYSIS_PREPROCESS",
                 "G3-failure-case-2", "G3-night-recall-0.8491"),
            )
        if ev["pass"] > policy.REANALYSIS_ONLY_PASS:
            return Decision(
                "archive",
                {"reason": "re-analysis pass concluded — evidence recorded; "
                           "no further automatic passes", **p_flags},
                (f"pass {ev['pass']} event: re-analysis already ran",
                 f"animal max conf {animal_conf:.4f} after re-parameterized "
                 "second pass"),
                ("REANALYSIS_ONLY_PASS",),
            )
        return Decision(
            "archive",
            {"reason": "confident wildlife detection — routine log into the "
                       "species timeline", **p_flags},
            (f"animal max conf {animal_conf:.4f}" +
             (f" ≥ {policy.NIGHT_REANALYSIS_MAX_CONF}" if night else
              " (day frame — night re-analysis does not apply)"),
             f"{species_note}; species annotates only (closed-set:v0)",
             f"history: {hist.get('events', 0)} events, species seen "
             f"{hist.get('species_seen', [])}"),
            ("species-annotate-only", "G3-failure-case-5"),
        )


# ---------------------------------------------------------------------------
# Cloud backend: documented Bedrock swap point (NO network calls here)
# ---------------------------------------------------------------------------


class BedrockAdapter:
    """Amazon Bedrock Converse-backed decider — DOCUMENTED STUB.

    Maps the loop onto the Converse tool-use protocol so the production
    contract is reviewable offline:

    - ``system``: the triage role — consult the four tools, then choose
      exactly one action from ``actions.ACTION_TYPES``; every person-
      related action must be an approval request; return a rationale
      citing the facts that fired (same rules the reference backend's
      traces already follow).
    - ``toolConfig``: :data:`agent.tools.TOOL_SPECS` serialized 1:1
      (``{"toolSpec": {"name", "description", "inputSchema"}}``).
    - messages: the event record as the first user turn, then the
      alternating tool calls/results the loop already produces — the
      :class:`LoopState` transcript IS the Converse message history.
    - stopReason ``tool_use`` → :class:`ToolCall`; ``end_turn`` → parse
      the action/rationale JSON from the text block → :class:`Decision`
      (a malformed reply falls back to the loop's safe default, so a
      bad model turn can never silently skip human review).

    Offline this class REFUSES to decide: constructing a client needs
    real credentials (``WFM_AGENT_BACKEND=bedrock`` + the boto3 chain).
    Nothing in this repo instantiates it with a live client — the
    deterministic :class:`RulesAdapter` is the default backend for every
    test, run, and committed artifact. When a Bedrock model is swapped
    in, the traces gain ``backend: "bedrock:<model-id>"`` and everything
    downstream (trace schema, actions, kill-switch comparison harness)
    is unchanged.

    Cost note (brief §7: "Bedrock para el agente — medir en S4"): with
    the rules backend the S4 spend is zero; a hosted model at
    ~4 tool rounds × (event + results) per decision is the unit to
    measure before enabling it in production.
    """

    name = "bedrock-stub"

    def __init__(self, model_id: str = "bedrock-model-id") -> None:
        self.model_id = model_id
        raise NotImplementedError(
            "BedrockAdapter is the documented production swap point; this "
            "repo is offline-first. Enable by instantiating a boto3 "
            "bedrock-runtime client with real credentials and implementing "
            "next_step() against Converse — see the class docstring for the "
            "exact mapping. The deterministic RulesAdapter is the default."
        )
