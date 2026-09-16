"""The reference backend: consult phase, decision table, safe defaults."""

from __future__ import annotations

import pytest

from agent import policy
from agent.adapters import (BedrockAdapter, Decision, DecisionAdapter,
                            LoopState, RulesAdapter, ToolCall)
from agent.scenario import det, synthetic_event, synthetic_zones
from agent.tools import TOOL_ORDER, ContextStore, ToolError


def decide(event, store):
    """Drive RulesAdapter to a Decision against ``store`` — the loop's
    transport (including error recording) without the loop module."""
    adapter = RulesAdapter()
    state = LoopState(event=event)
    while True:
        nxt = adapter.next_step(state)
        if isinstance(nxt, Decision):
            return nxt
        state.steps.append({"kind": "tool_call", "tool": nxt.tool,
                            "args": nxt.args})
        try:
            result = store.call(nxt.tool, nxt.args)
        except ToolError as exc:
            state.steps.append({"kind": "tool_error", "tool": nxt.tool,
                                "error": str(exc)})
            continue
        state.steps.append({"kind": "tool_result", "tool": nxt.tool,
                            "result": result})


def store_for(event, *extra):
    cams = {event["camera_id"], *(e["camera_id"] for e in extra)}
    return ContextStore(events=[event, *extra],
                        zones=synthetic_zones(*cams))


NIGHT = "2026-03-10T23:00:00-06:00"      # restricted-zone curfew hours
DAY = "2026-03-10T12:00:00-06:00"        # curfew inactive


class TestConsultPhase:
    def test_uniform_tool_order_then_decision(self):
        ev = synthetic_event()
        store = store_for(ev)
        state = LoopState(event=ev)
        adapter = RulesAdapter()
        seen = []
        while True:
            nxt = adapter.next_step(state)
            if isinstance(nxt, Decision):
                break
            assert isinstance(nxt, ToolCall)
            seen.append(nxt.tool)
            state.steps.append({"kind": "tool_call", "tool": nxt.tool,
                                "args": nxt.args})
            state.steps.append({"kind": "tool_result", "tool": nxt.tool,
                                "result": store.call(nxt.tool, nxt.args)})
        assert seen == list(TOOL_ORDER)

    def test_args_derivation(self):
        ev = synthetic_event()
        state = LoopState(event=ev)
        assert RulesAdapter._args_for("query_hour", state) == {
            "camera_id": ev["camera_id"], "at": ev["captured_at"],
            "clock_source": ev["clock_source"]}
        assert (RulesAdapter._args_for("query_history", state)
                ["lookback_hours"] == policy.HISTORY_LOOKBACK_HOURS)

    def test_satisfies_the_adapter_protocol(self):
        assert isinstance(RulesAdapter(), DecisionAdapter)


class TestDecisionTable:
    def test_empty_batch_archives(self):
        ev = synthetic_event(detections=[])
        assert decide(ev, store_for(ev)).action == "archive"

    def test_confirmed_person_restricted_curfew_escalates(self):
        ev = synthetic_event(captured_at=NIGHT, detections=[det("person", 0.7)])
        d = decide(ev, store_for(ev))
        assert d.action == "escalate"
        act = d.to_action()
        assert act["priority"] == "high"
        assert act["approval_required"] is True
        assert act["face_blur_required"] is True
        assert act["evidence_package"] is True
        assert any("curfew" in line for line in d.rationale)
        assert "PERSON_CONFIRM_CONF" in d.policy_refs

    def test_confirmed_person_open_zone_goes_to_approval_queue(self):
        ev = synthetic_event(camera_id="missouri-L05",
                             detections=[det("person", 0.7)])
        d = decide(ev, store_for(ev))
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "normal"

    def test_confirmed_person_restricted_but_day(self):
        ev = synthetic_event(captured_at=DAY, detections=[det("person", 0.7)])
        d = decide(ev, store_for(ev))
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "normal"

    def test_low_conf_person_single_is_normal(self):
        ev = synthetic_event(camera_id="missouri-L05",
                             detections=[det("person", 0.41)])
        d = decide(ev, store_for(ev))
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "normal"
        assert any("fragmentation" in line for line in d.rationale)

    def test_low_conf_person_repeat_raises_priority(self):
        now = synthetic_event(camera_id="missouri-L05",
                              captured_at="2026-03-10T12:00:00-06:00",
                              detections=[det("person", 0.41)])
        earlier = [
            synthetic_event(event_id="p1", camera_id="missouri-L05",
                            captured_at="2026-03-09T08:00:00-06:00",
                            detections=[det("person", 0.3)]),
            synthetic_event(event_id="p2", camera_id="missouri-L05",
                            captured_at="2026-03-09T09:00:00-06:00",
                            detections=[det("person", 0.35)]),
        ]
        d = decide(now, store_for(now, *earlier))
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "high"
        assert any("repeat" in line for line in d.rationale)

    def test_vehicle_restricted_curfew_is_a_ranger_check(self):
        ev = synthetic_event(captured_at=NIGHT, detections=[det("vehicle", 0.47)])
        d = decide(ev, store_for(ev))
        assert d.action == "approval_queue"
        assert any("curfew active" in line for line in d.rationale)
        assert "ranger" in d.to_action()["reason"]

    def test_vehicle_open_zone_archives(self):
        ev = synthetic_event(camera_id="missouri-L05",
                             detections=[det("vehicle", 0.47)])
        assert decide(ev, store_for(ev)).action == "archive"

    def test_marginal_night_animal_triggers_reanalysis(self):
        ev = synthetic_event(captured_at=NIGHT, night_content_ir=True,
                             detections=[det("animal", 0.34, "zebra")])
        d = decide(ev, store_for(ev))
        assert d.action == "night_reanalysis"
        act = d.to_action()
        assert act["preprocess"] == policy.REANALYSIS_PREPROCESS
        assert act["conf_floor"] == policy.REANALYSIS_CONF_FLOOR
        assert any("content-IR" in line for line in d.rationale)

    def test_confident_night_animal_archives(self):
        ev = synthetic_event(captured_at=NIGHT, night_content_ir=True,
                             detections=[det("animal", 0.93, "deer")])
        assert decide(ev, store_for(ev)).action == "archive"

    def test_day_marginal_animal_archives(self):
        ev = synthetic_event(captured_at=DAY, night_content_ir=False,
                             detections=[det("animal", 0.34, "zebra")])
        assert decide(ev, store_for(ev)).action == "archive"

    def test_pass2_event_never_reanalyzes(self):
        ev = synthetic_event(captured_at=NIGHT, night_content_ir=True,
                             detections=[det("animal", 0.2)], pass_=2,
                             reanalysis_of="ev-test")
        d = decide(ev, store_for(ev))
        assert d.action == "archive"
        assert any("re-analysis already ran" in line for line in d.rationale)


class TestSafeDefaults:
    def test_tool_error_routes_to_approval_queue(self):
        ev = synthetic_event()          # deliberately NOT in the store
        store = ContextStore(events=[], zones=synthetic_zones("missouri-L00"))
        d = decide(ev, store)
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "normal"
        assert any("get_detection errored" in line for line in d.rationale)

    def test_errored_tool_is_not_reconsulted(self):
        state = LoopState(event=synthetic_event())
        state.steps.append({"kind": "tool_error", "tool": "get_detection",
                            "error": "unknown event_id 'ev-test'"})
        nxt = RulesAdapter().next_step(state)
        assert isinstance(nxt, ToolCall) and nxt.tool == "query_zone"

    def test_person_event_with_broken_context_still_flags_blur(self):
        ev = synthetic_event(detections=[det("person", 0.7)])
        store = ContextStore(events=[], zones=synthetic_zones("missouri-L00"))
        d = decide(ev, store)
        assert d.to_action()["face_blur_required"] is True


def test_bedrock_stub_refuses_to_run_offline():
    with pytest.raises(NotImplementedError, match="offline"):
        BedrockAdapter()
