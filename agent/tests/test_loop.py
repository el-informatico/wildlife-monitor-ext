"""Loop transport: trace shape, determinism, budget, error handling."""

from __future__ import annotations

import pytest

from agent.adapters import ToolCall
from agent.loop import run_triage
from agent.scenario import det, synthetic_event, synthetic_zones
from agent.tools import ContextStore
from agent.trace import SCHEMA, canonical_json


def make_store(*events):
    cams = [e["camera_id"] for e in events] or ["missouri-L00"]
    return ContextStore(events=list(events), zones=synthetic_zones(*set(cams)))


class NeverDecides:
    """A pathological adapter: asks the same tool forever."""

    name = "never-decides"

    def next_step(self, state):
        return ToolCall("get_detection",
                        {"event_id": state.event["event_id"]})


class TestTraceShape:
    def test_happy_path_is_nine_steps(self):
        ev = synthetic_event()
        trace = run_triage(ev, make_store(ev))
        assert trace["schema"] == SCHEMA
        assert trace["backend"] == "rules-v1"
        assert trace["event_id"] == ev["event_id"]
        assert trace["camera_id"] == ev["camera_id"]
        assert trace["captured_at"] == ev["captured_at"]
        kinds = [s["kind"] for s in trace["steps"]]
        assert kinds == ["tool_call", "tool_result"] * 4 + ["decision"]
        assert [s["step"] for s in trace["steps"]] == list(range(1, 10))
        assert trace["steps"][-1]["action"] == trace["action"]
        assert trace["trace_id"].startswith("sha256:")
        assert "fallback" not in trace["steps"][-1]
        for call, result in zip(trace["steps"][0::2], trace["steps"][1::2]):
            assert call["tool"] == result["tool"]

    def test_decision_step_carries_rationale_and_refs(self):
        ev = synthetic_event(captured_at="2026-03-10T23:00:00-06:00",
                             detections=[det("person", 0.7)])
        trace = run_triage(ev, make_store(ev))
        last = trace["steps"][-1]
        assert last["rationale"] and last["policy_refs"]
        assert trace["action"]["type"] == "escalate"


class TestDeterminism:
    def test_two_runs_produce_identical_bytes(self):
        ev = synthetic_event(captured_at="2026-03-10T23:00:00-06:00",
                             night_content_ir=True,
                             detections=[det("animal", 0.3, "zebra")])
        t1 = run_triage(ev, make_store(ev))
        t2 = run_triage(ev, make_store(ev))
        assert canonical_json(t1) == canonical_json(t2)
        assert t1["trace_id"] == t2["trace_id"]
        assert t1["action"]["type"] == "night_reanalysis"


class TestBudget:
    def test_exhaustion_closes_with_the_safe_default(self):
        ev = synthetic_event()
        trace = run_triage(ev, make_store(ev), adapter=NeverDecides(),
                           max_steps=5)
        last = trace["steps"][-1]
        assert last["kind"] == "decision"
        assert last["fallback"] == "budget"
        assert trace["action"]["type"] == "approval_queue"
        assert trace["action"]["priority"] == "high"
        assert trace["action"]["approval_required"] is True
        assert trace["backend"] == "never-decides"

    def test_zero_budget_still_produces_a_decision(self):
        ev = synthetic_event()
        trace = run_triage(ev, make_store(ev), max_steps=0)
        assert trace["action"]["type"] == "approval_queue"
        assert trace["steps"][-1]["fallback"] == "budget"


class TestErrorHandling:
    def test_unregistered_event_recorded_then_safe_default(self):
        ev = synthetic_event()          # not registered in the store
        trace = run_triage(
            ev, ContextStore(events=[], zones=synthetic_zones("missouri-L00")))
        kinds = [s["kind"] for s in trace["steps"]]
        assert "tool_error" in kinds
        assert trace["action"]["type"] == "approval_queue"
        err = next(s for s in trace["steps"] if s["kind"] == "tool_error")
        assert "unknown event_id" in err["error"]

    def test_bad_adapter_return_raises(self):
        class Bad:
            name = "bad"

            def next_step(self, state):
                return "I have no idea"

        with pytest.raises(TypeError, match="ToolCall or Decision"):
            run_triage(synthetic_event(), make_store(), adapter=Bad())
