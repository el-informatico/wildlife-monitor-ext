"""Kill-switch B baseline branches, the head-to-head comparison, and
scenario assembly (tz attachment, zone rows, event validation)."""

from __future__ import annotations

import pytest

from agent import policy
from agent.baseline import BASELINE_NAME, decide_baseline, run_baseline
from agent.loop import run_triage
from agent.scenario import (attach_tz, build_event, det, synthetic_event,
                            synthetic_zones, zones_from_manifest_rows)
from agent.tools import ContextStore

NIGHT = "2026-03-10T23:00:00-06:00"
DAY = "2026-03-10T12:00:00-06:00"


class TestBaselineBranches:
    def test_empty_archives(self):
        d = decide_baseline(synthetic_event(detections=[]))
        assert d.action == "archive"

    def test_confirmed_person_always_escalates_without_context(self):
        ev = synthetic_event(camera_id="missouri-L05",   # an OPEN zone —
                             detections=[det("person", 0.6448)])  # invisible here
        d = decide_baseline(ev)
        assert d.action == "escalate"
        assert d.to_action()["priority"] == "high"
        assert any("no zone or clock context" in line for line in d.rationale)

    def test_low_conf_person_goes_to_review(self):
        d = decide_baseline(synthetic_event(detections=[det("person", 0.41)]))
        assert d.action == "approval_queue"
        assert d.to_action()["priority"] == "normal"

    def test_vehicle_archives_regardless_of_zone(self):
        d = decide_baseline(synthetic_event(captured_at=NIGHT,
                                            detections=[det("vehicle", 0.47)]))
        assert d.action == "archive"

    def test_night_marginal_animal_reanalysis_matches_agent_params(self):
        ev = synthetic_event(captured_at=NIGHT, night_content_ir=True,
                             detections=[det("animal", 0.34, "zebra")])
        act = decide_baseline(ev).to_action()
        assert act["type"] == "night_reanalysis"
        assert act["preprocess"] == policy.REANALYSIS_PREPROCESS
        assert act["conf_floor"] == policy.REANALYSIS_CONF_FLOOR

    def test_confident_animal_archives(self):
        d = decide_baseline(synthetic_event(detections=[det("animal", 0.93)]))
        assert d.action == "archive"

    def test_run_baseline_is_a_single_step_trace(self):
        tr = run_baseline(synthetic_event())
        assert tr["backend"] == BASELINE_NAME
        assert tr["schema"] == "wfm-triage-trace/v1"
        assert len(tr["steps"]) == 1
        assert tr["steps"][0]["kind"] == "decision"
        assert tr["steps"][0]["action"] == tr["action"]
        assert tr["trace_id"].startswith("sha256:")


class TestKillSwitchHeadToHead:
    """Same event, both lanes — where does the context loop change the
    action? This is the corpus comparison in miniature."""

    def _both(self, ev):
        store = ContextStore(events=[ev],
                             zones=synthetic_zones(ev["camera_id"]))
        agent = run_triage(ev, store)["action"]["type"]
        base = run_baseline(ev)["action"]["type"]
        return agent, base

    def test_open_zone_confirmed_person_diverges(self):
        ev = synthetic_event(camera_id="missouri-L05",
                             detections=[det("person", 0.6448)])
        assert self._both(ev) == ("approval_queue", "escalate")

    def test_restricted_curfew_confirmed_person_agrees(self):
        ev = synthetic_event(captured_at=NIGHT, detections=[det("person", 0.7)])
        assert self._both(ev) == ("escalate", "escalate")

    def test_confident_day_animal_agrees(self):
        ev = synthetic_event(detections=[det("animal", 0.93)])
        assert self._both(ev) == ("archive", "archive")

    def test_vehicle_restricted_curfew_diverges(self):
        ev = synthetic_event(captured_at=NIGHT,
                             detections=[det("vehicle", 0.47)])
        assert self._both(ev) == ("approval_queue", "archive")


class TestScenarioAssembly:
    def test_attach_tz_appends_dataset_offset(self):
        assert attach_tz("2026-03-10T04:00:00", "missouri") == \
            "2026-03-10T04:00:00-06:00"
        assert attach_tz("2026-03-10T04:00:00", "african_wildlife") == \
            "2026-03-10T04:00:00+02:00"

    def test_attach_tz_validates(self):
        with pytest.raises(ValueError, match="timezone-naive"):
            attach_tz("2026-03-10T04:00:00+02:00", "missouri")
        with pytest.raises(ValueError, match="no tz offset"):
            attach_tz("2026-03-10T04:00:00", "serengeti")

    def test_zones_from_manifest_rows(self):
        zones = zones_from_manifest_rows([
            {"camera_id": "missouri-L00", "zone_restricted": True},
            {"camera_id": "missouri-L05", "zone_restricted": False},
            {"camera_id": "missouri-L00", "zone_restricted": True},
        ])
        assert zones["missouri-L00"]["restricted"] is True
        assert zones["missouri-L00"]["restricted_window"] == \
            list(policy.RESTRICTED_WINDOW)
        assert zones["missouri-L05"]["restricted"] is False
        assert zones["missouri-L05"]["restricted_window"] is None
        assert "simulated" in zones["missouri-L00"]["policy_note"]

    def test_zone_inconsistency_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            zones_from_manifest_rows([
                {"camera_id": "c1", "zone_restricted": True},
                {"camera_id": "c1", "zone_restricted": False}])

    def test_build_event_validates_clock(self):
        with pytest.raises(ValueError, match="tz-aware"):
            synthetic_event(captured_at="2026-03-10T04:00:00")
        with pytest.raises(ValueError, match="clock_source"):
            synthetic_event(clock_source="gps")

    def test_build_event_normalizes_model_string(self):
        assert synthetic_event()["model"] == \
            {"runtime": "cv2.dnn (synthetic-test)"}

    def test_build_event_pass2_shape(self):
        ev = synthetic_event(pass_=2, reanalysis_of="ev-test",
                             model={"runtime": "cv2.dnn (pass 2)",
                                    "preprocess": policy.REANALYSIS_PREPROCESS})
        assert ev["pass"] == 2
        assert ev["reanalysis_of"] == "ev-test"
        assert ev["model"]["preprocess"] == policy.REANALYSIS_PREPROCESS

    def test_build_event_round_trip(self):
        ev = synthetic_event(detections=[det("animal", 0.34, "zebra")],
                             night_content_ir=True, night_synthetic_flag=True,
                             pass_=2, reanalysis_of="ev-test")
        rebuilt = build_event(
            event_id=ev["event_id"], camera_id=ev["camera_id"],
            dataset=ev["dataset"], source_key=ev["source_key"],
            captured_at=ev["captured_at"], clock_source=ev["clock_source"],
            detections=ev["detections"],
            night_content_ir=ev["night"]["content_ir"],
            night_synthetic_flag=ev["night"]["synthetic_flag"],
            model=ev["model"], pass_=ev["pass"],
            reanalysis_of=ev["reanalysis_of"])
        assert rebuilt == ev
