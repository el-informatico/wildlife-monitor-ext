"""Action-record invariants and policy-constant sanity (offline, stdlib)."""

from __future__ import annotations

import pytest

from agent import actions, policy


class TestBuildAction:
    def test_escalate_carries_approval_required(self):
        act = actions.build_action(
            "escalate", {"priority": "high", "reason": "person evidence"})
        assert act["type"] == "escalate"
        assert act["approval_required"] is True

    def test_approval_flag_cannot_be_overridden(self):
        with pytest.raises(ValueError, match="approval_required"):
            actions.build_action("escalate", {
                "priority": "high", "reason": "r", "approval_required": False})

    def test_approval_queue_requires_reason(self):
        with pytest.raises(ValueError, match="reason"):
            actions.build_action("approval_queue", {"priority": "normal"})

    def test_priority_must_be_normal_or_high(self):
        with pytest.raises(ValueError, match="priority"):
            actions.build_action("approval_queue",
                                 {"priority": "urgent", "reason": "r"})

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="unknown action type"):
            actions.build_action("detain", {"reason": "the product approves, "
                                                      "never accuses"})

    def test_archive_requires_reason_only(self):
        act = actions.build_action("archive", {"reason": "routine log"})
        assert act["type"] == "archive"
        assert "approval_required" not in act

    def test_night_reanalysis_needs_its_reparameterization(self):
        with pytest.raises(ValueError, match="preprocess"):
            actions.build_action("night_reanalysis",
                                 {"reason": "r", "conf_floor": 0.10})
        with pytest.raises(ValueError, match="conf_floor"):
            actions.build_action("night_reanalysis",
                                 {"reason": "r", "preprocess": "clahe"})

    def test_night_reanalysis_params_come_from_policy(self):
        act = actions.build_action("night_reanalysis",
                                   actions.night_reanalysis_params("why"))
        assert act["preprocess"] == policy.REANALYSIS_PREPROCESS
        assert act["conf_floor"] == policy.REANALYSIS_CONF_FLOOR
        assert act["deploy_conf"] == 0.25
        assert "approval_required" not in act  # no human needed to re-run
                                               # perception on pixels


class TestPersonFlags:
    def test_flag_when_person_present(self):
        ev = {"detections": [{"label": "animal"}, {"label": "person"}]}
        assert actions.person_flags(ev) == {"face_blur_required": True}

    def test_absent_otherwise(self):
        ev = {"detections": [{"label": "vehicle"}]}
        assert actions.person_flags(ev) == {}
        assert actions.person_flags({}) == {}


class TestPolicyConstants:
    def test_thresholds_are_the_documented_ones(self):
        assert policy.PERSON_CONFIRM_CONF == 0.50
        assert policy.NIGHT_REANALYSIS_MAX_CONF == 0.45
        assert policy.REANALYSIS_CONF_FLOOR == 0.10
        assert policy.REANALYSIS_ONLY_PASS == 1
        assert policy.RESTRICTED_WINDOW == ("18:00", "06:00")
        assert policy.HISTORY_LOOKBACK_HOURS == 72
        assert policy.PERSON_REPEAT_EVENTS == 2

    def test_budget_fits_the_uniform_consult(self):
        # 4 tool calls + 4 results + the decision = 9 recorded steps
        assert policy.MAX_STEPS >= 9

    def test_clock_tags_exist_for_both_paths(self):
        assert policy.CLOCK_PINNED == "pinned-demo"
        assert policy.CLOCK_MANIFEST_SYNTHETIC == "manifest-synthetic"
