"""Tool layer: dispatch validation, curfew math, context-store behavior."""

from __future__ import annotations

import pytest

from agent import policy
from agent.scenario import det, synthetic_event, synthetic_zones
from agent.tools import (TOOL_ORDER, TOOL_SPECS, ContextStore, ToolError,
                         curfew_active, parse_event_ts)


@pytest.fixture
def store():
    return ContextStore(events=[synthetic_event()],
                        zones=synthetic_zones("missouri-L00"))


class TestCurfewActive:
    def test_none_window_means_no_curfew(self):
        assert curfew_active(23, None) is None

    def test_wrapping_window_membership(self):
        win = ("18:00", "06:00")
        assert curfew_active(18, win) is True      # start inclusive
        assert curfew_active(23, win) is True
        assert curfew_active(0, win) is True
        assert curfew_active(5, win) is True
        assert curfew_active(6, win) is False      # end exclusive
        assert curfew_active(12, win) is False

    def test_non_wrapping_window(self):
        win = ("08:00", "17:00")
        assert curfew_active(8, win) is True
        assert curfew_active(16, win) is True
        assert curfew_active(17, win) is False
        assert curfew_active(7, win) is False


class TestTimestamps:
    def test_naive_ts_rejected(self):
        with pytest.raises(ToolError, match="tz-aware"):
            parse_event_ts("2026-03-10T04:00:00")

    def test_z_suffix_and_offset_accepted(self):
        assert parse_event_ts("2026-03-10T04:00:00Z").tzinfo is not None
        assert parse_event_ts("2026-03-10T04:00:00-06:00").hour == 4


class TestRegistry:
    def test_the_four_brief_tools_in_consult_order(self):
        assert set(TOOL_SPECS) == {"get_detection", "query_zone",
                                   "query_hour", "query_history"}
        assert TOOL_ORDER == tuple(TOOL_ORDER)
        assert len(TOOL_ORDER) == 4

    def test_specs_carry_toolconfig_shape(self):
        for spec in TOOL_SPECS.values():
            assert spec.description
            assert spec.required
            for name in spec.required:
                assert name in spec.params


class TestStoreDispatch:
    def test_unknown_tool_rejected(self, store):
        with pytest.raises(ToolError, match="unknown tool"):
            store.call("query_weather", {})

    def test_missing_required_arg(self, store):
        with pytest.raises(ToolError, match="missing required arg"):
            store.call("get_detection", {})

    def test_duplicate_event_ids_rejected(self):
        with pytest.raises(ToolError, match="duplicate event_id"):
            ContextStore(events=[synthetic_event(), synthetic_event()])

    def test_unknown_camera_zone(self, store):
        with pytest.raises(ToolError, match="unknown camera_id"):
            store.call("query_zone", {"camera_id": "ghost-cam"})


class TestGetDetection:
    def test_aggregates_and_rounds(self):
        ev = synthetic_event(detections=[
            det("animal", 0.912345, "deer"),
            det("animal", 0.333333),
            det("person", 0.6),
        ])
        s = ContextStore(events=[ev], zones=synthetic_zones(ev["camera_id"]))
        res = s.call("get_detection", {"event_id": "ev-test"})
        assert res["counts_by_label"] == {"animal": 2, "person": 1}
        assert res["max_conf_by_label"] == {"animal": 0.9123, "person": 0.6}
        assert res["night"] == ev["night"]
        assert res["pass"] == 1
        assert res["reanalysis_of"] is None
        assert res["model"] == {"runtime": "cv2.dnn (synthetic-test)"}
        assert "closed-set:v0" in res["species_scope"]

    def test_unknown_event(self, store):
        with pytest.raises(ToolError, match="unknown event_id"):
            store.call("get_detection", {"event_id": "nope"})


class TestQueryHour:
    def _hour(self, at, camera="missouri-L00"):
        ev = synthetic_event(camera_id=camera, captured_at=at)
        s = ContextStore(events=[ev], zones=synthetic_zones(camera))
        return s.call("query_hour", {"camera_id": camera, "at": at,
                                     "clock_source": policy.CLOCK_PINNED})

    def test_restricted_zone_night_curfew(self):
        res = self._hour("2026-03-10T23:00:00-06:00")
        assert res["restricted"] is True
        assert res["curfew_window"] == ["18:00", "06:00"]
        assert res["curfew_active"] is True
        assert res["hour"] == 23

    def test_restricted_zone_day_no_curfew(self):
        res = self._hour("2026-03-10T12:00:00-06:00")
        assert res["restricted"] is True
        assert res["curfew_active"] is False

    def test_open_zone_has_no_curfew(self):
        res = self._hour("2026-03-10T23:00:00-06:00", camera="missouri-L05")
        assert res["restricted"] is False
        assert res["curfew_active"] is None

    def test_clock_source_tag_passes_through(self):
        res = self._hour("2026-03-10T12:00:00-06:00")
        assert res["clock_source"] == policy.CLOCK_PINNED


class TestQueryHistory:
    def _store(self):
        events = [
            synthetic_event(event_id="h-old",
                            captured_at="2026-03-07T10:00:00-06:00",
                            detections=[det("animal", 0.8, "zebra")]),
            synthetic_event(event_id="h-person-1",
                            captured_at="2026-03-09T08:00:00-06:00",
                            detections=[det("person", 0.3)]),
            synthetic_event(event_id="h-after",
                            captured_at="2026-03-10T14:00:00-06:00",
                            detections=[det("person", 0.9)]),
            synthetic_event(event_id="h-now",
                            captured_at="2026-03-10T12:00:00-06:00",
                            detections=[det("animal", 0.5, "deer")]),
        ]
        return ContextStore(events=events,
                            zones=synthetic_zones("missouri-L00"))

    def test_lookback_excludes_old_future_and_current(self):
        # h-old is 74h before h-now → outside the 72h window;
        # h-after is later than h-now → not history; h-now is excluded.
        res = self._store().call("query_history", {
            "camera_id": "missouri-L00", "lookback_hours": 72,
            "exclude_event_id": "h-now"})
        assert res["events"] == 1
        assert res["by_label"] == {"person": 1}
        assert res["person_events"] == 1
        assert res["last_person_at"] == "2026-03-09T08:00:00-06:00"
        assert res["species_seen"] == []
        assert res["max_conf"] == 0.3

    def test_longer_lookback_includes_old(self):
        res = self._store().call("query_history", {
            "camera_id": "missouri-L00", "lookback_hours": 100,
            "exclude_event_id": "h-now"})
        assert res["events"] == 2
        assert res["species_seen"] == ["zebra"]

    def test_default_lookback_from_policy(self):
        res = self._store().call("query_history", {
            "camera_id": "missouri-L00", "exclude_event_id": "h-now"})
        assert res["lookback_hours"] == policy.HISTORY_LOOKBACK_HOURS

    def test_empty_history(self):
        ev = synthetic_event(camera_id="missouri-L02",
                             captured_at="2026-03-10T12:00:00-06:00")
        s = ContextStore(events=[ev], zones=synthetic_zones("missouri-L02"))
        res = s.call("query_history", {"camera_id": "missouri-L02",
                                       "exclude_event_id": ev["event_id"]})
        assert res["events"] == 0
        assert res["max_conf"] is None
        assert res["last_person_at"] is None
