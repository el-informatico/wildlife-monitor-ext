"""The agent's four tools, backed by an offline :class:`ContextStore`.

Tool set (fixed by the G4 brief: zone / hour / history + perception):

=================  =====================================================
``get_detection``  the perception record under triage — labels, per-label
                   max confidence, night flags, model runtime, pass.
``query_zone``     camera → zone policy: restricted or not, curfew window.
``query_hour``     event clock → local hour, curfew state for the zone.
``query_history``  what this camera reported recently (lookback window).
=================  =====================================================

Each tool carries the JSON-schema-ish spec a Bedrock ``toolConfig``
would need, so the same registry drives both the offline loop and the
documented cloud adapter (:mod:`agent.adapters.BedrockAdapter`).

Production mapping — the tools are thin reads over the ingest store
provisioned in ``infra/`` (see its README's access-pattern table):

- ``query_history`` → base index ``pk = CAM#<camera>`` timeline,
  minus the current event, filtered to the lookback window;
- ``get_detection`` → gsi-source ``gs2pk = SRC#<media key>``;
- ``query_zone``    → the (small, static) zone-policy table;
- ``query_hour``    → zone policy + the event timestamp.

Offline, the store is assembled from the E2E union outputs (detection
batches + manifest join + the G3 content-IR flag cache) — no AWS calls
anywhere. History here is the same corpus minus the event under triage;
in production it is whatever the timeline holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

from . import policy


class ToolError(ValueError):
    """A tool call the store cannot serve (unknown name or bad args)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Any]        # name → type/doc, toolConfig-shaped
    required: tuple[str, ...]


TOOL_SPECS: dict[str, ToolSpec] = {
    "get_detection": ToolSpec(
        name="get_detection",
        description="Ground on the perception record under triage: deployed "
                    "detections with label/species/confidence/bbox, per-label "
                    "max confidence, night flags (content-IR authoritative), "
                    "model runtime and pass number.",
        params={"event_id": "string — the image/event id"},
        required=("event_id",),
    ),
    "query_zone": ToolSpec(
        name="query_zone",
        description="Zone policy for a camera: zone id, restricted flag, "
                    "curfew window (local clock) and the policy note.",
        params={"camera_id": "string — camera id from the event"},
        required=("camera_id",),
    ),
    "query_hour": ToolSpec(
        name="query_hour",
        description="Local-clock reading for an event timestamp inside a "
                    "zone: hour, curfew_active for the zone's window, and "
                    "the clock provenance tag.",
        params={"camera_id": "string — camera id (zone lookup)",
                "at": "string — ISO-8601 timestamp of the event",
                "clock_source": "string — provenance tag carried from the event"},
        required=("camera_id", "at"),
    ),
    "query_history": ToolSpec(
        name="query_history",
        description="Prior perception events on the same camera within the "
                    "lookback window: event count, per-label counts, person "
                    "events, last person timestamp, species seen, max "
                    "confidence.",
        params={"camera_id": "string — camera id",
                "lookback_hours": "number — window length (default 72)",
                "exclude_event_id": "string — the event under triage"},
        required=("camera_id",),
    ),
}

#: The order the reference backend consults tools in (uniform on purpose:
#: every decision grounds on perception, then checks zone, hour, history —
#  the full context set — before deciding).
TOOL_ORDER = ("get_detection", "query_zone", "query_hour", "query_history")


def curfew_active(hour: int, window: tuple[str, str] | None) -> bool | None:
    """Is ``hour`` inside ``window``? Handles midnight-crossing windows.

    ``None`` window (unrestricted zone) → ``None`` (no curfew exists).
    """
    if window is None:
        return None
    start_h, start_m = (int(p) for p in window[0].split(":"))
    end_h, end_m = (int(p) for p in window[1].split(":"))
    start, end = start_h + start_m / 60.0, end_h + end_m / 60.0
    h = hour % 24
    if start <= end:                     # e.g. 08:00–17:00
        return start <= h < end
    return h >= start or h < end         # e.g. 18:00–06:00 (wraps)


def parse_event_ts(value: str) -> datetime:
    """Parse an event timestamp; must be tz-aware ISO-8601."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ToolError(f"event timestamp must be tz-aware, got {value!r}")
    return parsed


@dataclass
class ContextStore:
    """Offline tool backend: events (perception) + zones (policy)."""

    events: list[dict[str, Any]] = field(default_factory=list)
    zones: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_event: dict[str, dict[str, Any]] = {}
        self._by_camera: dict[str, list[dict[str, Any]]] = {}
        for ev in self.events:
            if ev["event_id"] in self._by_event:
                raise ToolError(f"duplicate event_id {ev['event_id']!r}")
            self._by_event[ev["event_id"]] = ev
            self._by_camera.setdefault(ev["camera_id"], []).append(ev)
        for cam in self._by_camera:
            self._by_camera[cam].sort(key=lambda e: e["captured_at"])

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        if name not in TOOL_SPECS:
            raise ToolError(f"unknown tool {name!r}; known: {sorted(TOOL_SPECS)}")
        missing = [p for p in TOOL_SPECS[name].required if p not in args]
        if missing:
            raise ToolError(f"{name}: missing required arg(s) {missing}")
        return getattr(self, f"_{name}")(**dict(args))

    # -- tool implementations ------------------------------------------------

    def _get_detection(self, event_id: str) -> dict[str, Any]:
        ev = self._by_event.get(event_id)
        if ev is None:
            raise ToolError(f"unknown event_id {event_id!r}")
        dets = ev["detections"]
        by_label: dict[str, int] = {}
        max_conf: dict[str, float] = {}
        for d in dets:
            by_label[d["label"]] = by_label.get(d["label"], 0) + 1
            max_conf[d["label"]] = max(max_conf.get(d["label"], 0.0), d["confidence"])
        return {
            "event_id": ev["event_id"],
            "camera_id": ev["camera_id"],
            "detections": dets,
            "counts_by_label": by_label,
            "max_conf_by_label": {k: round(v, 4) for k, v in max_conf.items()},
            "night": ev["night"],
            "captured_at": ev["captured_at"],
            "clock_source": ev["clock_source"],
            "pass": ev["pass"],
            "reanalysis_of": ev.get("reanalysis_of"),
            "model": ev["model"],
            "species_scope": "closed-set:v0 (out-of-vocabulary species map to "
                             "the nearest trained class — G3 failure case 5; "
                             "species never gates an action, it only annotates)",
        }

    def _query_zone(self, camera_id: str) -> dict[str, Any]:
        zone = self.zones.get(camera_id)
        if zone is None:
            raise ToolError(f"unknown camera_id {camera_id!r}")
        return zone

    def _query_hour(self, camera_id: str, at: str, clock_source: str | None = None) -> dict[str, Any]:
        zone = self._query_zone(camera_id)
        ts = parse_event_ts(at)
        return {
            "camera_id": camera_id,
            "local_time": at,
            "hour": ts.hour,
            "clock_source": clock_source,
            "zone_id": zone["zone_id"],
            "restricted": zone["restricted"],
            "curfew_window": zone["restricted_window"],
            "curfew_active": curfew_active(ts.hour, zone["restricted_window"]),
        }

    def _query_history(self, camera_id: str, lookback_hours: float | None = None,
                       exclude_event_id: str | None = None) -> dict[str, Any]:
        hours = policy.HISTORY_LOOKBACK_HOURS if lookback_hours is None else lookback_hours
        cam_events = self._by_camera.get(camera_id)
        if cam_events is None:
            raise ToolError(f"unknown camera_id {camera_id!r}")
        current = self._by_event.get(exclude_event_id)
        cutoff = None
        if current is not None:
            cutoff = parse_event_ts(current["captured_at"]) - timedelta(hours=hours)
        by_label: dict[str, int] = {}
        person_events = 0
        last_person_at = None
        species_seen: set[str] = set()
        max_conf = 0.0
        n = 0
        for ev in cam_events:
            if exclude_event_id is not None and ev["event_id"] == exclude_event_id:
                continue
            if cutoff is not None:
                if parse_event_ts(ev["captured_at"]) < cutoff:
                    continue
                if parse_event_ts(ev["captured_at"]) > parse_event_ts(current["captured_at"]):
                    continue   # history = strictly before the event under triage
            n += 1
            for d in ev["detections"]:
                by_label[d["label"]] = by_label.get(d["label"], 0) + 1
                max_conf = max(max_conf, d["confidence"])
                if d.get("species"):
                    species_seen.add(d["species"])
                if d["label"] == "person":
                    person_events += 1
                    last_person_at = ev["captured_at"]
        return {
            "camera_id": camera_id,
            "lookback_hours": hours,
            "events": n,
            "by_label": by_label,
            "person_events": person_events,
            "last_person_at": last_person_at,
            "species_seen": sorted(species_seen),
            "max_conf": round(max_conf, 4) if n else None,
        }
