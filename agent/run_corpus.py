"""Kill-switch B, executed: rules-only baseline vs the context agent over
the whole 200-event corpus.

Both lanes run the SAME events through the SAME trace schema — the agent
with its four context tools (rules-v1), the baseline with no context at
all (rules-only-v1, identical perception thresholds). Where they agree
the context loop added nothing; where they diverge the trace says which
fact made the difference. The committed artifact
(``agent/artifacts/corpus_comparison.{json,md}``) is the evidence the G4
verdict cites; per-event traces go to ``processed/`` (reproducible,
gitignored).

Usage (repo root, offline):

    ../wildlife-monitor/.venv/bin/python -m agent.run_corpus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import policy, runtime
from .adapters import RulesAdapter
from .baseline import BASELINE_NAME, run_baseline
from .loop import run_triage
from .tools import ContextStore, curfew_active, parse_event_ts

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SCHEMA = "wfm-s4-corpus-comparison/v1"
TRACES_AGENT = runtime.PROCESSED_DIR / "s4_corpus_traces_agent.jsonl"
TRACES_BASELINE = runtime.PROCESSED_DIR / "s4_corpus_traces_baseline.jsonl"


def _context_row(event: dict[str, Any], zones: dict[str, dict[str, Any]]) -> dict:
    """The zone/clock facts the agent sees and the baseline cannot."""
    hour = parse_event_ts(event["captured_at"]).hour
    window = zones[event["camera_id"]]["restricted_window"]
    return {
        "camera_id": event["camera_id"],
        "dataset": event["dataset"],
        "hour": hour,
        "zone_restricted": zones[event["camera_id"]]["restricted"],
        "curfew_active": curfew_active(hour, tuple(window)) if window else None,
        "night_content_ir": event["night"]["content_ir"],
        "max_conf_by_label": _max_conf_by_label(event),
    }


def _max_conf_by_label(event: dict[str, Any]) -> dict[str, float]:
    mx: dict[str, float] = {}
    for d in event["detections"]:
        mx[d["label"]] = max(mx.get(d["label"], 0.0), d["confidence"])
    return mx


def _lane(action: dict[str, Any]) -> dict[str, Any]:
    out = {"action": action["type"]}
    if "priority" in action:
        out["priority"] = action["priority"]
    return out


def _comparison_row(event: dict[str, Any], agent: dict[str, Any],
                    baseline: dict[str, Any],
                    zones: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = {"event_id": event["event_id"], **_context_row(event, zones),
           "agent": _lane(agent["action"]), "baseline": _lane(baseline["action"])}
    row["same_action"] = (row["agent"]["action"] == row["baseline"]["action"])
    row["diverges"] = (row["agent"] != row["baseline"])
    return row


def run_comparison() -> dict[str, Any]:
    events, zones = runtime.corpus_events_and_zones()
    store = ContextStore(events=events, zones=zones)

    rows, agent_traces, baseline_traces = [], [], []
    for event in events:
        agent_trace = run_triage(event, store)
        baseline_trace = run_baseline(event)
        agent_traces.append(agent_trace)
        baseline_traces.append(baseline_trace)
        rows.append(_comparison_row(event, agent_trace, baseline_trace, zones))

    TRACES_AGENT.write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in agent_traces))
    TRACES_BASELINE.write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in baseline_traces))

    def _counts(lane: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            key = row[lane]["action"]
            if "priority" in row[lane]:
                key += f":{row[lane]['priority']}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def _escalations(lane: str) -> int:
        return sum(n for k, n in _counts(lane).items()
                   if k.split(":")[0] == "escalate")

    person_rows = [r for r in rows if "person" in r["max_conf_by_label"]]
    divergent = [r for r in rows if r["diverges"]]
    reanalysis_ids = sorted(r["event_id"] for r in rows
                            if r["agent"]["action"] == "night_reanalysis")

    return {
        "schema": SCHEMA,
        "agent_backend": RulesAdapter.name,
        "baseline": BASELINE_NAME,
        "inputs": {
            "batches": "processed/e2e_union_v0_batches.jsonl",
            "manifest": "processed/manifest.parquet",
            "ir_flags": "processed/g3_ir_flags.json",
            "events": len(rows),
            "cameras": len(zones),
            "clock_source": policy.CLOCK_MANIFEST_SYNTHETIC,
            "history": "the corpus itself is the event history "
                       f"({policy.HISTORY_LOOKBACK_HOURS}h lookback, same camera)",
        },
        "action_counts": {"agent": _counts("agent"), "baseline": _counts("baseline")},
        "agreement": {
            "same_action": sum(r["same_action"] for r in rows),
            "same_action_but_priority_differs": sum(
                r["same_action"] and r["diverges"] for r in rows),
            "divergent_actions": sum(not r["same_action"] for r in rows),
        },
        "escalation_load": {
            "note": "events that page a human immediately (escalate) vs queue "
                    "for scheduled review — the operational cost the "
                    "kill-switch question is about",
            "agent_escalates": _escalations("agent"),
            "baseline_escalates": _escalations("baseline"),
            "agent_person_events_queued": len(
                [r for r in person_rows if r["agent"]["action"] == "approval_queue"]),
            "baseline_person_events_escalated": len(
                [r for r in person_rows if r["baseline"]["action"] == "escalate"]),
        },
        "person_events": person_rows,
        "vehicle_events": [r for r in rows if "vehicle" in r["max_conf_by_label"]],
        "night_reanalysis": {
            "events": len(reanalysis_ids),
            "event_ids": reanalysis_ids,
            "note": "both lanes fire this identically — the perception rule "
                    "is shared, so these are agreement by construction",
        },
        "divergent_events": divergent,
    }


def write_markdown(comp: dict[str, Any]) -> str:
    """The human-readable half of the artifact — same data, table form."""
    counts = comp["action_counts"]
    agree = comp["agreement"]
    esc = comp["escalation_load"]

    def _lane_cell(lane: dict[str, str]) -> str:
        return lane["action"] + (f" ({lane['priority']})"
                                 if "priority" in lane else "")

    lines = [
        "# S4 kill-switch B — rules-only baseline vs context agent, corpus run",
        "",
        f"- agent backend: `{comp['agent_backend']}` (4 context tools) vs "
        f"baseline `{comp['baseline']}` (no context, identical perception "
        "thresholds)",
        f"- corpus: {comp['inputs']['events']} events / "
        f"{comp['inputs']['cameras']} cameras, clock "
        f"`{comp['inputs']['clock_source']}`; history = the corpus itself",
        f"- agreement: **{agree['same_action']} same action** of "
        f"{comp['inputs']['events']}, of which "
        f"{agree['same_action_but_priority_differs']} differ in priority; "
        f"**{agree['divergent_actions']} diverge outright**",
        "",
        "## Action counts per lane",
        "",
        "| action (priority) | agent | baseline |",
        "|---|---|---|",
    ]
    keys = sorted(counts["agent"] | counts["baseline"])
    for key in keys:
        lines.append(f"| {key} | {counts['agent'].get(key, 0)} | "
                     f"{counts['baseline'].get(key, 0)} |")

    lines += [
        "",
        "## Every person event — the kill-switch evidence",
        "",
        "The baseline escalates every confirmed person (≥ "
        f"{policy.PERSON_CONFIRM_CONF}) because it cannot see zone or clock; "
        "the agent keeps every one of them in front of a human but changes "
        "the urgency when the context says no curfew is in force. No person "
        "event is ever dropped by either lane.",
        "",
        "| event | camera | hour | zone | curfew | person conf | agent | baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in comp["person_events"]:
        conf = r["max_conf_by_label"].get("person", 0.0)
        lines.append(
            f"| {r['event_id']} | {r['camera_id']} | {r['hour']} | "
            f"{'restricted' if r['zone_restricted'] else 'open'} | "
            f"{r['curfew_active']} | {conf:.4f} | "
            f"{_lane_cell(r['agent'])} | {_lane_cell(r['baseline'])} |")

    lines += [
        "",
        "## Divergent events",
        "",
    ]
    if comp["divergent_events"]:
        lines += [
            "| event | camera | hour | zone | curfew | agent | baseline |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in comp["divergent_events"]:
            lines.append(
                f"| {r['event_id']} | {r['camera_id']} | {r['hour']} | "
                f"{'restricted' if r['zone_restricted'] else 'open'} | "
                f"{r['curfew_active']} | {_lane_cell(r['agent'])} | "
                f"{_lane_cell(r['baseline'])} |")
    else:
        lines.append("_none_")

    nr = comp["night_reanalysis"]
    lines += [
        "",
        "## Night re-analysis",
        "",
        f"{nr['events']} events fire `night_reanalysis` in BOTH lanes "
        "(shared perception rule — agreement by construction): "
        + ", ".join(f"`{e}`" for e in nr["event_ids"]),
        "",
        "Per-event traces: `processed/s4_corpus_traces_agent.jsonl` and "
        "`processed/s4_corpus_traces_baseline.jsonl` (regenerated by "
        "`python -m agent.run_corpus`; gitignored like all of `processed/`).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    comp = run_comparison()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "corpus_comparison.json").write_text(
        json.dumps(comp, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (ARTIFACTS_DIR / "corpus_comparison.md").write_text(write_markdown(comp))

    print(f"corpus: {comp['inputs']['events']} events, "
          f"{comp['inputs']['cameras']} cameras")
    print(f"same action: {comp['agreement']['same_action']} "
          f"(+{comp['agreement']['same_action_but_priority_differs']} priority-only) / "
          f"divergent: {comp['agreement']['divergent_actions']}")
    print(f"escalates — agent: {comp['escalation_load']['agent_escalates']}, "
          f"baseline: {comp['escalation_load']['baseline_escalates']}")
    print("person events: "
          f"{comp['escalation_load']['agent_person_events_queued']} queued (agent) vs "
          f"{comp['escalation_load']['baseline_person_events_escalated']} "
          "escalated (baseline)")
    print(f"wrote {ARTIFACTS_DIR / 'corpus_comparison.json'} and .md")


if __name__ == "__main__":
    main()
