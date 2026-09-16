"""Run the G4 showcase: seven real corpus payloads, traced decisions, one
live second perception pass.

What lands where (all committed — this IS the demo artifact):

- ``agent/demo/artifacts/traces/<id>.json``    — full trace per scenario
- ``agent/demo/artifacts/traces/<id>_pass2.json`` — the s2 pass-2 trace
- ``agent/demo/artifacts/TRANSCRIPT.md``       — the human-readable run

Every ``expect`` in the fixture is asserted at run time: if the agent's
decision ever stops matching the documented one, the demo fails loudly
instead of silently rendering a stale transcript.

Usage (repo root, offline — s2 loads the ONNX detector, a few seconds):

    ../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import policy, runtime
from ..baseline import run_baseline
from ..loop import run_triage
from ..scenario import attach_tz, build_event, zones_from_manifest_rows
from ..tools import ContextStore

DEMO_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = DEMO_DIR / "showcase_scenarios.json"
TRACES_DIR = DEMO_DIR / "artifacts" / "traces"
TRANSCRIPT_PATH = DEMO_DIR / "artifacts" / "TRANSCRIPT.md"


def _detections_line(event: dict[str, Any]) -> str:
    if not event["detections"]:
        return "no detections (empty batch)"
    return ", ".join(
        f"{d['label']} {d['confidence']:.4f}"
        + (f" ({d['species']})" if d.get("species") else "")
        for d in sorted(event["detections"],
                        key=lambda d: -d["confidence"]))
    + f" — content-IR night: {str(event['night']['content_ir']).lower()}"


def _decision_cell(action: dict[str, Any]) -> str:
    return action["type"] + (f" ({action['priority']})"
                             if "priority" in action else "")


def _rationale_block(trace: dict[str, Any]) -> list[str]:
    step = trace["steps"][-1]
    lines = [f"decision: **{_decision_cell(trace['action'])}**"]
    lines += [f"  - {line}" for line in step["rationale"]]
    if step.get("policy_refs"):
        lines.append(f"  - policy refs: {', '.join(step['policy_refs'])}")
    return lines


def build_scenario_event(spec: dict[str, Any], batch: dict[str, Any],
                         man_row: dict[str, Any],
                         ir_flag: bool) -> dict[str, Any]:
    """Fixture spec + corpus rows → one agent event (validation via
    :func:`agent.scenario.build_event`)."""
    src = batch["source"]
    ctx = spec["context"]
    if ctx["mode"] == "pinned":
        camera_id, captured_at = ctx["camera_id"], ctx["captured_at"]
        clock_source = policy.CLOCK_PINNED
    else:
        camera_id = src["camera_id"]
        captured_at = attach_tz(man_row["camera_ts"], src["dataset"])
        clock_source = policy.CLOCK_MANIFEST_SYNTHETIC
    return build_event(
        event_id=src["image_id"],
        camera_id=camera_id,
        dataset=src["dataset"],
        source_key=src["key"],
        captured_at=captured_at,
        clock_source=clock_source,
        detections=batch["detections"],
        night_content_ir=ir_flag,
        night_synthetic_flag=bool(src["is_night"]),
        model=batch["model"],
    )


def run_demo() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text())
    batches = runtime.load_union_batches()
    manifest_rows = runtime.load_manifest_rows()   # incl. rel_path for s2
    ir_flags = runtime.load_ir_flags()
    batch_by_id = {b["source"]["image_id"]: b for b in batches}
    man_by_id = {r["image_id"]: r for r in manifest_rows}
    zones = zones_from_manifest_rows(
        {"camera_id": r["location_id"], "zone_restricted": r["zone_restricted"]}
        for r in manifest_rows)

    needs_pixels = any(s.get("reanalysis") for s in fixture["scenarios"])
    detector_bundle = runtime.load_detector() if needs_pixels else None

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for spec in fixture["scenarios"]:
        iid = spec["image_id"]
        event = build_scenario_event(spec, batch_by_id[iid], man_by_id[iid],
                                     ir_flags.get(iid, False))
        # One scenario at a time in the store: showcase history is empty
        # by design (the corpus run exercised history; here each event
        # stands alone so the transcript reads cleanly).
        store = ContextStore(events=[event], zones=zones)
        trace = run_triage(event, store)
        baseline = run_baseline(event)

        got = trace["action"]["type"]
        expected = spec.get("expect", spec.get("expect_pass1"))
        if got != expected:
            raise AssertionError(
                f"{spec['id']} ({iid}): expected {expected!r}, "
                f"agent decided {got!r}")

        entry: dict[str, Any] = {
            "spec": spec, "event": event, "trace": trace,
            "baseline_trace": baseline, "manifest_row": man_by_id[iid],
        }
        (TRACES_DIR / f"{spec['id']}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        if spec.get("reanalysis"):
            if got != "night_reanalysis":
                raise AssertionError(
                    f"{spec['id']}: reanalysis scenario expected "
                    f"night_reanalysis, got {got!r}")
            detector, cv2, e2e, mdv6 = detector_bundle
            bgr = runtime.load_image_bgr(man_by_id[iid]["rel_path"])
            dets2 = runtime.night_reanalysis_detections(
                bgr, detector, cv2, e2e, mdv6)
            event2 = runtime.build_pass2_event(event, dets2)
            trace2 = run_triage(event2, ContextStore(events=[event2],
                                                     zones=zones))
            got2 = trace2["action"]["type"]
            if got2 == spec.get("expect_pass2_not"):
                raise AssertionError(
                    f"{spec['id']}: pass-2 decision {got2!r} — the loop "
                    f"guard was supposed to prevent re-triggering")
            entry["pass2_event"] = event2
            entry["pass2_trace"] = trace2
            (TRACES_DIR / f"{spec['id']}_pass2.json").write_text(
                json.dumps(trace2, indent=2, sort_keys=True,
                           ensure_ascii=False) + "\n")

        results.append(entry)

    TRANSCRIPT_PATH.write_text(render_transcript(results))
    return {"scenarios": len(results),
            "transcript": str(TRANSCRIPT_PATH)}


def render_transcript(results: list[dict[str, Any]]) -> str:
    lines = [
        "# G4 demo transcript — perception output changing triage decisions",
        "",
        "Seven scenarios over real cv2.dnn corpus payloads (S2/E2E union",
        "run). Reproduce with:",
        "",
        "    ../wildlife-monitor/.venv/bin/python -m agent.demo.run_demo",
        "",
        "Every decision below is asserted against the fixture's",
        "`expect` at run time; full step-by-step traces (tool calls, tool",
        "results, rationale) sit next to this transcript in `traces/`.",
        "",
    ]
    for entry in results:
        spec, event, trace = entry["spec"], entry["event"], entry["trace"]
        ctx = spec["context"]
        lines.append(f"## {spec['id']} — {spec['image_id']}: {spec['title']}")
        lines.append("")
        lines.append(f"{spec['point']}.")
        lines.append("")
        lines.append(f"- perception (pass 1, corpus): {_detections_line(event)}")
        lines.append(f"- context: camera {event['camera_id']} "
                     f"({'restricted' if _zone_restricted(event, entry) else 'open'} zone), "
                     f"{event['captured_at']} — clock `{event['clock_source']}`")
        if ctx["mode"] == "pinned":
            lines.append(f"- pinned recontextualization — original corpus "
                         f"context: {ctx['original_corpus_context']}")
        lines.append(f"- rules-only baseline: "
                     f"**{_decision_cell(entry['baseline_trace']['action'])}**")
        lines.append("")
        lines += _rationale_block(trace)

        if "pass2_trace" in entry:
            ev2, tr2 = entry["pass2_event"], entry["pass2_trace"]
            lines.append("")
            lines.append(f"### pass 2 — live second cv2.dnn pass on "
                         f"`{entry['manifest_row']['rel_path']}`")
            lines.append("")
            lines.append(f"preprocess `{policy.REANALYSIS_PREPROCESS}`, "
                         f"decode floor {policy.REANALYSIS_CONF_FLOOR} "
                         f"(pass-1 floor context: the deploy stage ran at "
                         f"0.25):")
            lines.append("")
            if ev2["detections"]:
                lines.append("| label | confidence | species |")
                lines.append("|---|---|---|")
                for d in sorted(ev2["detections"],
                                key=lambda d: -d["confidence"]):
                    lines.append(f"| {d['label']} | {d['confidence']:.4f} | "
                                 f"{d.get('species', '—')} |")
            else:
                lines.append(f"_no detections at floor "
                             f"{policy.REANALYSIS_CONF_FLOOR}_")
            lines.append("")
            lines.append(f"The pass-2 output re-enters the loop as event "
                         f"`{ev2['event_id']}` (`reanalysis_of: "
                         f"{ev2['reanalysis_of']}`, pass 2 — the re-analysis "
                         f"never re-triggers):")
            lines.append("")
            lines += _rationale_block(tr2)
        lines.append("")

    lines += [
        "---",
        "",
        "Gate G4 asks for at least two distinct conditional actions driven",
        "by cv2.dnn output, documented in trace:",
        "",
        "- **night re-analysis** (s2): a marginal pass-1 animal changes the",
        "  agent's own next question — pass 2 re-runs cv2.dnn with the",
        "  motion-gate CLAHE recipe and the final action follows the pass-2",
        "  output, not the pass-1 one;",
        "- **escalation / approval** (s3–s6): person and vehicle confidences",
        "  from cv2.dnn route to escalation (restricted zone + curfew),",
        "  normal-priority human review (open zone, or below the confirm",
        "  bar), and a ranger check (vehicle + curfew) — each with its full",
        "  tool-call trace in `traces/`.",
        "",
        "Framing held throughout: every person event reaches a human",
        "(queue or escalation — never a drop, never an accusation); the",
        "simulated zone/hour layer is tagged in every trace",
        "(`pinned-demo` / `manifest-synthetic`).",
        "",
    ]
    return "\n".join(lines)


def _zone_restricted(event: dict[str, Any],
                     entry: dict[str, Any]) -> bool:
    """The zone fact the agent's query_zone step actually saw."""
    step = next(s for s in entry["trace"]["steps"]
                if s["kind"] == "tool_result" and s["tool"] == "query_zone")
    return bool(step["result"]["restricted"])


def main() -> None:
    summary = run_demo()
    print(f"showcase: {summary['scenarios']} scenarios, all expectations met")
    print(f"transcript: {summary['transcript']}")
    print(f"traces: {TRACES_DIR}/")


if __name__ == "__main__":
    main()
