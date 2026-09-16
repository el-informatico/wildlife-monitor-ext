"""The tool-calling triage loop — one event in, one traced decision out.

The loop owns transport, not policy: it relays the adapter's tool calls
to the :class:`~agent.tools.ContextStore`, records every step verbatim,
enforces the step budget, and closes with the documented safe default
if no decision lands inside it. *How* decisions are made is entirely
the adapter's business (:mod:`agent.adapters`).

Trace shape (``wfm-triage-trace/v1``)::

    {"trace_id", "schema", "backend", "event_id", "camera_id",
     "captured_at", "action": <action record>, "steps": [...]}

step kinds, numbered 1..n in order:

- ``tool_call``   — {"tool", "args"}: what the adapter asked
- ``tool_result`` — {"tool", "result"}: what the store answered
- ``tool_error``  — {"tool", "error"}: a call the store could not serve
- ``decision``    — {"action", "rationale", "policy_refs"} and, only
  when the loop itself closed the trace, {"fallback": "budget"}

The trace is byte-deterministic: no wall-clock anywhere, canonical
JSON, id derived by sha256 (:mod:`agent.trace`). An event that is not
registered in the store is not an error — ``get_detection`` fails, the
error is recorded, and the safe default fires, all inside the trace.
"""

from __future__ import annotations

from typing import Any

from . import actions, policy
from .adapters import Decision, LoopState, RulesAdapter, ToolCall
from .tools import ContextStore, ToolError
from .trace import SCHEMA, finalize


def run_triage(event: dict[str, Any], store: ContextStore,
               adapter: Any | None = None,
               max_steps: int | None = None) -> dict[str, Any]:
    """Run the loop for one event and return the finalized trace.

    ``adapter`` defaults to the deterministic reference backend;
    ``max_steps`` defaults to :data:`agent.policy.MAX_STEPS`.
    """
    adapter = RulesAdapter() if adapter is None else adapter
    budget = policy.MAX_STEPS if max_steps is None else max_steps
    state = LoopState(event=event)

    def record(step: dict[str, Any]) -> dict[str, Any]:
        step["step"] = len(state.steps) + 1
        state.steps.append(step)
        return step

    decision: Decision | None = None
    fallback: str | None = None
    while decision is None and len(state.steps) < budget:
        nxt = adapter.next_step(state)
        if isinstance(nxt, Decision):
            decision = nxt
            break
        if not isinstance(nxt, ToolCall):
            raise TypeError(f"adapter {adapter.name!r} returned {nxt!r}; "
                            "expected ToolCall or Decision")
        record({"kind": "tool_call", "tool": nxt.tool, "args": nxt.args})
        try:
            result = store.call(nxt.tool, nxt.args)
        except ToolError as exc:
            record({"kind": "tool_error", "tool": nxt.tool, "error": str(exc)})
            continue
        record({"kind": "tool_result", "tool": nxt.tool, "result": result})

    if decision is None:
        decision = Decision(
            "approval_queue",
            {"priority": "high",
             "reason": f"step budget ({budget}) exhausted before the "
                       "adapter decided — safe default: human review, "
                       "never a silent drop or an automatic action",
             **actions.person_flags(event)},
            (f"adapter {adapter.name!r} did not return a Decision within "
             f"{budget} recorded steps",),
            ("MAX_STEPS",),
        )
        fallback = "budget"

    action = decision.to_action()
    final_step: dict[str, Any] = {
        "kind": "decision",
        "action": action,
        "rationale": list(decision.rationale),
        "policy_refs": list(decision.policy_refs),
    }
    if fallback is not None:
        final_step["fallback"] = fallback
    record(final_step)

    return finalize({
        "schema": SCHEMA,
        "backend": adapter.name,
        "event_id": event["event_id"],
        "camera_id": event["camera_id"],
        "captured_at": event["captured_at"],
        "action": action,
        "steps": state.steps,
    })
