"""Triage actions and their invariants.

Four conditional actions, one vocabulary shared by the agent and the
kill-switch B baseline:

- ``archive``           — routine log; no human attention requested.
- ``night_reanalysis``  — re-run perception with night preprocessing at
                          a lower decode floor; the pass-2 output re-
                          enters the loop as a new event.
- ``approval_queue``    — a human looks before anything else happens.
- ``escalate``          — a human looks NOW, with an evidence package;
                          still an approval request, never an accusation
                          (project framing: the queue is the product).

Invariants enforced here (tested):

- ``escalate`` and ``approval_queue`` ALWAYS carry
  ``approval_required: true`` — no person-related action fires without
  a human in the loop.
- Any action derived from an event containing a ``person`` detection
  carries ``face_blur_required: true`` (facial blur on every image with
  people is a project-wide rule; the blur itself is applied downstream
  when the evidence package is rendered — this sprint records the flag).
- ``night_reanalysis`` carries its full re-parameterization
  (preprocess + floor) so the trace alone can reproduce the second pass.
"""

from __future__ import annotations

from typing import Any

from . import policy

ACTION_TYPES = ("archive", "night_reanalysis", "approval_queue", "escalate")

#: Actions that put a human in the loop (unconditionally true for both).
REQUIRES_APPROVAL = frozenset({"approval_queue", "escalate"})


def build_action(action_type: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble + validate one action record; raises ValueError on a bad type."""
    if action_type not in ACTION_TYPES:
        raise ValueError(f"unknown action type {action_type!r}; expected one of {ACTION_TYPES}")
    action = {"type": action_type}
    if action_type in REQUIRES_APPROVAL:
        action["approval_required"] = True
    if params:
        action.update(params)
    _check(action)
    return action


def _check(action: dict[str, Any]) -> None:
    at = action["type"]
    if at in REQUIRES_APPROVAL and action.get("approval_required") is not True:
        raise ValueError(f"{at} must carry approval_required=True")
    if at == "night_reanalysis":
        for key in ("preprocess", "conf_floor"):
            if key not in action:
                raise ValueError(f"night_reanalysis must carry {key}")
    if at in ("approval_queue", "escalate", "archive"):
        if "reason" not in action:
            raise ValueError(f"{at} must carry a reason")
    if at in ("approval_queue", "escalate") and action.get("priority") not in ("normal", "high"):
        raise ValueError(f"{at} must carry priority normal|high")


def night_reanalysis_params(reason: str) -> dict[str, Any]:
    """The re-analysis re-parameterization, from policy constants."""
    return {
        "preprocess": policy.REANALYSIS_PREPROCESS,
        "conf_floor": policy.REANALYSIS_CONF_FLOOR,
        "deploy_conf": 0.25,  # pass-1 deploy bar, kept for provenance
        "reason": reason,
    }


def person_flags(event: dict[str, Any]) -> dict[str, Any]:
    """face_blur_required on any action over an event with person evidence."""
    has_person = any(d.get("label") == "person" for d in event.get("detections", []))
    return {"face_blur_required": True} if has_person else {}
