"""Scenario assembly — plain rows to agent events and zones. Pure, stdlib-only.

The heavy side (reading the union batches, the manifest, the G3 IR-flag
cache; running the live re-analysis pass) lives in ``agent/runtime.py``
behind lazy imports. Everything here is dict-in, dict-out and
unit-tested offline.

Corpus provenance recap (constants and full stories in
:mod:`agent.policy` docstrings):

- zone policy is the ETL's per-camera ``zone_restricted`` flag
  (SIMULATED context, consistent per camera) plus this sprint's fixed
  restricted window;
- the manifest clock is a SYNTHETIC uniform assignment (G3 measured its
  agreement with content at ≈ chance), so corpus events carry
  ``clock_source: manifest-synthetic`` with a fixed per-dataset UTC
  offset, and showcase scenarios pin explicit clocks
  (``clock_source: pinned-demo``);
- night is ALWAYS the G3 content-IR flag, never the synthetic is_night.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from . import policy
from .tools import parse_event_ts


def attach_tz(naive_ts: str, dataset: str) -> str:
    """Append the dataset's fixed UTC offset to a timezone-naive manifest ts."""
    try:
        offset = policy.DATASET_TZ_OFFSET[dataset]
    except KeyError:
        raise ValueError(f"no tz offset registered for dataset {dataset!r}; "
                         f"known: {sorted(policy.DATASET_TZ_OFFSET)}") from None
    if datetime.fromisoformat(naive_ts).tzinfo is not None:
        raise ValueError(f"expected timezone-naive manifest ts, got {naive_ts!r}")
    return naive_ts + offset


def zones_from_manifest_rows(
        rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-camera zone-policy records from manifest rows.

    A row needs ``camera_id`` and ``zone_restricted``. The flag is
    consistent per camera across all 26,177 manifest rows; an
    inconsistency raises here rather than silently picking a side.
    """
    zones: dict[str, dict[str, Any]] = {}
    for row in rows:
        cam = row["camera_id"]
        restricted = bool(row["zone_restricted"])
        if cam in zones:
            if zones[cam]["restricted"] != restricted:
                raise ValueError(
                    f"zone_restricted inconsistent for camera {cam!r}")
            continue
        zones[cam] = {
            "zone_id": f"Z-{cam}",
            "zone_name": f"simulated sector — {cam}",
            "restricted": restricted,
            "restricted_window": (list(policy.RESTRICTED_WINDOW)
                                  if restricted else None),
            "policy_note": "simulated zone policy (S4): per-camera "
                           "restricted flag from the ETL synthetic layer; "
                           "restricted window fixed at 18:00–06:00 local",
        }
    return zones


def build_event(*, event_id: str, camera_id: str, dataset: str,
                source_key: str, captured_at: str, clock_source: str,
                detections: list[dict[str, Any]], night_content_ir: bool,
                night_synthetic_flag: bool, model: dict[str, Any] | str,
                pass_: int = 1,
                reanalysis_of: str | None = None) -> dict[str, Any]:
    """Assemble one agent event, validating what must never be wrong.

    ``captured_at`` must be tz-aware ISO-8601 (the corpus path gets
    there via :func:`attach_tz`; showcase fixtures pin it directly).
    ``model`` may be the plain runtime string (normalized to
    ``{"runtime": ...}``) or a dict — the pass-2 event adds its
    re-parameterization keys there.
    """
    parse_event_ts(captured_at)   # tz-aware, raises otherwise
    if clock_source not in (policy.CLOCK_PINNED, policy.CLOCK_MANIFEST_SYNTHETIC):
        raise ValueError(f"unknown clock_source {clock_source!r}; expected "
                         f"{policy.CLOCK_PINNED!r} or "
                         f"{policy.CLOCK_MANIFEST_SYNTHETIC!r}")
    if isinstance(model, str):
        model = {"runtime": model}
    return {
        "event_id": event_id,
        "camera_id": camera_id,
        "dataset": dataset,
        "source_key": source_key,
        "captured_at": captured_at,
        "clock_source": clock_source,
        "night": {"content_ir": bool(night_content_ir),
                  "synthetic_flag": bool(night_synthetic_flag)},
        "model": model,
        "detections": list(detections),
        "pass": pass_,
        "reanalysis_of": reanalysis_of,
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures — one set of builders shared by the offline tests,
# the corpus runner and the demo, so tests exercise the real assembly
# path rather than a parallel one.
#


def det(label: str, conf: float, species: str | None = None) -> dict[str, Any]:
    """One normalized detection record (contract v1 shape)."""
    d: dict[str, Any] = {"bbox": [0.1, 0.1, 0.5, 0.5],
                         "confidence": conf, "label": label}
    if species is not None:
        d["species"] = species
    return d


def synthetic_event(**overrides) -> dict[str, Any]:
    """A default synthetic event: confirmed day-frame animal on camera
    missouri-L00 (a restricted zone, but noon → curfew inactive), pinned
    clock. Override anything; :func:`build_event` validates the result."""
    kwargs: dict[str, Any] = dict(
        event_id="ev-test",
        camera_id="missouri-L00",
        dataset="missouri",
        source_key="missouri/Set1/TEST/SEQ00000/SEQ00000_IMG_0000.JPG",
        captured_at="2026-03-10T12:00:00-06:00",
        clock_source=policy.CLOCK_PINNED,
        detections=[det("animal", 0.9, "deer")],
        night_content_ir=False,
        night_synthetic_flag=False,
        model="cv2.dnn (synthetic-test)",
    )
    kwargs.update(overrides)
    return build_event(**kwargs)


def synthetic_zones(*camera_ids: str) -> dict[str, dict[str, Any]]:
    """Zone policy for synthetic cameras — mirrors real corpus facts:
    missouri-L00 is restricted, every other id is open."""
    rows = [{"camera_id": c, "zone_restricted": c == "missouri-L00"}
            for c in camera_ids]
    return zones_from_manifest_rows(rows)
