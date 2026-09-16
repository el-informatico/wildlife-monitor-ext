"""Canonical JSON and trace ids — byte-deterministic artifacts.

No wall-clock, no randomness, no dict-ordering dependence: canonical
JSON (sorted keys, tight separators) and a sha256-derived id. The same
inputs must produce byte-identical traces on every run — the committed
demo/corpus artifacts depend on this and a test pins it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Trace schema version. Bump on any breaking change to step kinds or
#: the header fields; consumers (demo renderer, corpus comparison) check it.
SCHEMA = "wfm-triage-trace/v1"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no insignificant space."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def compute_trace_id(trace: dict[str, Any]) -> str:
    """sha256 over the canonical trace minus the id field itself."""
    payload = {k: v for k, v in trace.items() if k != "trace_id"}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "sha256:" + digest[:16]


def finalize(trace: dict[str, Any]) -> dict[str, Any]:
    """Stamp schema (if absent) and the trace id; returns the same dict."""
    trace.setdefault("schema", SCHEMA)
    trace["trace_id"] = compute_trace_id(trace)
    return trace
