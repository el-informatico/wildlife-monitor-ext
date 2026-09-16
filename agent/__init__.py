"""Sprint 4 — the triage agent (Gate G4, Agentic Vision lane).

A tool-calling loop over the E2E perception output: for every detection
batch the agent grounds on the perception record (``get_detection``) and
consults three context tools (``query_zone``, ``query_hour``,
``query_history``) before choosing one of four conditional actions
(``archive``, ``night_reanalysis``, ``approval_queue``, ``escalate``).
Every step — tool calls, tool results, the decision and its rationale —
lands in a byte-deterministic JSON trace.

The decider behind the loop is an *adapter*: :class:`agent.adapters.
RulesAdapter` is the deterministic reference backend (no cloud calls,
the default everywhere), and :class:`agent.adapters.BedrockAdapter`
documents the real-LLM swap point without making any network call in
this repo. Everything in the core modules is stdlib-only and covered by
the offline test suite; heavy perception (the live night re-analysis
second pass) lives in ``agent/runtime.py`` behind lazy imports.
"""
