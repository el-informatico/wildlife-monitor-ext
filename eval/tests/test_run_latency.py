"""Offline tests for the S5 latency/cost stage (eval/run_latency.py).

Pure-logic coverage: the cost arithmetic, lane normalization from a
stored-artifact-shaped dict, the fresh-vs-stored renderer, and the
Graviton honesty (never a measured-looking number). No pixels, no cv2,
no subprocess — bench_cv2/bench_pytorch are exercised for real by the
runner itself, not here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import run_latency  # noqa: E402


# ---------------------------------------------------------------------------
# Cost arithmetic
#


def test_hours_per_1m_unit_math():
    # 3.6 ms/img × 1M = 3.6e6 ms = 1 h
    assert run_latency.hours_per_1m(3.6) == 1.0
    # 3600 ms/img (1 s/img) → 1M s ≈ 277.78 h
    assert run_latency.hours_per_1m(3600) == 1000.0


def test_cost_per_1m_price_math():
    # 1000 h × $0.089/hr
    assert run_latency.cost_per_1m(3600, 0.089) == 89.0
    assert run_latency.cost_per_1m(3.6, 0.0725) == 0.07


def test_cost_cheaper_on_graviton_price_only():
    """At equal ms, the cheaper hourly price wins — the only claim made."""
    ms = 150.0
    assert (run_latency.cost_per_1m(ms, 0.0725)
            < run_latency.cost_per_1m(ms, 0.089))


# ---------------------------------------------------------------------------
# Lane normalization
#


def _artifact(engine: str, forward_ms: float, e2e_ms: float) -> dict:
    infer = {"mean": forward_ms + 2, "p50": forward_ms,
             "p95": forward_ms + 30}
    infer_key = ("preprocess_plus_forward_plus_nms"
                 if engine.startswith("pytorch") else "forward")
    return {
        "engine": engine,
        "n_images": 800,
        "latency_ms": {
            "decode": {"mean": 10, "p50": 10, "p95": 20},
            "end_to_end": {"mean": e2e_ms + 2, "p50": e2e_ms,
                           "p95": e2e_ms + 35},
            infer_key: infer,
        },
        "model": {"file": "MDV6-yolov10-c.onnx", "sha256": "x" * 64},
        "params": {"conf_floor": 0.1, "warmup_images_excluded": 3},
        "versions": {"cv2": "5.0.0"},
    }


def test_lane_row_cv2_shape():
    row = run_latency.lane_row(
        {"lane": "cv2.dnn new (deploy)", "kind": "new engine"},
        _artifact("cv2-dnn-new", 125.13, 148.47))
    assert row["infer_stage"] == "forward"
    assert row["infer_ms"]["p50"] == 125.13
    assert row["e2e_ms"]["p50"] == 148.47
    assert row["images_per_s_at_p50"] == round(1000 / 148.47, 2)
    # hours/1M at 148.47 ms = 41.24 h; × 0.089 = $3.67
    assert row["hours_per_1m_at_e2e_p50"] == 41.24
    assert row["cost_usd_per_1m"]["c7i.large"] == 3.67
    assert row["cost_usd_per_1m"]["c7g.large"] == 2.99
    assert row["provenance"]["versions"]["cv2"] == "5.0.0"


def test_lane_row_pytorch_infer_stage_key():
    row = run_latency.lane_row(
        {"lane": "PyTorch + Pytorch-Wildlife", "kind": "legacy engine"},
        _artifact("pytorch-pytorchwildlife", 157.15, 168.41))
    assert row["infer_stage"] == "preprocess_plus_forward_plus_nms"
    assert row["infer_ms"]["p50"] == 157.15


def test_load_stored_reads_all_three_lanes(tmp_path):
    engines = {"cv2_detections_s2_new.json": "cv2-dnn-new",
               "cv2_detections_s2_classic.json": "cv2-dnn-classic",
               "pytorch_detections_s2.json": "pytorch-pytorchwildlife"}
    for fname, engine in engines.items():
        (tmp_path / fname).write_text(
            json.dumps(_artifact(engine, 100.0, 120.0)))
    rows = run_latency.load_stored(tmp_path)
    assert [r["engine_id"] for r in rows] == list(engines.values())
    assert all(r["n_images"] == 800 for r in rows)
    # the deploy engine (new) comes first
    assert rows[0]["kind"] == "new engine"


# ---------------------------------------------------------------------------
# Renderer + honesty
#


def _report():
    lanes = [
        run_latency.lane_row(spec, art) for spec, art in zip(
            run_latency.STORED_LANES,
            [_artifact("cv2-dnn-new", 125.13, 148.47),
             _artifact("cv2-dnn-classic", 141.64, 162.16),
             _artifact("pytorch-pytorchwildlife", 157.15, 168.41)])]
    return {
        "schema": "wfm-s5-latency/v1",
        "protocol": "stored S2 artifacts",
        "lanes": lanes,
        "fresh_confirmation": None,
        "cost_model": {
            "instances": run_latency.INSTANCES,
            "prices_retrieved": run_latency.PRICE_RETRIEVED,
            "prices_source": "public EC2 price trackers — ASSUMPTION",
            "hours_per_1m": "ms_per_image / 3.6",
            "includes": "decode + letterbox + forward + NMS",
            "graviton_methodology": run_latency.GRAVITON_METHOD,
        },
        "environment": {"machine": "x86_64", "cpu_count": 6},
    }


def test_render_md_stored_table():
    md = run_latency.render_md(_report())
    assert "| lane | role | n |" in md
    assert "cv2.dnn new (deploy)" in md
    assert "PyTorch + Pytorch-Wildlife" in md
    # headline gain: 168.41/148.47 - 1 ≈ 13.4% → "+13%"
    assert "+13%" in md
    # cost column present for the x86 instance
    assert "$/1M c7i.large" in md
    # no raw python dict leaks
    assert "{'" not in md and "{\"" not in md


def test_render_md_graviton_never_measured():
    md = run_latency.render_md(_report())
    assert "NOT MEASURED" in md
    assert "At-parity arithmetic" in md
    assert "--fresh bench re-run on the c7g.large" in md
    # the c7g instance row states the gap, not a measurement
    assert "no AWS access this sprint" in md
    # never a doubled negation from flag + text
    assert "NOT MEASURED — NO" not in md


def test_render_md_fresh_confirmation_section():
    report = _report()
    fresh_new = run_latency.lane_row(
        {"lane": "cv2.dnn new (fresh n=100)", "kind": "confirmation"},
        _artifact("cv2-dnn-new", 127.0, 151.0))
    fresh_new["n_images"] = 97
    report["fresh_confirmation"] = {
        "n_bench_images": 100, "warmup_images_excluded": 3,
        "new": fresh_new,
        "classic": run_latency.lane_row(
            {"lane": "cv2.dnn classic (fresh n=100)", "kind": "confirmation"},
            _artifact("cv2-dnn-classic", 143.0, 165.0)),
        "pytorch": run_latency.lane_row(
            {"lane": "PyTorch + Pytorch-Wildlife (fresh n=100)",
             "kind": "confirmation"},
            _artifact("pytorch-pytorchwildlife", 160.0, 172.0)),
    }
    md = run_latency.render_md(report)
    assert "Fresh confirmation (first 100" in md
    assert "vs stored p50" in md
    assert "2.53 ms" in md  # 151.0 - 148.47
    assert "fresh n=100" in md
