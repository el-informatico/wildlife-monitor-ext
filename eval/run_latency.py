"""Sprint 5 latency + cost-per-1M-images table: new engine vs legacy.

Primary source: the three STORED same-protocol 800-image artifacts the
S2 comparison already produced (same sample, same warmup-3-excluded
per-image timing, same machine) — re-running inference would measure
the same thing noisier. A fresh confirmation re-run (first N sample
images) is available and was executed for this report at N=100 for
every lane, including the PyTorch lane via the baseline venv.

Lanes:

- ``cv2-dnn-new``      — the DEPLOY engine (OpenCV 5 DNN, graph engine
  "new"; pipeline default). Forward p50 125 ms / e2e p50 148 ms.
- ``cv2-dnn-classic``  — same ONNX through cv2's default graph engine;
  the S2 ablation, kept to show the engine choice itself.
- ``pytorch-pytorchwildlife`` — the LEGACY engine (Pytorch-Wildlife +
  ultralytics, torch CPU), the baseline the cv2 port must beat.

Cost model (explicitly an ASSUMPTION block, not a measurement): EC2
us-east-1 on-demand Linux list prices from public trackers, retrieved
2026-09-16 — c7i.large (x86, 2 vCPU) $0.089/hr, c7g.large (Graviton3,
2 vCPU) $0.0725/hr. Sequential per-image processing: hours per 1M
images = ms/image ÷ 3.6; cost = hours × price. No batching, queueing,
I/O wait, or retries are modeled. Graviton latency is NOT MEASURED —
this environment has no AWS access — so the Graviton cell carries the
methodology and an at-parity arithmetic illustration only.

Outputs: ``processed/s5_latency.json`` + ``processed/s5_latency.md``
(the fragment eval/S5-REPORT.md embeds verbatim).

    python -m eval.run_latency                  # stored artifacts only
    python -m eval.run_latency --fresh 100      # + fresh confirmation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "processed"
DATA = REPO_ROOT / "data"
MODELS = REPO_ROOT / "models"
DEFAULT_BASELINE_PY = REPO_ROOT / "baseline-venv" / "bin" / "python"
WARMUP_IMAGES = 3

#: the three stored same-protocol artifacts (S2) — role as the report
#: frames each lane.
STORED_LANES = [
    {"artifact": "cv2_detections_s2_new.json",
     "lane": "cv2.dnn new (deploy)", "kind": "new engine"},
    {"artifact": "cv2_detections_s2_classic.json",
     "lane": "cv2.dnn classic", "kind": "cv2 graph-engine ablation"},
    {"artifact": "pytorch_detections_s2.json",
     "lane": "PyTorch + Pytorch-Wildlife", "kind": "legacy engine"},
]

#: instance prices — ASSUMPTIONS, retrieved 2026-09-16 from public
#: EC2 price trackers (us-east-1, on-demand Linux). Not a quote; verify
#: at aws.amazon.com/ec2/pricing before budgeting.
PRICE_RETRIEVED = "2026-09-16"
INSTANCES = {
    "c7i.large": {"arch": "x86_64 (Intel)", "vcpu": 2,
                  "price_hr_us_east_1": 0.089,
                  "measured_flag": "yes",
                  "measured": "this workstation (AMD Ryzen 9 7940HS, "
                              "6 cores, WSL2), pipeline venv"},
    "c7g.large": {"arch": "Graviton3 (arm64)", "vcpu": 2,
                  "price_hr_us_east_1": 0.0725,
                  "measured_flag": "NOT MEASURED",
                  "measured": "no AWS access this sprint"},
}

GRAVITON_METHOD = (
    "Same --fresh bench re-run on the c7g.large itself (arm64 wheels of "
    "opencv-python and torch-cpu; the ONNX weights are "
    "architecture-independent), same first-100 sample images, warmup 3 "
    "excluded, then cost = ms ÷ 3.6 × $0.0725. The at-parity figure "
    "shown is that arithmetic with the measured x86 ms — it isolates "
    "the hourly-price difference ONLY; it is not a Graviton benchmark.")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def hours_per_1m(ms_per_image: float) -> float:
    """Sequential CPU-hours to process 1M images at ms/image.

    1e6 × ms / 1000 = seconds; / 3600 = hours → ms / 3.6.
    """
    return ms_per_image / 3.6


def cost_per_1m(ms_per_image: float, price_per_hour: float) -> float:
    return round(hours_per_1m(ms_per_image) * price_per_hour, 2)


def _percentiles(vals: list[float]) -> dict[str, float]:
    import numpy as np

    a = np.asarray(vals, dtype=np.float64)
    return {"mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2)}


def lane_row(lane_meta: dict[str, str], artifact: dict[str, Any]) -> dict[str, Any]:
    """Normalize one stored/fresh artifact into the report's row shape.

    Pure: derived numbers only, nothing re-timed.
    """
    lat = artifact["latency_ms"]
    infer_key = ("forward" if "forward" in lat
                 else "preprocess_plus_forward_plus_nms")
    infer = lat[infer_key]
    e2e = lat["end_to_end"]
    return {
        "lane": lane_meta["lane"],
        "kind": lane_meta["kind"],
        "n_images": artifact["n_images"],
        "engine_id": artifact["engine"],
        "infer_stage": infer_key,
        "infer_ms": {"p50": infer["p50"], "p95": infer["p95"]},
        "e2e_ms": {"p50": e2e["p50"], "p95": e2e["p95"]},
        "images_per_s_at_p50": round(1000 / e2e["p50"], 2),
        "hours_per_1m_at_e2e_p50": round(hours_per_1m(e2e["p50"]), 2),
        "cost_usd_per_1m": {
            name: cost_per_1m(e2e["p50"], inst["price_hr_us_east_1"])
            for name, inst in INSTANCES.items()},
        "provenance": {
            "model": artifact.get("model"),
            "params": artifact.get("params"),
            "versions": artifact.get("versions"),
        },
    }


# ---------------------------------------------------------------------------
# Stored artifacts
# ---------------------------------------------------------------------------


def load_stored(processed: Path = PROCESSED) -> list[dict[str, Any]]:
    rows = []
    for spec in STORED_LANES:
        path = processed / spec["artifact"]
        art = json.loads(path.read_text())
        art = {k: v for k, v in art.items() if k != "detections"}
        rows.append(lane_row(spec, art))
    return rows


# ---------------------------------------------------------------------------
# Fresh confirmation benches (first-N sample images, same protocol)
# ---------------------------------------------------------------------------


def bench_cv2(engine: str, n: int) -> dict[str, Any]:
    """Mirror of the S2 run_cv2 timing loop at size N, pipeline venv."""
    import cv2
    import numpy as np

    from pipeline import e2e

    mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
    sample = json.loads((PROCESSED / "eval_sample_s2.json").read_text())
    images = sample["images"][:n + WARMUP_IMAGES]

    detector = mdv6.MDv6OpenCV(
        MODELS / "MDV6-yolov10-c.onnx", engine=engine)
    ms: dict[str, list[float]] = {"decode": [], "pre": [], "forward": [],
                                  "post": []}
    for i, img in enumerate(images):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(DATA / img["rel_path"]), cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        if bgr is None:
            raise RuntimeError(f"decode failed: {img['rel_path']}")
        h, w = bgr.shape[:2]
        params = mdv6.letterbox_params(w, h, detector.input_size)
        padded = mdv6.letterbox_image(bgr, params, detector.input_size)
        blob = mdv6.to_blob(padded)
        t2 = time.perf_counter()
        detector.net.setInput(blob)
        output = np.asarray(detector.net.forward())
        t3 = time.perf_counter()
        mdv6.decode_prediction(output, params, 0.10,
                               mdv6.IOU_THRESHOLD, mdv6.MAX_DETECTIONS)
        t4 = time.perf_counter()
        if i >= WARMUP_IMAGES:
            ms["decode"].append((t1 - t0) * 1000)
            ms["pre"].append((t2 - t1) * 1000)
            ms["forward"].append((t3 - t2) * 1000)
            ms["post"].append((t4 - t3) * 1000)

    e2e_ms = [sum(x) for x in zip(ms["decode"], ms["pre"],
                                  ms["forward"], ms["post"])]
    return {
        "engine": f"cv2-dnn-{engine}",
        "n_images": len(ms["decode"]),
        "latency_ms": {k: _percentiles(v) for k, v in ms.items()}
                      | {"end_to_end": _percentiles(e2e_ms)},
        "params": {"n_bench_images": n,
                   "warmup_images_excluded": WARMUP_IMAGES,
                   "input_size": detector.input_size,
                   "conf_floor": 0.10},
        "versions": {"cv2": cv2.__version__,
                     "cv2_threads": cv2.getNumThreads()},
    }


def bench_pytorch(n: int, baseline_py: Path = DEFAULT_BASELINE_PY) -> dict[str, Any]:
    """Delegate to eval._pytorch_bench under the baseline venv."""
    if not baseline_py.exists():
        raise FileNotFoundError(
            f"baseline venv python not found at {baseline_py} — pass "
            "--baseline-python or skip the pytorch fresh lane")
    proc = subprocess.run(
        [str(baseline_py), "-m", "eval._pytorch_bench", "--n", str(n)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pytorch bench failed:\n{proc.stderr[-2000:]}")
    out = PROCESSED / "s5_latency_fresh_pytorch.json"
    return json.loads(out.read_text())


def run_fresh(n: int, baseline_py: Path = DEFAULT_BASELINE_PY
              ) -> dict[str, Any]:
    """All three lanes fresh; cv2 lanes persisted next to the pytorch one."""
    fresh: dict[str, Any] = {"n_bench_images": n,
                             "warmup_images_excluded": WARMUP_IMAGES}
    for engine in ("new", "classic"):
        art = bench_cv2(engine, n)
        path = PROCESSED / f"s5_latency_fresh_cv2_{engine}.json"
        path.write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
        spec = {"lane": f"cv2.dnn {engine} (fresh n={n})",
                "kind": "new engine fresh confirmation"}
        fresh[engine] = lane_row(spec, art)
    art = bench_pytorch(n, baseline_py)
    fresh["pytorch"] = lane_row(
        {"lane": f"PyTorch + Pytorch-Wildlife (fresh n={n})",
         "kind": "legacy engine fresh confirmation"}, art)
    return fresh


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _env_block() -> dict[str, Any]:
    import os
    import platform

    env: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import cv2

        env["cv2_threads"] = cv2.getNumThreads()
    except ImportError:  # pragma: no cover — pipeline venv always has cv2
        pass
    return env


def build_report(fresh: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "wfm-s5-latency/v1",
        "protocol": ("stored S2 artifacts: same 800-image sample, "
                     "per-image timing, first 3 images (warmup) "
                     "excluded; fresh confirmation = first-N sample "
                     "images, same exclusion, same order"),
        "lanes": load_stored(),
        "fresh_confirmation": fresh,
        "cost_model": {
            "instances": INSTANCES,
            "prices_retrieved": PRICE_RETRIEVED,
            "prices_source": ("public EC2 price trackers, us-east-1 "
                              "on-demand Linux list — ASSUMPTION, not a "
                              "quote; verify at "
                              "aws.amazon.com/ec2/pricing"),
            "hours_per_1m": "ms_per_image / 3.6 (sequential, no "
                            "batching/queueing/I/O-wait/retries)",
            "includes": "image decode + letterbox + blob + forward + "
                        "decode/NMS (end_to_end percentile of the "
                        "stored artifacts)",
            "graviton_methodology": GRAVITON_METHOD,
        },
        "environment": _env_block(),
    }


def render_md(report: dict[str, Any]) -> str:
    """The markdown fragment S5-REPORT.md embeds verbatim."""
    md: list[str] = []
    md.append("## Latency — new engine vs legacy (same 800-image sample, "
              "warmup 3 excluded)\n")
    md.append("| lane | role | n | forward/infer p50 | e2e p50 | e2e p95 "
              "| img/s @p50 | h/1M | $/1M c7i.large |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in report["lanes"]:
        md.append(
            f"| {r['lane']} | {r['kind']} | {r['n_images']} "
            f"| {r['infer_ms']['p50']} | {r['e2e_ms']['p50']} "
            f"| {r['e2e_ms']['p95']} | {r['images_per_s_at_p50']} "
            f"| {r['hours_per_1m_at_e2e_p50']} "
            f"| {r['cost_usd_per_1m']['c7i.large']} |")
    md.append("")

    best = min(report["lanes"], key=lambda r: r["e2e_ms"]["p50"])
    worst = max(report["lanes"], key=lambda r: r["e2e_ms"]["p50"])
    gain = round(worst["e2e_ms"]["p50"] / best["e2e_ms"]["p50"] - 1, 4)
    md.append(f"Fastest lane: **{best['lane']}** (e2e p50 "
              f"{best['e2e_ms']['p50']} ms); the legacy PyTorch engine is "
              f"**+{gain:.0%}** at e2e p50 ({worst['e2e_ms']['p50']} ms) "
              "on identical inputs and weights.\n")

    fresh = report.get("fresh_confirmation")
    if fresh:
        md.append(f"Fresh confirmation (first {fresh['n_bench_images']} "
                  "sample images, re-run today, same protocol):\n")
        md.append("| lane | e2e p50 | e2e p95 | vs stored p50 |")
        md.append("|---|---|---|---|")
        stored_by_engine = {r["engine_id"]: r for r in report["lanes"]}
        for key in ("new", "classic", "pytorch"):
            fr = fresh[key]
            st = stored_by_engine.get(fr["engine_id"])
            delta = ("—" if st is None else
                     f"{round(fr['e2e_ms']['p50'] - st['e2e_ms']['p50'], 2)} ms")
            md.append(f"| {fr['lane']} | {fr['e2e_ms']['p50']} "
                      f"| {fr['e2e_ms']['p95']} | {delta} |")
        md.append("")

    cm = report["cost_model"]
    md.append("### Cost per 1M images — ASSUMPTIONS, not a quote\n")
    md.append("| instance | arch | vCPU | $/hr us-east-1 | measured here? |")
    md.append("|---|---|---|---|---|")
    for name, inst in cm["instances"].items():
        md.append(f"| {name} | {inst['arch']} | {inst['vcpu']} "
                  f"| {inst['price_hr_us_east_1']} "
                  f"| {inst['measured_flag']} — {inst['measured']} |")
    md.append("")
    md.append(f"- Prices retrieved {cm['prices_retrieved']}; "
              f"{cm['prices_source']}.")
    md.append(f"- Model: {cm['hours_per_1m']}.")
    md.append(f"- Includes: {cm['includes']}.")
    g = cm["instances"]["c7g.large"]
    parity = report["lanes"][0]["cost_usd_per_1m"]["c7g.large"]
    md.append(f"- **Graviton (c7g.large): {g['measured_flag']}** — "
              f"{g['measured']}.")
    md.append(f"- At-parity arithmetic (x86 ms × c7g price) would be "
              f"**${parity}**/1M for the deploy engine. "
              f"{GRAVITON_METHOD}")
    md.append("")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fresh", type=int, default=0, metavar="N",
                    help="also re-run the first N sample images through "
                         "all three lanes as a confirmation")
    ap.add_argument("--baseline-python", type=Path,
                    default=DEFAULT_BASELINE_PY,
                    help="baseline-venv python for the PyTorch fresh lane")
    ap.add_argument("--out-prefix", default="s5_latency")
    args = ap.parse_args(argv)

    fresh = run_fresh(args.fresh, args.baseline_python) if args.fresh else None
    report = build_report(fresh)

    out_json = PROCESSED / f"{args.out_prefix}_report.json"
    out_md = PROCESSED / f"{args.out_prefix}.md"
    out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    out_md.write_text(render_md(report))

    for r in report["lanes"]:
        print(f"{r['lane']}: e2e p50 {r['e2e_ms']['p50']} ms, "
              f"${r['cost_usd_per_1m']['c7i.large']}/1M "
              f"(c7i.large assumption)")
    print(f"-> {out_json.relative_to(REPO_ROOT)}")
    print(f"-> {out_md.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
