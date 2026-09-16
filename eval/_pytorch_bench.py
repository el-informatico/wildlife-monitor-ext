"""PyTorch-baseline latency bench, executed BY THE BASELINE VENV.

This file is a script, not a library: the legacy engine lives in the
baseline venv (Pytorch-Wildlife + ultralytics + torch), which is a
different interpreter from the pipeline venv that runs everything else.
``eval.run_latency`` invokes it as

    <baseline-venv>/bin/python -m eval._pytorch_bench --n 100

and it writes ``processed/s5_latency_fresh_pytorch.json``. The timing
protocol mirrors the S2 baseline runner exactly (decode timed apart
from `single_image_detection`, first WARMUP images excluded, conf floor
0.10) so the numbers are comparable with the stored full-800 artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF_FLOOR = 0.10
WARMUP_IMAGES = 3
MODEL_NAME = "MDV6-yolov10-c"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100,
                    help="sample images to time (warmup 3 excluded)")
    ap.add_argument("--out", default="s5_latency_fresh_pytorch")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    from PytorchWildlife.models import detection as pw_det

    sample = json.loads(
        (REPO_ROOT / "processed" / "eval_sample_s2.json").read_text())
    images = sample["images"][:args.n + WARMUP_IMAGES]

    detector = pw_det.MegaDetectorV6(
        version=MODEL_NAME, device="cpu",
        weights=str(REPO_ROOT / "models" / f"{MODEL_NAME}.pt"))

    ms_decode: list[float] = []
    ms_infer: list[float] = []
    for i, img in enumerate(images):
        path = REPO_ROOT / "data" / img["rel_path"]
        t0 = time.perf_counter()
        rgb = np.array(Image.open(path).convert("RGB"))
        t1 = time.perf_counter()
        detector.single_image_detection(rgb, img_path=str(path),
                                        det_conf_thres=CONF_FLOOR)
        t2 = time.perf_counter()
        if i >= WARMUP_IMAGES:
            ms_decode.append((t1 - t0) * 1000)
            ms_infer.append((t2 - t1) * 1000)

    def _pct(vals: list[float]) -> dict[str, float]:
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": round(float(a.mean()), 2),
                "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2)}

    e2e = [a + b for a, b in zip(ms_decode, ms_infer)]
    result = {
        "engine": "pytorch-pytorchwildlife",
        "n_images": len(ms_decode),
        "warmup_images_excluded": WARMUP_IMAGES,
        "versions": {"torch": torch.__version__,
                     "ultralytics": ultralytics.__version__,
                     "device": "cpu",
                     "torch_threads": torch.get_num_threads()},
        "latency_ms": {
            "decode": _pct(ms_decode),
            "preprocess_plus_forward_plus_nms": _pct(ms_infer),
            "end_to_end": _pct(e2e),
        },
    }
    out = REPO_ROOT / "processed" / f"{args.out}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    lat = result["latency_ms"]
    print(f"pytorch fresh: n={result['n_images']} "
          f"infer p50={lat['preprocess_plus_forward_plus_nms']['p50']}ms "
          f"e2e p50={lat['end_to_end']['p50']}ms -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
