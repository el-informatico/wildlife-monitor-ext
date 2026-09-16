"""Generate the G3 failure catalog (Sprint 3, deliverable 4).

Each case is mined deterministically from the same artifacts the eval
stages used — no new inference except where the case IS inference (the
OOV case classifies one Missouri crop) — rendered as one annotated
image under ``eval/failures/images/``, and documented in
``eval/failures/CATALOG.md`` with the measured numbers, a one-paragraph
diagnosis, and a reproducible key (image_id / sequence_id / crop_id +
the pick rule, which lives in this file).

Case registry (pick rules are the ``_pick_*`` functions):

1. ``det-fn-day``       missed detection, day — MDv6 ≥ deploy conf found
                        nothing matching GT (green) on an african_wildlife
                        sample image; sub-threshold floor dets in yellow
                        show whether the box was found-but-dropped;
2. ``det-fn-night``     missed detection, IR night — same, on a Missouri
                        boxed frame (content-IR night flag);
3. ``clf-tiny-crop``    species confusion on the smallest misclassified
                        held-out test crop (upscaling artifact);
4. ``clf-frame-edge``   species confusion where the GT box touches the
                        frame edge (truncated animal);
5. ``oov-closed-set``   Missouri species forced through classifier v0's
                        closed 4-species AW vocabulary;
6. ``seq-selection-loss`` max-per-trigger selection picked a frame that
                        missed GT although a gated sibling frame matched;
7. ``gate-fp-empty``    MOG2 gate fired on a GT-empty Missouri frame —
                        the exact per-camera mask, replayed with the
                        stage-motion parameters and verified against the
                        cached fg_ratio.

Usage (sprints-1-2 venv, see pipeline/README.md):

    python -m eval.run_failures [--only det-fn-day,...]

The script skips cases whose inputs are not present yet (e.g. the seq
cases before the seq stage ran) and says so on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from pipeline import e2e
from . import metrics, runtime

OUT_DIR = e2e.ext_root() / "eval" / "failures"
IMG_REL = "images"
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 200, 255)
ORANGE = (0, 120, 255)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Hand-written diagnoses, filled after inspecting the rendered images
# and the measured numbers (the auto-emitted data lines carry the
# numbers; these paragraphs say WHY).
DIAGNOSES: dict[str, str] = {
    "det-fn-day": (
        "Fragmentation of a close-up subject. The GT box spans almost the "
        "whole frame (417×314 of 550×412 — a near-camera animal) and MDv6 "
        "answers with overlapping boxes that each cover part of the body; "
        "the best fragment reaches IoU 0.414 against the full-body GT, below "
        "the 0.5 protocol threshold. Under the parity matcher's no-fallback "
        "rule every fragment stays unmatched, so this one mechanism costs "
        "recall (GT missed) and precision (the fragments count as FPs) on "
        "the same frame. The animal is detected — just not as one box; a "
        "fragment-merge post-step (or IoU criterion relaxed for very large "
        "GT boxes) would recover both numbers."
    ),
    "det-fn-night": (
        "Blind spot, not a threshold problem: no stored detection at any "
        "confidence (floor 0.10) overlaps the GT box — best IoU 0.000 — so "
        "no operating-threshold choice recovers this frame. The animal (a "
        "paca, folder 1.06-Paca) is a small low-contrast IR target "
        "(257×209 px, 1.7% of the frame) hugging the bottom-left frame "
        "corner, while the frame's only deployed detection (conf 0.269) "
        "sits at the far right — a spatially disjoint false positive. "
        "Missouri-night remains the best adequately-sized scope in the "
        "detection table (recall 0.8458 at deploy conf vs 0.7945 overall — "
        "IR monochrome is not the problem on average; the nominally higher "
        "african_wildlife|night 0.9167 rests on 12 GT boxes); the "
        "systematic gap is small edge-hugging night targets. Fixes live in "
        "the detector input (tiling / edge padding), not in the threshold."
    ),
    "clf-tiny": (
        "Upsampling past the information content. The crop is 54×26 px "
        "(~1.4k source pixels) stretched to 224×224 — ×4.1 horizontally, "
        "×8.6 vertically, aspect-distorting because the training recipe "
        "resizes to a square. At that magnification species texture is "
        "invented by interpolation, and the confusion lands exactly where "
        "grey-on-grey would predict: rhino → elephant. This is the "
        "smallest crop in the misclassified set, and rhino — the weakest "
        "class in the per-class table (0.9733 vs 0.9848–0.9926) — loses "
        "only this crop and one other small one (104×100 px, also "
        "upsampled); a min-resolution guard on deployed crops — or "
        "an aspect-preserving resize — is the direct fix."
    ),
    "clf-edge": (
        "Truncation at the frame edge. The GT box starts at (0, 0) — the "
        "animal is cut by the frame — and the margin expansion clamps "
        "against the border, so the crop shows a partial animal. The "
        "prediction (elephant → zebra, p=0.598) is a hedge — barely above "
        "half — but the closed softmax must still name an AW species. Note "
        "the contrast with the tiny-crop "
        "case: the two misclassified extremes are a 54×26 px crop and a "
        "605×365 px one, while everything in between sits at 0.99 — "
        "v0's errors are concentrated at the edges of its input "
        "distribution, not spread across it."
    ),
    "oov-closed-set": (
        "Out-of-vocabulary input with no reject option. Classifier v0's "
        "vocabulary is the 4 african_wildlife species; a Missouri roe deer "
        "(folder 1.58-Roe_Deer) forced through it comes out 'rhino' at "
        "p=0.5704 — a mid-grade, entirely confident-looking answer, because "
        "the softmax must distribute its mass over AW classes no matter "
        "what is in the frame. This is why G3 reports no classifier "
        "metrics for Missouri: without an OOV/none class (or an embedding "
        "distance threshold) every such number would be meaningless by "
        "construction, and a confidence gate alone would not catch it — "
        "0.57 is not a low confidence."
    ),
    "seq-selection": (
        "Selection optimizes detector confidence, and confidence does not "
        "track correctness. All three frames pass the gate (close-up "
        "subject, fg_ratio 1.0), so the motion stage is not at fault; the "
        "selector then takes the frame with the highest-confidence animal "
        "detection — 0.81 on the first frame — and that detection is one "
        "of three overlapping fragments, none of which reaches IoU 0.5 "
        "with GT (the same close-up fragmentation mode as case 1). The "
        "sibling frame one step later has a single 0.69 detection that "
        "matches GT cleanly and would have been the right representative. "
        "Selection loss is a tail mode, not a systematic flaw (12 of 215 "
        "GT sequences, 5.6%; african_wildlife 3/114) — but its mechanism "
        "is inherited from the detector: a fragment-merge step before "
        "selection would fix both this case and case 1."
    ),
    "gate-fp-empty": (
        "Stale background model at burst start. fg_ratio is exactly 1.0000 "
        "— every pixel flagged foreground — while the neighbouring frames "
        "sit at 0.002–0.017, and 21 of the 22 full-foreground frames on "
        "this camera are the FIRST frame after a capture gap: MOG2 only "
        "updates when a frame arrives, so during an idle gap the model "
        "freezes while the scene drifts (vegetation, sun angle, a bumped "
        "camera), and the first real frame differs globally from the "
        "stale background. The auto learning rate re-adapts within one "
        "frame, which is why the spike is isolated. The cost is bounded — "
        "one wasted detector call per burst — but on animal-free cameras "
        "every burst pays it (this camera gates 506/912 frames). A "
        "two-consecutive-frames debounce, or a global-change guard "
        "(fg ≳ 0.9 is scene change, not an animal — Missouri animals "
        "occupy a few percent of frame), removes the class of trigger."
    ),
}


# ---------------------------------------------------------------------------
# shared context: report + S2 sample + stored detections + manifest
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self) -> None:
        ext = e2e.ext_root()
        self.ext = ext
        self.data_dir = ext / "data"
        self.proc = ext / "processed"
        self.report = json.loads((self.proc / "g3_report.json").read_text())
        sample = json.loads((self.proc / "eval_sample_s2.json").read_text())
        self.sample_order = [i["image_id"] for i in sample["images"]]
        self.sample = {i["image_id"]: i for i in sample["images"]}
        self.floor = json.loads(
            (self.proc / "cv2_detections_s2_new.json").read_text()
        )["detections"]
        self.manifest = runtime.load_manifest()
        self.row_by_id = {r["image_id"]: r for r in self.manifest}
        self.row_by_rel = {r["rel_path"]: r for r in self.manifest}
        self.ir = runtime.IRFlagCache()
        self._motion: list[dict] | None = None
        self._clf = None
        self._classes: dict[int, str] | None = None
        self._mo_gt = None
        self._aw_load = None

    # -- lazy bits ------------------------------------------------------
    def motion(self) -> list[dict] | None:
        if self._motion is None:
            path = self.proc / "g3_motion.parquet"
            if not path.exists():
                return None
            import pyarrow.parquet as pq

            self._motion = pq.read_table(path).to_pylist()
        return self._motion

    def classes(self) -> dict[int, str]:
        if self._classes is None:
            index = json.loads(
                (self.proc / "classifier_crops_v0.json").read_text()
            )
            self._classes = {int(k): v for k, v in index["classes"].items()}
        return self._classes

    def classifier(self):
        if self._clf is None:
            clf = e2e.import_wildlife("wildlife_monitor.classify.model")
            self._clf = clf.ClassifierOpenCV(
                self.ext / "models" / "classifier_v0_resnet18.onnx",
                engine="new",
            )
        return self._clf

    def mo_gt(self):
        if self._mo_gt is None:
            self._mo_gt = runtime.missouri_gt()
        return self._mo_gt

    def aw_load(self):
        if self._aw_load is None:
            self._aw_load = runtime.gt_loader(
                self.data_dir, self.classes(), self.mo_gt()
            )
        return self._aw_load

    def ir_flag(self, image_id: str, rel_path: str) -> bool:
        flag = self.ir.get(image_id, self.data_dir / rel_path)
        assert flag is not None, f"no IR flag for {image_id}"
        return flag

    def seq_dets(self, dataset: str) -> dict[str, list[dict]] | None:
        name = MO_SEQ_DETS if dataset == "missouri" else AW_TEST_DETS
        path = self.proc / name
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def floor_dets(self, image_id: str) -> list[dict]:
        return self.floor.get(image_id, [])


MO_SEQ_DETS = "g3_seq_dets_missouri.json"
AW_TEST_DETS = "g3_seq_dets_aw_test.json"


# ---------------------------------------------------------------------------
# mining helpers (same sources + matching protocol as stage_det)
# ---------------------------------------------------------------------------


def _match(ctx: Ctx, img: dict, conf: float) -> tuple[list[dict], list[bool]]:
    """Deployed dets at `conf` + their match flags vs the sample GT."""
    deployed = [d for d in ctx.floor_dets(img["image_id"]) if d["conf"] >= conf]
    mapping = metrics.greedy_match(
        [b["bbox"] for b in img["gt_boxes"]],
        [d["bbox"] for d in deployed],
        [d["conf"] for d in deployed],
    )
    return deployed, [m >= 0 for m in mapping]


def _best_floor_iou(ctx: Ctx, img: dict, gt_idx: int) -> tuple[float, float] | None:
    """(IoU, conf) of the best floor detection (≥ 0.10) vs one GT box."""
    gt = img["gt_boxes"][gt_idx]["bbox"]
    gt_xyxy = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
    best: tuple[float, float] | None = None
    for d in ctx.floor_dets(img["image_id"]):
        iou = metrics.iou_xyxy(gt_xyxy, d["bbox"])
        if best is None or iou > best[0]:
            best = (iou, d["conf"])
    return best


def _missed_gt(ctx: Ctx, img: dict, deployed: list[dict],
               matched: list[bool]) -> list[int]:
    """GT indices no deployed detection claimed (parity rule)."""
    claimed = set(
        metrics.greedy_match(
            [b["bbox"] for b in img["gt_boxes"]],
            [d["bbox"] for d in deployed],
            [d["conf"] for d in deployed],
        )
    )
    return [gi for gi in range(len(img["gt_boxes"])) if gi not in claimed]


def _pick_fn(ctx: Ctx, dataset: str, day_night: str,
             require_deployed: bool) -> dict | None:
    """First sample image (sample order) of a stratum with an unmatched GT."""
    for image_id in ctx.sample_order:
        img = ctx.sample[image_id]
        if img["source_dataset"] != dataset or not img["gt_boxes"]:
            continue
        dn = "night" if ctx.ir_flag(image_id, img["rel_path"]) else "day"
        if dn != day_night:
            continue
        deployed, matched = _match(ctx, img, e2e.DEPLOY_CONF)
        if require_deployed and not deployed:
            continue
        missed = _missed_gt(ctx, img, deployed, matched)
        if missed:
            return {"img": img, "deployed": deployed, "matched": matched,
                    "missed": missed, "day_night": dn}
    return None


def _species_labels(ctx: Ctx, img: dict) -> list[str]:
    if img["source_dataset"] != "african_wildlife":
        return ["GT animal"] * len(img["gt_boxes"])
    row = ctx.row_by_id[img["image_id"]]
    boxes = e2e.aw_species_boxes(
        e2e.aw_label_path(ctx.data_dir, img["rel_path"]),
        row["width"], row["height"], ctx.classes(),
    )
    if len(boxes) != len(img["gt_boxes"]):
        return ["GT"] * len(img["gt_boxes"])
    return [f"GT {b['species']}" for b in boxes]


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _load(ctx: Ctx, rel_path: str, cap: int = 1400) -> tuple[np.ndarray, float]:
    bgr = cv2.imread(str(ctx.data_dir / rel_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"decode failed: {rel_path}")
    long_side = max(bgr.shape[:2])
    scale = min(1.0, cap / long_side)
    if scale < 1.0:
        bgr = cv2.resize(
            bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return bgr, scale


def _text_bar(img: np.ndarray, lines: list[str], height: int = 0) -> np.ndarray:
    """Prepend a black caption bar with 1-2 text lines above the image."""
    line_h = 26
    height = height or line_h * len(lines) + 8
    bar = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 20 + i * line_h), FONT, 0.55,
                    WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _box(img: np.ndarray, xyxy_scaled, color, thickness, label: str) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy_scaled)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        y_text = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 6
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2),
                      BLACK, -1)
        cv2.putText(img, label, (x1 + 3, y_text - 3), FONT, 0.5, color, 1,
                    cv2.LINE_AA)


def _draw_det_case(ctx: Ctx, picked: dict, missed: list[int],
                   title_lines: list[str]) -> np.ndarray:
    img_rec = picked["img"]
    bgr, s = _load(ctx, img_rec["rel_path"])
    labels = _species_labels(ctx, img_rec)
    th = max(2, round(3 * s))
    for gi, gt in enumerate(img_rec["gt_boxes"]):
        b = gt["bbox"]
        color = ORANGE if gi in missed else GREEN
        _box(bgr, [(b[0]) * s, (b[1]) * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s],
             color, th + 1, ("MISSED " if gi in missed else "") + labels[gi])
    for d, ok in zip(picked["deployed"], picked["matched"]):
        _box(bgr, [v * s for v in d["bbox"]], RED, th,
             f"det {d['conf']:.2f}{'' if ok else ' FP'}")
    for d in ctx.floor_dets(img_rec["image_id"]):
        if e2e.DEPLOY_CONF > d["conf"] >= e2e.CONF_FLOOR:
            _box(bgr, [v * s for v in d["bbox"]], YELLOW, max(1, th - 1),
                 f"floor {d['conf']:.2f}")
    return _text_bar(bgr, title_lines)


def _save(name: str, bgr: np.ndarray) -> str:
    (OUT_DIR / IMG_REL).mkdir(parents=True, exist_ok=True)
    rel = f"{IMG_REL}/{name}.jpg"
    cv2.imwrite(str(OUT_DIR / rel), bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return rel


def _hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Horizontal stack with white padding below shorter panels + 6px gaps."""
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
        if i:
            p = np.hstack([np.full((h, 6, 3), 255, np.uint8), p])
        out.append(p)
    return np.hstack(out)


def _montage(cells: list[tuple[np.ndarray, list[str]]], cols: int,
             cell_w: int = 420) -> np.ndarray:
    """Grid of captioned cells (each already annotated, uncaptioned)."""
    rendered = []
    for img, caption in cells:
        h = round(img.shape[0] * cell_w / img.shape[1])
        cell = cv2.resize(img, (cell_w, h), interpolation=cv2.INTER_AREA)
        cell = _text_bar(cell, caption)
        cell = cv2.copyMakeBorder(
            cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=WHITE)
        rendered.append(cell)
    # mixed portrait/landscape frames: pad every cell to one grid height
    max_h = max(c.shape[0] for c in rendered)
    rendered = [
        c if c.shape[0] == max_h else np.vstack(
            [c, np.full((max_h - c.shape[0], c.shape[1], 3), 255, np.uint8)])
        for c in rendered
    ]
    step = cell_w + 8
    rows = []
    for i in range(0, len(rendered), cols):
        row = np.hstack(rendered[i:i + cols])
        if len(rendered) - i < cols:  # pad the last row out to grid width
            pad = np.full((row.shape[0], (cols * step) - row.shape[1], 3),
                          255, np.uint8)
            row = np.hstack([row, pad])
        rows.append(row)
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_det_fn_day(ctx: Ctx) -> dict | None:
    picked = (_pick_fn(ctx, "african_wildlife", "day", require_deployed=True)
              or _pick_fn(ctx, "african_wildlife", "day", require_deployed=False))
    if not picked:
        return None
    img = picked["img"]
    missed = picked["missed"]
    best = _best_floor_iou(ctx, img, missed[0])
    n_floor_sub = sum(
        1 for d in ctx.floor_dets(img["image_id"])
        if e2e.DEPLOY_CONF > d["conf"] >= e2e.CONF_FLOOR
    )
    gt = img["gt_boxes"][missed[0]]["bbox"]
    frac = gt[2] * gt[3] / (img["width"] * img["height"])
    title = [
        f"{img['image_id']}  {img['source_dataset']} day  deploy>={e2e.DEPLOY_CONF}"
        f"  GT green / MISSED orange / det red / floor yellow",
        f"{len(img['gt_boxes'])} GT, {len(picked['deployed'])} deployed "
        f"({sum(picked['matched'])} matched), missed GT "
        f"{gt[2]:.0f}x{gt[3]:.0f}px = {100 * frac:.1f}% of frame"
        + (f", best floor det IoU {best[0]:.2f} @ conf {best[1]:.2f}" if best else ""),
    ]
    image = _save("det-fn-day", _draw_det_case(ctx, picked, missed, title))
    return {
        "id": "det-fn-day",
        "title": "Missed detection (FN) — day frame, african_wildlife",
        "image": image,
        "key": (f"image_id `{img['image_id']}` ({img['rel_path']}), S2 sample "
                f"stratum {img['stratum']}, split {img['split']}; pick rule: "
                "first AW-day sample image (sample order) with an unmatched GT"),
        "data": [
            f"{len(img['gt_boxes'])} GT box(es); {len(picked['deployed'])} "
            f"deployed detections (conf ≥ {e2e.DEPLOY_CONF}), "
            f"{sum(picked['matched'])} matched — {len(missed)} GT missed",
            f"missed GT box {gt[2]:.0f}×{gt[3]:.0f} px "
            f"({100 * frac:.1f}% of the frame)",
            (f"best sub-deploy detection: IoU {best[0]:.2f} at conf "
             f"{best[1]:.2f} (< {e2e.DEPLOY_CONF} deploy threshold)"
             if best else "no detection at any conf ≥ 0.10 overlaps the GT "
             "(best IoU below the matcher's view)"),
            f"{n_floor_sub} detection(s) in the 0.10–0.25 band on this frame "
            "(drawn yellow)",
        ],
    }


def case_det_fn_night(ctx: Ctx) -> dict | None:
    picked = _pick_fn(ctx, "missouri", "night", require_deployed=True)
    if not picked:
        return None
    img = picked["img"]
    missed = picked["missed"]
    best = _best_floor_iou(ctx, img, missed[0])
    gt = img["gt_boxes"][missed[0]]["bbox"]
    title = [
        f"{img['image_id']}  missouri NIGHT (content-IR)  deploy>={e2e.DEPLOY_CONF}"
        f"  GT green / MISSED orange / det red / floor yellow",
        f"{len(img['gt_boxes'])} GT, {len(picked['deployed'])} deployed "
        f"({sum(picked['matched'])} matched), missed GT "
        f"{gt[2]:.0f}x{gt[3]:.0f}px"
        + (f", best floor IoU {best[0]:.2f} @ {best[1]:.2f}" if best else ""),
    ]
    image = _save("det-fn-night", _draw_det_case(ctx, picked, missed, title))
    row = ctx.row_by_id[img["image_id"]]
    seq_dir = row["rel_path"].rsplit("/", 1)[0]
    return {
        "id": "det-fn-night",
        "title": "Missed detection (FN) — IR night frame, missouri",
        "image": image,
        "key": (f"image_id `{img['image_id']}` ({img['rel_path']}), sequence "
                f"dir {seq_dir}; pick rule: first missouri-night sample image "
                "(sample order) with an unmatched GT and ≥1 deployed det"),
        "data": [
            f"IR monochrome frame (content flag: mean HSV saturation < "
            f"{metrics.IR_SAT_THRESHOLD})",
            f"{len(img['gt_boxes'])} GT box(es); {len(picked['deployed'])} "
            f"deployed, {sum(picked['matched'])} matched — {len(missed)} missed",
            (f"best sub-deploy detection vs the missed GT: IoU {best[0]:.2f} "
             f"at conf {best[1]:.2f}" if best else
             "no floor detection (≥ 0.10) reached IoU 0.5 with the missed GT"),
        ],
    }


def _clf_panel(ctx: Ctx, mis: dict) -> tuple[np.ndarray, dict]:
    """Cached crop (nearest-upscaled) | source frame with the crop window."""
    crop = cv2.imread(str(ctx.proc / mis["crop_path"]), cv2.IMREAD_COLOR)
    h = 420
    crop_big = cv2.resize(
        crop, (round(crop.shape[1] * h / crop.shape[0]), h),
        interpolation=cv2.INTER_NEAREST,
    )
    crop_big = _text_bar(
        crop_big,
        [f"cached test crop {mis['box_xywh'][2]}x{mis['box_xywh'][3]}px",
         f"true {mis['true']}  ->  pred {mis['pred']}"],
    )

    frame, s = _load(ctx, mis["source_image"], cap=1000)
    row = ctx.row_by_rel[mis["source_image"]]
    boxes = e2e.aw_species_boxes(
        e2e.aw_label_path(ctx.data_dir, mis["source_image"]),
        row["width"], row["height"], ctx.classes(),
    )
    x, y, w, hh = mis["box_xywh"]
    th = max(2, round(3 * s))
    for b in boxes:
        bb = b["bbox"]
        _box(frame, [bb[0] * s, bb[1] * s, (bb[0] + bb[2]) * s, (bb[1] + bb[3]) * s],
             GREEN, th, f"GT {b['species']}")
    _box(frame, [x * s, y * s, (x + w) * s, (y + hh) * s], ORANGE, th + 1,
         f"crop ({mis['crop_id'].rsplit('__', 1)[1]})")
    info = {"n_boxes": len(boxes)}
    return _hstack_panels([crop_big, frame]), info


def case_clf_confusion(ctx: Ctx, mode: str) -> dict | None:
    mis = ctx.report.get("clf", {}).get("misclassified", [])
    if not mis:
        return None

    def _area(m):
        b = m["box_xywh"]
        return b[2] * b[3]

    if mode == "tiny":
        chosen = min(mis, key=lambda m: (_area(m), m["crop_id"]))
        rule = "smallest crop area, tie by crop_id"
    else:  # frame-edge: window touching the frame boundary
        edges = [m for m in mis
                 if m["box_xywh"][0] <= 1 or m["box_xywh"][1] <= 1]
        if not edges:
            return None
        chosen = min(edges, key=lambda m: (_area(m), m["crop_id"]))
        rule = "smallest area among crops at the frame boundary"

    probs = chosen["probs"]
    ranked = sorted(range(len(probs)), key=lambda i: -probs[i])
    names = ctx.classes()
    panel, info = _clf_panel(ctx, chosen)
    title = (f"{chosen['crop_id']}  test-split crop  "
             f"true {chosen['true']} -> pred {chosen['pred']} "
             f"(p={max(probs):.3f})")
    image = _save(f"clf-{mode}", _text_bar(panel, [title]))
    b = chosen["box_xywh"]
    return {
        "id": f"clf-{mode}",
        "title": ("Species confusion — smallest misclassified crop (upscaling)"
                  if mode == "tiny" else
                  "Species confusion — GT box at the frame edge (truncation)"),
        "image": image,
        "key": (f"crop_id `{chosen['crop_id']}` from source image "
                f"`{chosen['source_image']}` (AW test split, trainer's 375-crop "
                f"held-out cache); pick rule: {rule}"),
        "data": [
            f"crop {b[2]}×{b[3]} px → resized to 224×224 "
            f"(×{224 / max(b[2], b[3]):.1f} on the long edge) by the training recipe",
            f"true {chosen['true']} → predicted {chosen['pred']}; top probs: "
            + ", ".join(f"{names[i]} {probs[i]:.3f}" for i in ranked[:2]),
            f"{info['n_boxes']} species-labelled GT box(es) on the source frame",
        ],
    }


def case_oov(ctx: Ctx) -> dict | None:
    mo_gt = ctx.mo_gt()
    target = None
    for image_id in ctx.sample_order:
        img = ctx.sample[image_id]
        if img["source_dataset"] != "missouri" or img["stratum"] != "mo_with_gt":
            continue
        if ctx.ir_flag(image_id, img["rel_path"]):
            continue  # day frame for a legible image
        boxes = mo_gt.boxes_for(img["rel_path"])
        if any(b["bbox"][2] >= 100 and b["bbox"][3] >= 100 for b in boxes):
            target = (img, boxes)
            break
    if not target:
        return None
    img, boxes = target
    b = max(boxes, key=lambda x: x["bbox"][2] * x["bbox"][3])["bbox"]
    row = ctx.row_by_id[img["image_id"]]
    frame, s = _load(ctx, img["rel_path"], cap=900)
    th = max(2, round(3 * s))
    _box(frame, [b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s],
         GREEN, th, "GT box (unlabelled species)")
    win = e2e.crop_window_xywh(b, img["width"], img["height"])
    full = cv2.imread(str(ctx.data_dir / img["rel_path"]), cv2.IMREAD_COLOR)
    crop = e2e.extract_crop(full, win)
    pred, _ = ctx.classifier().classify(crop)
    crop_big = cv2.resize(crop, (420, 420), interpolation=cv2.INTER_NEAREST)
    crop_big = _text_bar(
        crop_big,
        ["GT crop (training recipe) -> classifier v0",
         f"pred {pred['class_name']}  p={pred['prob']:.3f}"],
    )
    frame = _text_bar(frame, [f"{img['image_id']} missouri day (content flag)"])
    image = _save("oov-closed-set", _hstack_panels([crop_big, frame]))
    species_dir = img["rel_path"].split("/")[2]
    return {
        "id": "oov-closed-set",
        "title": "Out-of-vocabulary — Missouri species through the closed "
                 "4-species classifier",
        "image": image,
        "key": (f"image_id `{img['image_id']}` ({img['rel_path']}), GT box "
                f"{b[2]:.0f}×{b[3]:.0f}px; pick rule: first missouri mo_with_gt "
                "day sample image (sample order) with a ≥100px GT box; "
                "classifier v0 ONNX, cv2.dnn engine new"),
        "data": [
            f"dataset directory species context: `{species_dir}` "
            "(Missouri folders carry species names; box GT is animal-only)",
            f"classifier v0 vocabulary: {sorted(ctx.classes().values())} — "
            "closed set, no reject option",
            f"forced call on the GT crop: {pred['class_name']} "
            f"(p={pred['prob']:.3f}) — a confident AW species for a "
            "non-AW animal",
        ],
    }


def case_seq_selection(ctx: Ctx) -> dict | None:
        seq = ctx.report.get("seq")
        if not seq:
            return None
        mining = seq["mining"]
        pool = mining["selection_loss"] or mining["gate_lost_gt"]
        if not pool:
            return None
        rec = sorted(pool, key=lambda r: r["sequence_id"])[0]
        sid = rec["sequence_id"]
        dataset = rec["dataset"]
        dets = ctx.seq_dets(dataset)
        if dets is None:
            return None

        frames = rec["frames"]
        cells: list[tuple[np.ndarray, list[str]]] = []
        keep = list(range(min(7, len(frames))))
        sel_idx = next((i for i, f in enumerate(frames) if f.get("selected")), None)
        if sel_idx is not None and sel_idx not in keep:
            keep = keep[:6] + [sel_idx]
        for i in keep:
            f = frames[i]
            row = ctx.row_by_id[f["image_id"]]
            bgr, s = _load(ctx, f["rel_path"], cap=560)
            th = max(2, round(3 * s))
            gt = (ctx.mo_gt().boxes_for(f["rel_path"]) if dataset == "missouri"
                  else ctx.aw_load()(row))
            for g in gt:
                b = g["bbox"]
                _box(bgr, [b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s],
                     GREEN, th, "GT")
            deployed = [d for d in dets.get(f["image_id"], [])
                        if d["conf"] >= e2e.DEPLOY_CONF
                        and e2e.DET_LABELS[d["cls"]] == "animal"]
            for d in deployed:
                _box(bgr, [v * s for v in d["bbox"]], RED, th,
                     f"{d['conf']:.2f}")
            fg = f.get("fg_ratio")
            conf = max((a["conf"] for a in f.get("animal_dets", [])), default=None)
            caption = [
                f"#{i}{' SELECTED' if f.get('selected') else ''}"
                f"  gate={'Y' if f['gate'] else 'N'}"
                + (f" fg={fg:.3f}" if fg is not None else ""),
                ("det conf " + (f"{conf:.2f}" if conf is not None else "-"))
                + (f"  matched={'Y' if f['matched'] else 'N'}"),
            ]
            cells.append((bgr, caption))
        montage = _montage(cells, cols=min(4, len(cells)))
        title = [
            f"sequence `{sid}` ({dataset}, {rec['day_night']}) — "
            f"{len(frames)} frames, showing {len(cells)}"
            + (f" (+{len(frames) - len(cells)} not shown)" if len(frames) > len(cells) else ""),
            "GT green / deployed animal det red; selection = highest-conf "
            "animal det among gated frames",
        ]
        image = _save("seq-selection", _text_bar(montage, title))
        gated_matched = sum(
            1 for f in frames if f["gate"] and f["matched"])
        best_gated = next((f for f in frames if f["gate"] and f["matched"]), None)
        return {
            "id": "seq-selection",
            "title": ("Sequence failure — selection picked a frame that missed GT"
                      if mining["selection_loss"] else
                      "Sequence failure — motion gate suppressed a GT sequence"),
            "image": image,
            "key": (f"sequence_id `{sid}`, {len(frames)} frames "
                    f"({dataset}, {rec['day_night']}); pick rule: first record "
                    "by sequence_id in report seq.mining."
                    + ("selection_loss" if mining["selection_loss"]
                       else "gate_lost_gt")),
            "data": [
                f"selected_reason: {rec['reason']}",
                f"{gated_matched} gated frame(s) matched GT; the selected frame "
                f"{'missed' if mining['selection_loss'] else 'had no'} GT match"
                + (f"; best gated+matched frame: `{best_gated['image_id']}`"
                   if best_gated else ""),
                f"gate pass on {sum(1 for f in frames if f['gate'])}/{len(frames)} "
                "frames (MOG2 fg_ratio > "
                f"{metrics.TRIGGER_FG_THRESHOLD}, per-camera warmup excluded)",
            ],
        }


def case_gate_fp(ctx: Ctx) -> dict | None:
    motion = ctx.motion()
    if not motion:
        return None
    mo_empty = {i["image_id"] for i in ctx.sample.values()
                if i["stratum"] == "mo_empty"}
    by_id = {m["image_id"]: m for m in motion}
    cam_sizes: dict[str, int] = defaultdict(int)
    for m in motion:
        cam_sizes[m["location_id"]] += 1
    candidates = [
        m for m in motion
        if m["image_id"] in mo_empty and m["gate"] and not m["warmup"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (-m["fg_ratio"], m["image_id"]))
    target = next((m for m in candidates
                   if cam_sizes[m["location_id"]] <= 800), candidates[0])
    loc = target["location_id"]

    # exact replay of stage_motion for this camera (same params, same order)
    # to recover the MOG2 mask at the target frame
    mog2 = e2e.import_wildlife("wildlife_monitor.motion.mog2")
    night = e2e.import_wildlife("wildlife_monitor.motion.night")
    sub = cv2.createBackgroundSubtractorMOG2(
        history=50, varThreshold=24.0, detectShadows=False)
    enhancer = night.NightEnhancer()
    rows = sorted(
        (r for r in ctx.manifest if r["location_id"] == loc),
        key=lambda r: r["camera_ts"],
    )
    mask = gray = None
    for r in rows:
        bgr = runtime.decode_capped(ctx.data_dir / r["rel_path"])
        ir = ctx.ir.of(r["image_id"], bgr)
        g = night.to_gray(bgr)
        if ir:
            g = enhancer.enhance(g)
        mask = sub.apply(g, learningRate=-1.0)
        gray = g
        if r["image_id"] == target["image_id"]:
            fg = round(float(np.count_nonzero(mask)) / mask.size, 6)
            assert fg == target["fg_ratio"], (
                f"replay diverged: {fg} != {target['fg_ratio']}")
            break
    assert mask is not None

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = vis.copy()
    red[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(vis, 0.55, red, 0.45, 0)
    overlay = _text_bar(overlay, [
        f"MOG2 foreground mask (exact replay, fg={target['fg_ratio']:.4f})",
        f"threshold {metrics.TRIGGER_FG_THRESHOLD} -> gate PASSES on this "
        "GT-empty frame",
    ])
    color, s = _load(ctx, ctx.row_by_id[target["image_id"]]["rel_path"], cap=768)
    color = _text_bar(color, [f"{target['image_id']} — missouri GT-empty "
                              "stratum (color frame for context)"])
    image = _save("gate-fp-empty", _hstack_panels([overlay, color]))

    cam_frames = [m for m in motion if m["location_id"] == loc]
    return {
        "id": "gate-fp-empty",
        "title": "Motion-gate false positive — MOG2 fires on a GT-empty frame",
        "image": image,
        "key": (f"image_id `{target['image_id']}` (camera `{loc}`, "
                f"{len(cam_frames)} frames), S2 mo_empty stratum; pick rule: "
                "highest fg_ratio among gated non-warmup empty frames "
                "(camera ≤ 800 frames preferred, tie by image_id); mask "
                "replayed with the stage-motion parameters and asserted "
                "equal to the cached fg_ratio"),
        "data": [
            f"fg_ratio {target['fg_ratio']:.4f} vs trigger threshold "
            f"{metrics.TRIGGER_FG_THRESHOLD} "
            f"({target['fg_ratio'] / metrics.TRIGGER_FG_THRESHOLD:.1f}×)",
            f"camera `{loc}`: {sum(1 for m in cam_frames if m['gate'])} of "
            f"{len(cam_frames)} frames gated — every gated frame means a "
            "detector call on an animal-free burst",
            f"IR flag on this frame: {target['ir']} (CLAHE applied by the "
            "motion stage when IR)",
        ],
    }


# ---------------------------------------------------------------------------
# catalog assembly
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="",
                        help="comma-separated case ids (default: all)")
    args = parser.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    ctx = Ctx()
    builders = [
        ("det-fn-day", case_det_fn_day),
        ("det-fn-night", case_det_fn_night),
        ("clf-tiny", lambda c: case_clf_confusion(c, "tiny")),
        ("clf-edge", lambda c: case_clf_confusion(c, "edge")),
        ("oov-closed-set", case_oov),
        ("seq-selection", case_seq_selection),
        ("gate-fp-empty", case_gate_fp),
    ]
    cases: list[dict] = []
    for case_id, build in builders:
        if only and case_id not in only:
            continue
        try:
            case = build(ctx)
        except FileNotFoundError as exc:
            print(f"skip {case_id} ({exc.filename} missing)", file=sys.stderr)
            case = None
        if not case:
            continue
        cases.append(case)
    if only:
        missing = only - {c["id"] for c in cases}
        if missing:
            print(f"warning: cases not produced: {sorted(missing)}",
                  file=sys.stderr)
    ctx.ir.flush()

    stages = ctx.report.get("meta", {}).get("stages_run", [])
    lines = [
        "# G3 failure catalog",
        "",
        "One annotated image per case under `images/`; numbers below are read",
        "from the same artifacts the eval stages produced (no re-measurement).",
        "Regenerate everything with:",
        "",
        "    ../wildlife-monitor/.venv/bin/python -m eval.run_failures",
        "",
        "Built from `processed/g3_report.json` (stages: "
        + (", ".join(stages) if stages else "none")
        + "), the S2 800-image sample + stored cv2.dnn detections, and "
        "`processed/g3_motion.parquet`. Deploy conf "
        f"{e2e.DEPLOY_CONF}, "
        f"decode floor {e2e.CONF_FLOOR}, match IoU {metrics.MATCH_IOU} "
        "(S2 parity protocol). Day/night = content IR flag "
        f"(mean HSV saturation < {metrics.IR_SAT_THRESHOLD}).",
        "",
    ]
    for n, case in enumerate(cases, 1):
        lines += [f"## {n}. {case['title']}", ""]
        lines += [f"![{case['id']}]( {case['image']})", ""]
        lines += [f"**Key** — {case['key']}", ""]
        lines += [f"- {d}" for d in case["data"]]
        lines += [""]
        diagnosis = DIAGNOSES.get(case["id"], "(diagnosis pending — inspect image)")
        lines += [f"> {diagnosis}", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "CATALOG.md").write_text("\n".join(lines) + "\n")
    print(f"wrote eval/failures/CATALOG.md ({len(cases)} cases)")
    for c in cases:
        print(f"  {c['id']}: {c['image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
