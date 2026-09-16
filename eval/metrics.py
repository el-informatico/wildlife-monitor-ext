"""Pure glue for the G3 perception evaluation (Sprint 3).

numpy-only pieces of the evaluation harness so the offline test suite
covers them without cv2, model files, or datasets:

- top-1 / top-k accuracy, micro AND macro, with per-class counts —
  micro is plain accuracy, macro is the unweighted mean of per-class
  accuracy (the long-tail view the brief asks for);
- split-integrity checking: a held-out number must never come from a
  training image, so the checker validates the evaluated crop set
  against the trainer's own split index and fails loudly on leaks,
  count drift, or duplicate ids;
- greedy detection↔GT matching — the S2 parity protocol (IoU ≥ 0.5,
  class-agnostic, each GT consumed once, greedy by detection score),
  reimplemented pure so tests run offline; the runner cross-checks it
  against wildlife_monitor.detect.parity like the e2e stage does;
- the content-based IR (night) flag from mean HSV saturation. The
  manifest's `is_night` is a synthetic random assignment uncorrelated
  with actual imaging conditions, so G3 defines day/night from pixels
  and only reports the synthetic flag for transparency;
- sequence logic: the MOG2 trigger gate + max-per-trigger selection
  over a sequence's frames, and the aggregation that turns per-frame
  gate/detection records into trigger-coverage and selection-loss
  numbers;
- CSV / markdown emitters for the metrics table.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# --- shared parameters (mirrors of upstream constants, asserted at run) -----
IR_SAT_THRESHOLD = 25.0   # mean HSV saturation below this => IR/night frame
TRIGGER_FG_THRESHOLD = 0.02  # motion.bench.TRIGGER_FG_THRESHOLD
MATCH_IOU = 0.5           # detect parity protocol IoU


class SplitIntegrityError(ValueError):
    """Raised when a held-out evaluation would touch non-held-out data."""


# ---------------------------------------------------------------------------
# Day/night: content-based IR flag
# ---------------------------------------------------------------------------


def ir_flag(mean_sat: float, threshold: float = IR_SAT_THRESHOLD) -> bool:
    """True when a frame's mean HSV saturation marks it as IR monochrome.

    Camera-trap night frames are infrared grayscale (saturation ~0); day
    frames are color (saturation tens-to-hundreds). The Missouri sample
    is sharply bimodal (p50 0.8 vs p95 122), so any threshold in the
    ~[5, 35] moat separates the modes; 25 sits in the middle of it.
    """
    return float(mean_sat) < threshold


def agreement_rate(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Fraction of positions where two boolean flags agree (0..1)."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.asarray(a, bool) == np.asarray(b, bool)))


# ---------------------------------------------------------------------------
# Top-k metrics, micro and macro
# ---------------------------------------------------------------------------


def topk_metrics(
    labels: Sequence[int],
    ranked: Sequence[Sequence[int]],
    n_classes: int,
    k: int = 5,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Top-1 / top-k accuracy, micro (plain) and macro (mean per class).

    `ranked[i]` is sample i's predicted class ids best-first (length ≥ 1;
    shorter than k is fine). Returns per-class counts so small splits are
    legible as small, and flags the degenerate case k ≥ n_classes where
    top-k is 1.0 by construction (classifier v0 has 4 classes).
    """
    if len(labels) != len(ranked):
        raise ValueError(f"length mismatch: {len(labels)} vs {len(ranked)}")
    if not labels:
        return {"n": 0, "k_effective": k, "degenerate": n_classes <= k}
    if any(not r for r in ranked):
        raise ValueError("every sample needs at least one ranked class")

    k_eff = min(k, n_classes)
    per_class: dict[int, dict[str, int]] = {
        c: {"n": 0, "top1": 0, "topk": 0} for c in range(n_classes)
    }
    for label, preds in zip(labels, ranked):
        hit1 = int(preds[0]) == int(label)
        hitk = int(label) in {int(p) for p in preds[:k_eff]}
        pc = per_class[int(label)]
        pc["n"] += 1
        pc["top1"] += hit1
        pc["topk"] += hitk

    def _acc(part: str) -> tuple[float, float]:
        micro = sum(pc[part] for pc in per_class.values()) / len(labels)
        present = [pc for pc in per_class.values() if pc["n"]]
        macro = sum(pc[part] / pc["n"] for pc in present) / len(present)
        return round(micro, 4), round(macro, 4)

    micro1, macro1 = _acc("top1")
    microk, macrok = _acc("topk")
    names = class_names or {c: str(c) for c in range(n_classes)}
    return {
        "n": len(labels),
        "k": k,
        "k_effective": k_eff,
        "degenerate_topk": n_classes <= k,  # top-k is 1.0 by construction
        "top1_micro": micro1,
        "top1_macro": macro1,
        f"top{k}_micro": microk,
        f"top{k}_macro": macrok,
        "per_class": {
            names[c]: {"n": pc["n"],
                       "top1_acc": round(pc["top1"] / pc["n"], 4) if pc["n"] else None,
                       f"top{k}_acc": round(pc["topk"] / pc["n"], 4) if pc["n"] else None}
            for c, pc in per_class.items()
        },
    }


def confusion_matrix(
    labels: Sequence[int],
    top1_preds: Sequence[int],
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Rows = true class, cols = predicted; plus per-class accuracy."""
    n = _n_classes_from(labels, top1_preds, class_names)
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, top1_preds):
        mat[int(t), int(p)] += 1
    names = class_names or {c: str(c) for c in range(n)}
    per_class = {}
    for c in range(n):
        total = int(mat[c].sum())
        per_class[names[c]] = {
            "n": total,
            "accuracy": round(float(mat[c, c] / total), 4) if total else None,
        }
    return {"matrix": mat.tolist(), "per_class": per_class}


def _n_classes_from(labels, preds, class_names) -> int:
    if class_names:
        return len(class_names)
    return max(max(labels), max(preds)) + 1 if labels and preds else 0


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


def check_split_integrity(
    entries: Iterable[dict],
    *,
    allowed_splits: set[str],
    expected_counts: dict[str, int] | None = None,
    vocab: set[str] | None = None,
    heldout_sources: set[str] | None = None,
    other_split_sources: set[str] | None = None,
) -> dict[str, Any]:
    """Validate that an evaluated set is exactly the held-out split.

    `entries`: crop/image records with at least {id, split, class_name,
    source_image}. Checks, in order:

    1. every entry's split is in `allowed_splits` (one set per number —
       a number must never mix splits);
    2. ids are unique;
    3. class names are inside the classifier vocabulary;
    4. if `expected_counts` is given (the trainer's own per-split crop
       counts), the evaluated n matches the held-out count exactly;
    5. if `heldout_sources` is given (the manifest's held-out split
       rel_paths), every entry's source image is one of them — the
       train-image leak guard;
    6. if `other_split_sources` is given (train+val rel_paths), the
       intersection with evaluated sources is empty.

    Returns a summary dict; raises SplitIntegrityError on any violation.
    """
    entries = list(entries)
    if not entries:
        raise SplitIntegrityError("no entries to evaluate — held-out set is empty")

    for e in entries:
        if e["split"] not in allowed_splits:
            raise SplitIntegrityError(
                f"entry {e['id']!r} has split {e['split']!r}, "
                f"allowed: {sorted(allowed_splits)}"
            )

    ids = [e["id"] for e in entries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SplitIntegrityError(f"duplicate ids in evaluated set: {dupes[:5]}")

    if vocab is not None:
        outside = sorted({e["class_name"] for e in entries} - vocab)
        if outside:
            raise SplitIntegrityError(f"class names outside vocabulary: {outside}")

    if expected_counts:
        got = {s: sum(1 for e in entries if e["split"] == s) for s in allowed_splits}
        for split, want in expected_counts.items():
            if split in allowed_splits and got[split] != want:
                raise SplitIntegrityError(
                    f"split {split!r}: evaluated {got[split]} crops, "
                    f"trainer's index says {want}"
                )

    sources = {e["source_image"] for e in entries}
    if heldout_sources is not None:
        stray = sorted(sources - heldout_sources)
        if stray:
            raise SplitIntegrityError(
                f"{len(stray)} evaluated source images are not in the "
                f"held-out manifest split (e.g. {stray[:3]})"
            )
    if other_split_sources is not None:
        leak = sorted(sources & other_split_sources)
        if leak:
            raise SplitIntegrityError(
                f"TRAIN/VAL LEAK: {len(leak)} evaluated source images belong "
                f"to a non-held-out split (e.g. {leak[:3]})"
            )

    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["class_name"]] = by_class.get(e["class_name"], 0) + 1
    return {
        "n": len(entries),
        "splits": sorted(allowed_splits),
        "n_source_images": len(sources),
        "by_class": dict(sorted(by_class.items())),
        "checks": [
            "split membership", "unique ids", "vocabulary",
            *(["trainer count match"] if expected_counts else []),
            *(["held-out sources", "no train/val overlap"] if heldout_sources else []),
        ],
    }


# ---------------------------------------------------------------------------
# Detection ↔ GT matching (parity protocol, pure reimplementation)
# ---------------------------------------------------------------------------


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt_boxes_xywh: Sequence[Sequence[float]],
    det_boxes_xyxy: Sequence[Sequence[float]],
    det_scores: Sequence[float],
    iou_threshold: float = MATCH_IOU,
) -> list[int]:
    """Greedy score-ordered matching; returns the GT index per detection.

    Same protocol as wildlife_monitor.detect.parity.match_image (IoU ≥
    threshold, class-agnostic, each GT box consumed at most once, greedy
    by detection score): each detection, best IoU first, claims its
    argmax GT box — and if that box is already consumed the detection is
    UNMATCHED (no fallback to a lesser-overlapping free GT; a
    cross-check against the parity module over the real evaluation
    images pins this exact rule). -1 = unmatched. Pure numpy so the
    offline tests exercise it directly.
    """
    order = np.argsort(-np.asarray(det_scores, dtype=np.float64), kind="stable")
    gts = np.asarray(
        [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes_xywh],
        dtype=np.float64,
    ).reshape(-1, 4)
    dets = np.asarray(det_boxes_xyxy, dtype=np.float64).reshape(-1, 4)

    det_gt = [-1] * len(det_boxes_xyxy)
    matched: set[int] = set()
    for di in order:
        di = int(di)
        if not len(gts):
            break
        ious = [iou_xyxy(dets[di], gts[gi]) for gi in range(len(gts))]
        best = int(np.argmax(ious))  # ties resolve to the lowest gt index
        if ious[best] >= iou_threshold and best not in matched:
            matched.add(best)
            det_gt[di] = best
    return det_gt


def pr_counts(
    det_matched: Sequence[bool], n_gt: int, n_images: int
) -> dict[str, Any]:
    """TP/FP/FN + recall/precision/FP-per-image from per-detection match flags."""
    tp = sum(1 for m in det_matched if m)
    fp = len(det_matched) - tp
    fn = n_gt - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_images": n_images,
        "n_gt": n_gt,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "fp_per_image": round(fp / n_images, 4) if n_images else None,
    }


# ---------------------------------------------------------------------------
# Sequence logic: MOG2 trigger gate + max-per-trigger selection
# ---------------------------------------------------------------------------


def gate_pass(fg_ratio: float, warmup: bool, threshold: float = TRIGGER_FG_THRESHOLD) -> bool:
    """A frame passes the motion gate when MOG2's foreground ratio exceeds
    the trigger threshold. Warmup frames (no background model yet) never
    pass — the S1 motion-bench convention."""
    return (not warmup) and fg_ratio > threshold


def select_max_per_trigger(frames: Sequence[dict]) -> tuple[int | None, str]:
    """Max-per-trigger selection over one trigger window (sequence).

    `frames`, in capture order, each: {"gate": bool, "dets": [{"conf":
    float, "matched": bool}, ...]} with only deployed animal detections
    listed. Selection: among frames that pass the MOG2 gate, the frame
    holding the highest-confidence animal detection; its winning
    detection is the sequence representative. Returns (frame_index,
    reason) with reason in {"selected", "no_gate_pass", "no_detection"}:
    the latter two mean the sequence produced no representative — the
    motion gate suppressed it, or nothing was detected on gated frames.
    """
    best_idx, best_conf = None, -1.0
    any_gate = False
    for i, fr in enumerate(frames):
        if not fr.get("gate"):
            continue
        any_gate = True
        for det in fr.get("dets", ()):
            if det["conf"] > best_conf:
                best_conf = det["conf"]
                best_idx = i
    if best_idx is not None:
        return best_idx, "selected"
    return None, ("no_gate_pass" if not any_gate else "no_detection")


def sequence_rollup(frames: Sequence[dict], selection_reason: str) -> dict[str, Any]:
    """Per-sequence roll-up from per-frame records.

    Frame fields: {"image_id", "warmup", "gate", "has_gt" (a GT box on
    this frame), "matched" (a deployed animal detection matched GT on
    this frame), "animal_dets"/"animal_fp" (deployed animal detections /
    unmatched ones on gated frames), "selected": bool (this frame chosen
    by select_max_per_trigger)}. `selection_reason` is the reason string
    returned by select_max_per_trigger for the whole sequence.
    """
    n = len(frames)
    analyzed = [f for f in frames if f.get("gate")]
    selected = next((f for f in frames if f.get("selected")), None)
    return {
        "frames": n,
        "frames_warmup": sum(1 for f in frames if f.get("warmup")),
        "frames_gated": len(analyzed),
        "has_gt": any(f.get("has_gt") for f in frames),
        "gt_frames": sum(1 for f in frames if f.get("has_gt")),
        "gt_frames_gated": sum(1 for f in frames if f.get("has_gt") and f.get("gate")),
        "union_matched": any(f.get("matched") for f in frames),
        "gated_union_matched": any(f.get("matched") for f in analyzed),
        "selected_reason": selection_reason,
        "selected_matched": bool(selected and selected.get("matched")),
        "animal_dets_analyzed": sum(len(f.get("animal_dets", ())) for f in analyzed),
        "animal_fp_analyzed": sum(len(f.get("animal_fp", ())) for f in analyzed),
        "animal_fp_selected": len(selected.get("animal_fp", ())) if selected else 0,
    }


def aggregate_sequences(
    records: Sequence[dict], scopes: dict[str, list[str]]
) -> dict[str, dict[str, Any]]:
    """Aggregate per-sequence rollups into per-scope metrics.

    `records`: one sequence_rollup output per sequence, plus the scope
    key/value pairs named in `scopes` (e.g. {"dataset": ["missouri"],
    "day_night": ["day", "night"]}) — every combination of scope values
    is emitted (a sequence with a value not listed lands in no scope).
    Scope "all" is always included.
    """
    import itertools

    def _block(rows: list[dict]) -> dict[str, Any]:
        gt_rows = [r for r in rows if r["has_gt"]]
        n_frames = sum(r["frames"] for r in rows)
        n_gated = sum(r["frames_gated"] for r in rows)

        def _rate(num: int, den: int) -> float | None:
            return round(num / den, 4) if den else None

        return {
            "sequences": len(rows),
            "frames": n_frames,
            "frames_warmup": sum(r["frames_warmup"] for r in rows),
            "frames_gated": n_gated,
            "frame_gate_rate": _rate(n_gated, n_frames - sum(r["frames_warmup"] for r in rows)),
            "inference_saving": _rate(n_frames - n_gated, n_frames),
            "gt_sequences": len(gt_rows),
            "gt_boxes": sum(r["gt_frames"] for r in rows),
            "gt_frames_gated": sum(r["gt_frames_gated"] for r in rows),
            "gt_frame_gate_rate": _rate(
                sum(r["gt_frames_gated"] for r in rows),
                sum(r["gt_frames"] for r in rows),
            ),
            "seq_union_recall": _rate(
                sum(1 for r in gt_rows if r["union_matched"]), len(gt_rows)
            ),
            "seq_gated_recall": _rate(
                sum(1 for r in gt_rows if r["gated_union_matched"]), len(gt_rows)
            ),
            "seq_selected_recall": _rate(
                sum(1 for r in gt_rows if r["selected_matched"]), len(gt_rows)
            ),
            "selection_loss": _rate(
                sum(1 for r in gt_rows if r["gated_union_matched"] and not r["selected_matched"]),
                len(gt_rows),
            ),
            "animal_dets_analyzed": sum(r["animal_dets_analyzed"] for r in rows),
            "fp_per_analyzed_frame": _rate(
                sum(r["animal_fp_analyzed"] for r in rows), n_gated
            ),
            "fp_per_sequence_selected": _rate(
                sum(r["animal_fp_selected"] for r in rows), len(rows)
            ),
        }

    keys = list(scopes)
    out: dict[str, dict[str, Any]] = {"all": _block(list(records))}
    for combo in itertools.product(*(scopes[k] for k in keys)):
        rows = [r for r in records
                if all(r.get(k) == v for k, v in zip(keys, combo))]
        label = "|".join(f"{k}={v}" for k, v in zip(keys, combo))
        out[label] = _block(rows)
    return out


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------


def rows_to_csv(rows: Sequence[dict]) -> str:
    """List of flat dicts -> CSV text (columns = first-seen key order)."""
    if not rows:
        return ""
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        return f'"{s}"' if ("," in s or '"' in s or "\n" in s) else s

    lines = [",".join(columns)]
    lines += [",".join(_cell(r.get(c)) for c in columns) for r in rows]
    return "\n".join(lines) + "\n"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Simple pipe table; None renders as '-'; '|' inside a cell is
    escaped so scope labels like "dataset|day_night" stay one column."""

    def _cell(v: Any) -> str:
        if v is None:
            return "-"
        return str(v).replace("|", "\\|")

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
