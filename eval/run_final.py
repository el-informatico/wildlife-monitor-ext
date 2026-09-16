"""Run the S5 final evaluation: FP per camera per day + person analysis.

Sprint 5's brief asks for "eval completa (Wellington persona P/R,
FP/cámara/día)". This runner delivers what the local environment can
measure honestly and states exactly what it cannot:

- **Wellington probe** — is the Wellington camera-traps dataset on disk?
  (It was never downloaded: `downloads/` holds african-wildlife,
  missouri images, and the two metadata JSONs only.) If absent, the
  person P/R cell is reported as a GAP, not substituted silently.
- **Person ground-truth audit** — do ANY local GT sources carry person
  boxes? (Missouri's 21 categories and CCT20's 22 are all animals +
  empty; african_wildlife GT is animal-only.) With zero person boxes
  anywhere, person precision/recall is UNDEFINED at any scope — what is
  measurable is the person-detection load (counts, cameras, days,
  confidences) that flows into the approval queue.
- **FP/camera/day** — false positives per camera per calendar day from
  the stored S2 cv2.dnn detections over the 800-image sample, deploy
  conf 0.25, parity protocol (greedy IoU ≥ 0.5, class-agnostic, no
  fallback). Overall P/R is recomputed first and cross-checked against
  G3's committed numbers before any per-cell table is emitted.

Inputs (all existing artifacts — no pixels touched, no re-inference):

- ``processed/eval_sample_s2.json`` — GT boxes (xywh absolute) + strata;
- ``processed/cv2_detections_s2_new.json`` — stored detections at floor
  0.10 (deploy filter applied here);
- ``processed/manifest.parquet`` — camera_id (location_id) + camera_ts
  per image_id (the ETL's SYNTHETIC per-camera clock: deterministic,
  seed-pinned, monotonic per camera — the day partition is simulated,
  stated in every table that uses it);
- ``processed/g3_ir_flags.json`` — content-IR night flags;
- ``agent/artifacts/corpus_comparison.json`` — committed S4 comparison,
  for the person-event queue cross-reference.

Outputs: ``processed/s5_final_report.json`` + ``processed/
s5_final_tables.md`` (the fragment eval/S5-REPORT.md embeds verbatim).

    python -m eval.run_final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "processed"

#: deploy confidence — identical to the G3 detection stage / pipeline.
from pipeline import e2e  # noqa: E402

DEPLOY_CONF = e2e.DEPLOY_CONF
DET_LABELS = e2e.DET_LABELS
PERSON_CLS = [cls for cls, name in DET_LABELS.items() if name == "person"]

#: G3's committed overall TP/FP/FN at deploy 0.25 (eval/G3-REPORT.md) —
#: the cross-check target; a mismatch means the protocol drifted.
G3_COMMITTED = {"tp": 812, "fp": 676, "fn": 210}

#: what "person" looks like in a GT category vocabulary.
PERSON_GT_NAMES = {"person", "human", "people", "homo_sapiens"}

WELLINGTON_DIRS = ("wellington", "wellington_camera_traps",
                   "wellington-camera-traps")


# ---------------------------------------------------------------------------
# Inputs
#


def load_sample(path: Path = PROCESSED / "eval_sample_s2.json"
                ) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_detections(path: Path = PROCESSED / "cv2_detections_s2_new.json"
                    ) -> dict[str, dict[str, Any]]:
    """Stored per-image detections + the artifact's provenance block."""
    art = json.loads(path.read_text())
    return {"detections": art["detections"], "meta": {
        k: art[k] for k in ("model", "params", "versions") if k in art}}


def load_camera_days(path: Path = PROCESSED / "manifest.parquet",
                     image_ids: set[str] | None = None,
                     ) -> dict[str, dict[str, str]]:
    """image_id → {camera_id, day} from the manifest (lazy pyarrow).

    ``day`` is the calendar day of the synthetic camera_ts (YYYY-MM-DD)
    — the only clock either dataset has.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image_id", "location_id",
                                         "camera_ts"])
    out: dict[str, dict[str, str]] = {}
    for iid, loc, ts in zip(table.column("image_id").to_pylist(),
                            table.column("location_id").to_pylist(),
                            table.column("camera_ts").to_pylist()):
        if image_ids is not None and iid not in image_ids:
            continue
        if iid in out:
            raise ValueError(f"duplicate manifest row for {iid!r}")
        out[iid] = {"camera_id": loc, "day": str(ts)[:10]}
    return out


def load_ir_flags(path: Path = PROCESSED / "g3_ir_flags.json"
                  ) -> dict[str, bool]:
    return {k: bool(v) for k, v in
            json.loads(path.read_text())["flags"].items()}


# ---------------------------------------------------------------------------
# Stage: Wellington + person-GT presence
#


def wellington_probe(data_dir: Path = REPO_ROOT / "data",
                     downloads_dir: Path = REPO_ROOT / "downloads",
                     manifests_dir: Path = REPO_ROOT / "manifests",
                     ) -> dict[str, Any]:
    """Presence checks for Wellington and for person GT anywhere local.

    Scans data/ subdirectories and downloads/ archives for a Wellington
    name, and audits every GT vocabulary reachable offline (the two
    committed metadata manifests + the S2 sample itself) for a person
    category. Returns a factual record — the caller renders the gap.
    """
    hits = []
    for base in (data_dir, downloads_dir):
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            name = entry.name.lower()
            if any(w in name for w in WELLINGTON_DIRS):
                hits.append(str(entry))

    vocab: dict[str, list[str]] = {}
    for mf in sorted(manifests_dir.glob("*.json")):
        try:
            doc = json.loads(mf.read_text())
        except (OSError, ValueError):
            continue
        cats = [c.get("name", "") for c in doc.get("categories", [])]
        if cats:
            vocab[mf.name] = cats
    sample_cats = sorted({b["category"] for img in load_sample()["images"]
                          for b in img.get("gt_boxes", [])})
    vocab["eval_sample_s2 (both datasets)"] = sample_cats

    person_in_vocab = sorted(
        name for name, cats in vocab.items()
        if any(c.strip().lower() in PERSON_GT_NAMES for c in cats))

    return {
        "wellington_present": bool(hits),
        "wellington_hits": hits,
        "checked_locations": [str(data_dir), str(downloads_dir)],
        "gt_vocabularies": vocab,
        "person_gt_in_any_vocabulary": person_in_vocab,
        "person_p_r_status": (
            "computable" if hits or person_in_vocab else "NOT_COMPUTABLE"),
    }


# ---------------------------------------------------------------------------
# Stage: matching (parity protocol, reused verbatim)
#


def match_image(img: dict[str, Any], dets: list[dict[str, Any]],
                conf: float) -> dict[str, Any]:
    """One image through the G3 detection protocol.

    Returns GT count, per-detection match flags (deploy set only), the
    unmatched deployed detections (the FPs), and the person/vehicle
    labels of the deployed set.
    """
    deployed = [d for d in dets if d["conf"] >= conf]
    mapping = metrics.greedy_match(
        [b["bbox"] for b in img["gt_boxes"]],
        [d["bbox"] for d in deployed],
        [d["conf"] for d in deployed],
    )
    matched = [m >= 0 for m in mapping]
    return {
        "n_gt": len(img["gt_boxes"]),
        "n_deployed": len(deployed),
        "matched": matched,
        "fp_dets": [d for d, m in zip(deployed, matched) if not m],
        "person_dets": [d for d in deployed
                        if d["cls"] in PERSON_CLS],
        "vehicle_dets": [d for d in deployed
                         if DET_LABELS.get(d["cls"]) == "vehicle"],
    }


# ---------------------------------------------------------------------------
# Stage: FP per camera per day
#


def _cells_add(cell: dict[str, int], fps: int, images: int = 1) -> None:
    cell["fp"] += fps
    cell["images"] += images


def fp_camera_day(sample: dict[str, Any], detections: dict[str, list[dict]],
                  camera_days: dict[str, dict[str, str]],
                  ir_flags: dict[str, bool], conf: float = DEPLOY_CONF,
                  ) -> dict[str, Any]:
    """The FP/camera/day table + the parity cross-check in one pass."""
    per_scope: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0})
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"fp": 0, "images": 0})
    persons: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"dets": 0, "images": 0, "max_conf": 0.0})
    person_conf: list[float] = []

    for img in sample["images"]:
        iid = img["image_id"]
        res = match_image(img, detections.get(iid, []), conf)
        cam_day = camera_days[iid]
        key = (cam_day["camera_id"], cam_day["day"])
        _cells_add(cells[key], len(res["fp_dets"]))
        if res["person_dets"]:
            p = persons[key]
            p["dets"] += len(res["person_dets"])
            p["images"] += 1
            p["max_conf"] = max(p["max_conf"],
                                *(d["conf"] for d in res["person_dets"]))
            person_conf.extend(d["conf"] for d in res["person_dets"])

        night = "night" if ir_flags.get(iid, False) else "day"
        tp = sum(res["matched"])
        for scope in (("all", "all"), (img["source_dataset"], "all"),
                      ("all", night), (img["source_dataset"], night)):
            s = per_scope[scope]
            s["tp"] += tp
            s["fp"] += len(res["fp_dets"])
            s["fn"] += res["n_gt"] - tp

    overall = dict(per_scope[("all", "all")])
    crosscheck = "match" if overall == G3_COMMITTED else "MISMATCH"

    # per-cell distribution over ACTIVE cells (camera actually imaged that day)
    fp_values = [c["fp"] for c in cells.values()]
    import numpy as np

    def _pct(q: float) -> float:
        return round(float(np.percentile(fp_values, q)), 3) if fp_values else None  # noqa: E501

    per_camera: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"fp": 0, "images": 0, "days": set()})
    for (cam, day), c in cells.items():
        pc = per_camera[cam]
        pc["fp"] += c["fp"]
        pc["images"] += c["images"]
        pc["days"].add(day)
    camera_rows = [
        {"camera_id": cam, "images": v["images"], "days_active": len(v["days"]),
         "fp": v["fp"],
         "fp_per_day": round(v["fp"] / len(v["days"]), 4),
         "fp_per_image": round(v["fp"] / v["images"], 4)}
        for cam, v in sorted(per_camera.items())
    ]

    worst = sorted(
        ({"camera_id": cam, "day": day, **c} for (cam, day), c in cells.items()),
        key=lambda r: (-r["fp"], r["camera_id"], r["day"]))[:10]

    by_dataset = defaultdict(lambda: {"fp": 0, "images": 0, "cameras": set(),
                                      "cells": 0})
    for (cam, _day), c in cells.items():
        ds = cam.rsplit("-L", 1)[0] if "-L" in cam else cam
        b = by_dataset[ds]
        b["fp"] += c["fp"]
        b["images"] += c["images"]
        b["cameras"].add(cam)
        b["cells"] += 1

    def _rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "conf": conf,
        "parity_crosscheck": {"computed": overall, "g3_committed": G3_COMMITTED,
                              "verdict": crosscheck},
        "pr_by_scope": {
            f"{ds}|{dn}": {**v,
                           "recall": _rate(v["tp"], v["tp"] + v["fn"]),
                           "precision": _rate(v["tp"], v["tp"] + v["fp"])}
            for (ds, dn), v in sorted(per_scope.items())
        },
        "cells": {
            "n_cells": len(cells),
            "n_images": sum(c["images"] for c in cells.values()),
            "fp_total": sum(fp_values),
            "fp_per_active_cell": {
                "mean": round(float(np.mean(fp_values)), 4) if fp_values else None,  # noqa: E501
                "p50": _pct(50), "p95": _pct(95),
                "max": max(fp_values) if fp_values else None,
            },
            "zero_fp_cells": sum(1 for v in fp_values if v == 0),
            "worst_10": worst,
        },
        "per_camera": camera_rows,
        "by_dataset": [
            {"dataset": ds, "cameras": len(v["cameras"]), "cells": v["cells"],
             "images": v["images"], "fp": v["fp"],
             "fp_per_camera_day": round(v["fp"] / v["cells"], 4)}
            for ds, v in sorted(by_dataset.items())
        ],
        "clock": ("calendar day of the manifest camera_ts — the ETL's "
                  "SYNTHETIC per-camera clock (deterministic, seed-pinned, "
                  "monotonic per camera): FP counts are measured, the day "
                  "partition is simulated"),
    }


# ---------------------------------------------------------------------------
# Stage: person-event load (the honest substitute for person P/R)
#


def person_load(sample: dict[str, Any], detections: dict[str, list[dict]],
                camera_days: dict[str, dict[str, str]],
                conf: float = DEPLOY_CONF) -> dict[str, Any]:
    """Person-labeled deployed detections: where, when, how confident.

    No local GT carries person boxes, so nothing here is a TP/FP claim;
    this is the load that reaches the approval queue and triggers face
    blur, characterized the way G3's catalog characterized it.
    """
    per_cam: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"dets": 0, "images": 0, "max_conf": 0.0})
    per_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"dets": 0, "images": 0})
    confs: list[float] = []
    images_with_person = 0

    for img in sample["images"]:
        iid = img["image_id"]
        pd = [d for d in detections.get(iid, [])
              if d["cls"] in PERSON_CLS and d["conf"] >= conf]
        if not pd:
            continue
        images_with_person += 1
        cd = camera_days[iid]
        c = per_cam[cd["camera_id"]]
        c["dets"] += len(pd)
        c["images"] += 1
        c["max_conf"] = max(c["max_conf"], *(d["conf"] for d in pd))
        d = per_day[cd["day"]]
        d["dets"] += len(pd)
        d["images"] += 1
        confs.extend(x["conf"] for x in pd)

    import numpy as np
    confs_arr = np.asarray(confs, dtype=np.float64)
    return {
        "conf": conf,
        "images_with_person_det": images_with_person,
        "person_dets": len(confs),
        "conf_stats": {
            "min": round(float(confs_arr.min()), 4) if len(confs) else None,
            "p50": round(float(np.percentile(confs_arr, 50)), 4) if len(confs) else None,  # noqa: E501
            "max": round(float(confs_arr.max()), 4) if len(confs) else None,
        },
        "per_camera": [
            {"camera_id": cam, **v} for cam, v in sorted(per_cam.items())
        ],
        "per_day": [
            {"day": day, **v} for day, v in sorted(per_day.items())
        ],
        "gt_status": ("zero person boxes in every local GT vocabulary — "
                      "precision/recall for the person class is UNDEFINED "
                      "here; counts below are detection load, not errors"),
    }


def agent_queue_crossref(path: Path = REPO_ROOT / "agent" / "artifacts"
                         / "corpus_comparison.json") -> dict[str, Any] | None:
    """The committed S4 comparison — person events the queue actually saw."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return {
        "schema": doc.get("schema"),
        "person_events_total": len(doc.get("person_events", [])),
        "escalation_load": doc.get("escalation_load"),
        "divergent_event_ids": [e["event_id"] for e in
                                doc.get("divergent_events", [])],
    }


# ---------------------------------------------------------------------------
# Tables
#


def render_tables(report: dict[str, Any]) -> str:
    """The markdown fragment S5-REPORT.md embeds verbatim."""
    md: list[str] = []
    fp = report["fp_camera_day"]
    pc = fp["cells"]

    md.append("## FP per camera per day (S2 800-image sample, deploy conf "
              f"{fp['conf']}, parity protocol IoU ≥ {metrics.MATCH_IOU})\n")
    x = fp["parity_crosscheck"]

    def _tf(v: dict[str, int]) -> str:
        return f"TP {v['tp']} / FP {v['fp']} / FN {v['fn']}"

    md.append(f"Parity cross-check vs G3 committed TP/FP/FN: **{x['verdict']}** "
              f"(computed {_tf(x['computed'])} vs committed "
              f"{_tf(x['g3_committed'])}).\n")

    md.append("| scope | TP | FP | FN | recall | precision |")
    md.append("|---|---|---|---|---|---|")
    for scope, v in fp["pr_by_scope"].items():
        rec = "—" if v["recall"] is None else f"{v['recall']:.4f}"
        prec = "—" if v["precision"] is None else f"{v['precision']:.4f}"
        md.append(f"| {scope} | {v['tp']} | {v['fp']} | {v['fn']} "
                  f"| {rec} | {prec} |")
    md.append("")

    d = pc["fp_per_active_cell"]
    md.append(f"Active camera×day cells: **{pc['n_cells']}** "
              f"({pc['n_images']} images), total FP {pc['fp_total']}; "
              f"{pc['zero_fp_cells']} cells at zero FP.")
    md.append(f"FP per active cell: mean {d['mean']} · p50 {d['p50']} · "
              f"p95 {d['p95']} · **max {d['max']}**.\n")

    md.append("| dataset | cameras | cells | images | FP | FP/camera/day |")
    md.append("|---|---|---|---|---|---|")
    for b in fp["by_dataset"]:
        md.append(f"| {b['dataset']} | {b['cameras']} | {b['cells']} "
                  f"| {b['images']} | {b['fp']} | {b['fp_per_camera_day']} |")
    md.append("")

    md.append("Worst 10 cells:\n")
    md.append("| camera | day | FP | images |")
    md.append("|---|---|---|---|")
    for w in pc["worst_10"]:
        md.append(f"| {w['camera_id']} | {w['day']} | {w['fp']} "
                  f"| {w['images']} |")
    md.append("")

    md.append("Per camera (all 47 in the sample; sorted by FP desc):\n")
    md.append("| camera | images | days active | FP | FP/day | FP/image |")
    md.append("|---|---|---|---|---|---|")
    for r in sorted(fp["per_camera"], key=lambda r: (-r["fp"], r["camera_id"])):
        md.append(f"| {r['camera_id']} | {r['images']} | {r['days_active']} "
                  f"| {r['fp']} | {r['fp_per_day']} | {r['fp_per_image']} |")
    md.append("")
    md.append(f"Clock: {fp['clock']}.\n")

    pl = report["person_load"]
    md.append("## Person-event load (deploy conf "
              f"{pl['conf']}) — no person GT exists locally\n")
    md.append(pl["gt_status"] + ".\n")
    cs = pl["conf_stats"]
    md.append(f"{pl['images_with_person_det']} images with ≥1 person-labeled "
              f"detection, {pl['person_dets']} person detections "
              f"(conf p50 {cs['p50']}, max {cs['max']}).\n")
    md.append("| camera | images | person dets | max conf |")
    md.append("|---|---|---|---|")
    for r in sorted(pl["per_camera"], key=lambda r: (-r["dets"], r["camera_id"])):
        md.append(f"| {r['camera_id']} | {r['images']} | {r['dets']} "
                  f"| {r['max_conf']:.4f} |")
    md.append("")
    md.append("| day | images | person dets |")
    md.append("|---|---|---|")
    for r in pl["per_day"]:
        md.append(f"| {r['day']} | {r['images']} | {r['dets']} |")
    md.append("")

    wq = report["wellington"]
    md.append("## Wellington / person-GT probe\n")
    md.append(f"- Wellington dataset on disk: **{'yes' if wq['wellington_present'] else 'NO'}**"
              + (f" ({wq['wellington_hits']})" if wq["wellington_hits"] else "")
              + ".")
    person_vocab = wq["person_gt_in_any_vocabulary"]
    md.append(f"- Person category in any local GT vocabulary: "
              f"**{'yes: ' + ', '.join(person_vocab) if person_vocab else 'NO'}**.")
    for name, cats in wq["gt_vocabularies"].items():
        shown = ", ".join(cats[:8]) + (" …" if len(cats) > 8 else "")
        md.append(f"  - {name}: {len(cats)} categories — {shown}")
    md.append(f"- Person P/R status: **{wq['person_p_r_status']}**.\n")

    aq = report.get("agent_queue_crossref")
    if aq:
        md.append("## Agent queue cross-reference (committed S4 artifact)\n")
        md.append(f"- Person events in the 200-event corpus: "
                  f"{aq['person_events_total']}.")
        el = aq.get("escalation_load") or {}
        md.append(f"- Escalation load: agent escalates "
                  f"{el.get('agent_escalates')} / queues "
                  f"{el.get('agent_person_events_queued')} person events; "
                  f"baseline escalates {el.get('baseline_escalates')}.\n")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Main
#


def build_report() -> dict[str, Any]:
    t0 = time.perf_counter()
    sample = load_sample()
    det_art = load_detections()
    image_ids = {img["image_id"] for img in sample["images"]}
    camera_days = load_camera_days(image_ids=image_ids)
    missing = image_ids - set(camera_days)
    if missing:
        raise RuntimeError(f"sample images absent from manifest: "
                           f"{sorted(missing)[:5]}")
    ir_flags = load_ir_flags()

    report: dict[str, Any] = {
        "schema": "wfm-s5-final/v1",
        "inputs": {
            "sample": {"images": len(sample["images"]),
                       "gt_boxes": sample["counts"]["gt_boxes"]},
            "detections": det_art["meta"],
        },
        "wellington": wellington_probe(),
        "fp_camera_day": fp_camera_day(sample, det_art["detections"],
                                       camera_days, ir_flags),
        "person_load": person_load(sample, det_art["detections"], camera_days),
        "agent_queue_crossref": agent_queue_crossref(),
    }
    report["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-prefix", default="s5_final",
                    help="processed/ output prefix (default s5_final)")
    args = ap.parse_args(argv)

    report = build_report()
    out_json = PROCESSED / f"{args.out_prefix}_report.json"
    out_md = PROCESSED / f"{args.out_prefix}_tables.md"
    out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    tables = render_tables(report)
    out_md.write_text(tables)

    x = report["fp_camera_day"]["parity_crosscheck"]
    print(f"parity cross-check vs G3: {x['verdict']} ({x['computed']})")
    cells = report["fp_camera_day"]["cells"]
    print(f"camera×day cells: {cells['n_cells']}, "
          f"FP/active-cell mean {cells['fp_per_active_cell']['mean']} "
          f"max {cells['fp_per_active_cell']['max']}")
    pl = report["person_load"]
    print(f"person load: {pl['person_dets']} dets on "
          f"{pl['images_with_person_det']} images; "
          f"person P/R: {report['wellington']['person_p_r_status']}")
    print(f"-> {out_json.relative_to(REPO_ROOT)}")
    print(f"-> {out_md.relative_to(REPO_ROOT)}")
    if x["verdict"] != "match":
        print("REFUSING to bless a drifted protocol — inspect before using "
              "these tables.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
