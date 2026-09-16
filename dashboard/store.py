"""Read-only store over the pipeline's local artifacts.

One object, built once at server startup, aggregating exactly what the
G5 dashboard pages need — no writes anywhere:

- ``cameras``  — manifest.parquet → per-camera stats + a SCHEMATIC map
  layout (no GPS/lat-lon exists in any local metadata; positions are a
  deterministic two-cluster grid, stated in the payload);
- ``species``  — e2e_union_v0_records.parquet → deployed-animal
  species counts (the union stage's own numbers, not re-derived);
- ``queue``    — the S4 agent corpus traces
  (processed/s4_corpus_traces_agent.jsonl): every ``approval_queue``
  action, with person evidence from the e2e union batches and the
  committed G4 corpus comparison as cross-reference;
- ``failures`` — the committed G3 failure catalog
  (eval/failures/CATALOG.md) parsed into gallery entries;
- ``resolve_image`` — the ONLY image-resolution path the server has
  (``frame:<image_id>`` via the manifest, ``failure:<file>`` via the
  committed gallery directory), each returning the stored person boxes
  for the frame so the serving path can blur them.

Everything is plain JSON-serializable data; pyarrow is imported lazily
so stdlib-only callers (the server core) stay importable without it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "processed"
DATA = REPO_ROOT / "data"
FAILURES_DIR = REPO_ROOT / "eval" / "failures"
FAILURES_IMAGES = FAILURES_DIR / "images"

#: schematic map: cameras per row inside each dataset cluster.
MAP_COLS = 6


class Store:
    """Aggregated, immutable view of the artifacts the pages serve."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.root = Path(repo_root)
        self.processed = self.root / "processed"
        self.data = self.root / "data"
        self.failures_images = self.root / "eval" / "failures" / "images"

        self._manifest_images: dict[str, dict[str, Any]] = {}
        self._rel_path: dict[str, str] = {}
        self._load_manifest()

        self._batch_person: dict[str, list[dict[str, Any]]] = {}
        self._load_batches()

        self.cameras = self._build_cameras()
        self.species = self._build_species()
        self.queue = self._build_queue()
        self.failures = self._build_failures()

    # -- manifest ----------------------------------------------------------

    def _load_manifest(self) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.processed / "manifest.parquet",
            columns=["image_id", "source_dataset", "rel_path", "location_id",
                     "camera_ts", "zone_restricted"])
        for iid, ds, rel, loc, ts, zone in zip(
                table.column("image_id").to_pylist(),
                table.column("source_dataset").to_pylist(),
                table.column("rel_path").to_pylist(),
                table.column("location_id").to_pylist(),
                table.column("camera_ts").to_pylist(),
                table.column("zone_restricted").to_pylist()):
            self._manifest_images[iid] = {
                "dataset": ds, "rel_path": rel, "camera_id": loc,
                "ts": str(ts), "zone_restricted": bool(zone)}
            self._rel_path[iid] = rel

    def _build_cameras(self) -> dict[str, Any]:
        agg: dict[str, dict[str, Any]] = {}
        for img in self._manifest_images.values():
            c = agg.setdefault(img["camera_id"], {
                "camera_id": img["camera_id"], "dataset": img["dataset"],
                "images": 0, "first_ts": img["ts"], "last_ts": img["ts"],
                "zone_restricted": img["zone_restricted"]})
            c["images"] += 1
            c["first_ts"] = min(c["first_ts"], img["ts"])
            c["last_ts"] = max(c["last_ts"], img["ts"])
            # any restricted image marks the camera restricted
            c["zone_restricted"] = c["zone_restricted"] or img["zone_restricted"]

        # optional richer stats from the S5 final report (processed/,
        # may be absent on a fresh clone — every field guarded)
        fp_by_cam: dict[str, int] = {}
        person_by_cam: dict[str, int] = {}
        s5 = self.processed / "s5_final_report.json"
        if s5.exists():
            doc = json.loads(s5.read_text())
            for row in doc["fp_camera_day"]["per_camera"]:
                fp_by_cam[row["camera_id"]] = row["fp"]
            for row in doc["person_load"]["per_camera"]:
                person_by_cam[row["camera_id"]] = row["dets"]

        # schematic layout: one cluster per dataset, deterministic grid
        clusters = sorted({c["dataset"] for c in agg.values()})
        cluster_w = 1.0 / len(clusters)
        out = []
        for ci, ds in enumerate(clusters):
            cams = sorted((c for c in agg.values() if c["dataset"] == ds),
                          key=lambda c: c["camera_id"])
            rows = (len(cams) + MAP_COLS - 1) // MAP_COLS
            for i, c in enumerate(cams):
                out.append({
                    **c,
                    "fp_sample": fp_by_cam.get(c["camera_id"]),
                    "person_dets_sample": person_by_cam.get(c["camera_id"]),
                    "x": round((ci + (i % MAP_COLS + 0.5) / MAP_COLS)
                               * cluster_w, 4),
                    "y": round((i // MAP_COLS + 0.5) / rows, 4)})
        return {
            "layout": ("schematic two-cluster grid — NO GPS/lat-lon "
                       "exists in any local metadata; x/y are positions "
                       "in this layout, not geography"),
            "clock": ("camera_ts is the ETL SYNTHETIC per-camera clock "
                      "(deterministic, seed-pinned)"),
            "n_cameras": len(out),
            "cameras": out,
        }

    # -- union batches (person evidence + species) --------------------------

    def _load_batches(self) -> None:
        path = self.processed / "e2e_union_v0_batches.jsonl"
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            batch = json.loads(line)
            src = batch["source"]
            iid = src["image_id"]
            persons = [d for d in batch["detections"] if d["label"] == "person"]
            if persons:
                self._batch_person[iid] = [
                    {"bbox_norm": d["bbox"], "conf": d["confidence"]}
                    for d in persons]

    def _build_species(self) -> dict[str, Any]:
        import pyarrow.parquet as pq

        table = pq.read_table(self.processed / "e2e_union_v0_records.parquet",
                              columns=["image_id", "label", "species"])
        labels = table.column("label").to_pylist()
        species = table.column("species").to_pylist()
        counts = Counter(sp for sp, lab in zip(species, labels)
                         if lab == "animal" and sp)
        images = {iid for iid, lab in zip(table.column("image_id").to_pylist(),
                                          labels) if lab == "animal"}
        return {
            "source": ("processed/e2e_union_v0_records.parquet — deployed "
                       "animal detections of the S4 union run (200-image "
                       "deterministic subsample), species by classifier v0"),
            "n_images_with_animal": len(images),
            "n_animal_records": sum(counts.values()),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }

    # -- approval queue -----------------------------------------------------

    def _build_queue(self) -> dict[str, Any]:
        traces_path = self.processed / "s4_corpus_traces_agent.jsonl"
        items: list[dict[str, Any]] = []
        if traces_path.exists():
            for line in traces_path.read_text().splitlines():
                t = json.loads(line)
                if t.get("action", {}).get("type") != "approval_queue":
                    continue
                iid = t["event_id"]
                persons = self._batch_person.get(iid, [])
                items.append({
                    "event_id": iid,
                    "image_id": iid,  # corpus event ids are the image ids
                    "camera_id": t.get("camera_id"),
                    "captured_at": t.get("captured_at"),
                    "reason": t["action"].get("reason"),
                    "rationale": t["action"].get("rationale", [])[:3],
                    "person_evidence": bool(persons),
                    "person_max_conf": (max(p["conf"] for p in persons)
                                        if persons else None),
                    "face_blur_required": bool(persons),
                    "image_url": f"/image?src=frame:{iid}",
                })
        items.sort(key=lambda it: (not it["person_evidence"], it["event_id"]))

        crossref = self.root / "agent" / "artifacts" / "corpus_comparison.json"
        baseline_note = None
        if crossref.exists():
            doc = json.loads(crossref.read_text())
            by_id = {e["event_id"]: e for e in doc.get("person_events", [])}
            for it in items:
                if it["event_id"] in by_id:
                    it["baseline_action"] = by_id[it["event_id"]]["baseline"]["action"]  # noqa: E501
            baseline_note = (f"committed G4 comparison: {len(by_id)} person "
                            "events, agent queues "
                            f"{doc['escalation_load']['agent_person_events_queued']} "
                            "vs baseline escalates "
                            f"{doc['escalation_load']['baseline_escalates']}")
        return {"n": len(items), "items": items, "baseline_note": baseline_note}

    # -- failure gallery ----------------------------------------------------

    _CASE_RE = re.compile(r"^## (\d+)\. (.+)$", re.MULTILINE)
    _IMG_RE = re.compile(r"!\[[^\]]*\]\(\s*images/([^)\s]+)\s*\)")
    _KEY_RE = re.compile(r"\*\*Key\*\* — (.+)")

    def _build_failures(self) -> dict[str, Any]:
        catalog = self.root / "eval" / "failures" / "CATALOG.md"
        if not catalog.exists():
            return {"n": 0, "cases": []}
        text = catalog.read_text()
        heads = list(self._CASE_RE.finditer(text))
        cases = []
        for i, m in enumerate(heads):
            body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads)
                        else len(text)]
            img = self._IMG_RE.search(body)
            key = self._KEY_RE.search(body)
            if not img:
                continue
            cases.append({
                "case": int(m.group(1)),
                "title": m.group(2).strip(),
                "image": img.group(1),
                "key": key.group(1).strip() if key else None,
                "image_url": f"/image?src=failure:{img.group(1)}",
            })
        return {
            "source": "eval/failures/CATALOG.md (committed G3 catalog)",
            "n": len(cases), "cases": cases,
        }

    # -- the single image-resolution path -----------------------------------

    def resolve_image(self, spec: str) -> tuple[Path, list[list[float]]]:
        """``frame:<image_id>`` | ``failure:<file>`` → (path, person boxes).

        Boxes are normalized [x1, y1, x2, y2, conf] from the e2e union
        batches; the serving path denormalizes against the decoded frame.
        Raises KeyError (unknown id/file) or ValueError (path outside the
        allowed roots — the server maps both to 404). The serving path
        additionally runs a live MDv6 person pass, so an empty list here
        never means "skip blur".
        """
        kind, _, value = spec.partition(":")
        if kind == "frame":
            img = self._manifest_images.get(value)
            if img is None:
                raise KeyError(value)
            path = (self.data / img["rel_path"]).resolve()
            if not path.is_relative_to(self.data.resolve()):
                raise ValueError(f"rel_path escapes data/: {img['rel_path']}")
            boxes: list[list[int]] = []
            for p in self._batch_person.get(value, []):
                x1, y1, x2, y2 = p["bbox_norm"]
                # normalized against the source dims stored in the batch;
                # recover them from the decoded frame at serve time
                boxes.append([round(x1, 6), round(y1, 6),
                              round(x2, 6), round(y2, 6), round(p["conf"], 4)])
            return path, boxes
        if kind == "failure":
            name = Path(value).name  # strip any path components
            path = (self.failures_images / name).resolve()
            if not path.is_relative_to(self.failures_images.resolve()):
                raise ValueError(f"failure image escapes gallery: {value}")
            if not path.is_file():
                raise KeyError(name)
            return path, []
        raise ValueError(f"unknown image spec kind: {kind!r}")

    # -- overview -----------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        return {
            "cameras": self.cameras["n_cameras"],
            "manifest_images": len(self._manifest_images),
            "species": self.species["counts"],
            "queue_size": self.queue["n"],
            "queue_person": sum(1 for it in self.queue["items"]
                                if it["person_evidence"]),
            "failure_cases": self.failures["n"],
        }
