"""Draw the deterministic Seattle(ish) held-out sample (seed 17).

Regenerates ``downloads/seattleish/sample_spec.json`` + ``fetch_urls.txt``
from the published metadata. Rules, all pinned for reproducibility:

- **Still images only.** The metadata mixes stills (``.jpg``) with
  video entries (``.avi``/``.mp4`` from ``converted_camera_trap_videos``);
  this evaluation runs the still-frame detector path, and a video's
  class label does not guarantee the subject is visible in any single
  frame — so videos are excluded, not first-framed. Consequence: the
  ``vehicle`` stratum is all stills that carry the label (**60** images;
  the other 19 vehicle-labelled entries are videos).
- Strata by label set: ``dan`` if dan ∈ labels (the dataset's one human;
  multi-label frames like dan+dog count as dan), ``vehicle`` likewise,
  ``empty`` only if labels == {empty}, ``animal`` for animal-only frames
  (species sub-quotas below; an image with several animal labels is
  drawn into the first sub-quota that still needs filling, species
  priority alphabetical).
- Deterministic: ``random.Random(SEED).sample(sorted(pool), k)`` — the
  sort makes the pool order machine-independent.

    ../wildlife-monitor/.venv/bin/python -m eval.sample_seattle [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
from collections import defaultdict
from pathlib import Path

SEED = 17
STILL_EXTS = {".jpg", ".jpeg", ".png"}
BASE = ("https://storage.googleapis.com/public-datasets-lila/"
        "seattleish-camera-traps/")
SOURCE = "seattleish-camera-traps @ lila.science (CDLA-Permissive-1.0)"

DAN_QUOTA = 800          # of 2,217 stills
EMPTY_QUOTA = 500        # of 14,273 stills
ANIMAL_QUOTA = {         # 380 total, of the animal-only stills
    "coyote": 120, "deer": 60, "dog": 30, "raccoon": 40,
    "rabbit": 30, "squirrel": 100,
}
# VEHICLE: all stills with the label — 60 of the 79 labelled entries.


def build_spec(meta: dict) -> dict:
    cat = {c["id"]: c["name"] for c in meta["categories"]}
    labels_by_image: dict[str, set[str]] = defaultdict(set)
    for a in meta["annotations"]:
        labels_by_image[a["image_id"]].add(cat[a["category_id"]])

    pools: dict[str, list[str]] = defaultdict(list)
    for img in meta["images"]:
        rel = img["file_name"]
        if Path(rel).suffix.lower() not in STILL_EXTS:
            continue
        labels = labels_by_image.get(img["id"], set())
        if "dan" in labels:
            pools["dan"].append(rel)
        elif "vehicle" in labels:
            pools["vehicle"].append(rel)
        elif labels == {"empty"}:
            pools["empty"].append(rel)
        elif labels:  # animal-only (any non-empty, non-person/vehicle set)
            pools[f"animal:{','.join(sorted(labels))}"].append(rel)

    rng = random.Random(SEED)
    files: list[str] = []
    labels_out: dict[str, list[str]] = {}
    strata: dict[str, int] = {}

    def draw(stratum: str, pool: list[str], quota: int) -> list[str]:
        take = rng.sample(sorted(pool), min(quota, len(pool)))
        strata[stratum] = strata.get(stratum, 0) + len(take)
        return take

    for rel in draw("dan", pools["dan"], DAN_QUOTA):
        files.append(rel); labels_out[rel] = ["dan"]
    for rel in draw("vehicle", pools["vehicle"], len(pools["vehicle"])):
        files.append(rel); labels_out[rel] = ["vehicle"]
    for rel in draw("empty", pools["empty"], EMPTY_QUOTA):
        files.append(rel); labels_out[rel] = ["empty"]
    animal_pools = sorted(k.split(":", 1)[1] for k in pools
                          if k.startswith("animal:"))
    drawn: set[str] = set()
    for species in sorted(ANIMAL_QUOTA):
        quota = ANIMAL_QUOTA[species]
        candidates = sorted(
            rel for combo in animal_pools if species in combo.split(",")
            for rel in pools[f"animal:{combo}"] if rel not in drawn)
        for rel in draw("animal", candidates, quota):
            files.append(rel); labels_out[rel] = ["animal"]; drawn.add(rel)

    # stable overall order: stratum-major (dan, vehicle, empty, animal)
    return {
        "seed": SEED, "source": SOURCE, "strata": strata,
        "labels": {f: labels_out[f] for f in files}, "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary, write nothing")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent / "downloads" / "seattleish"
    meta = json.loads((root / "seattleish_camera_traps.json").read_text())
    spec = build_spec(meta)

    print(f"strata: {spec['strata']}  total {len(spec['files'])}")
    n_vid = sum(1 for img in meta["images"]
                if Path(img["file_name"]).suffix.lower() not in STILL_EXTS)
    print(f"excluded video entries in metadata: {n_vid} "
          f"(.avi/.mp4 — still-frame eval; documented in the report)")
    if args.dry_run:
        return 0

    (root / "sample_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    urls = [BASE + urllib.parse.quote(rel) for rel in spec["files"]]
    (root / "fetch_urls.txt").write_text("\n".join(urls) + "\n")
    print(f"-> {root / 'sample_spec.json'}")
    print(f"-> {root / 'fetch_urls.txt'} ({len(urls)} urls, %-encoded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
