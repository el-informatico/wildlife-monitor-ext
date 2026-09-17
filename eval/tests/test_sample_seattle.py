"""Offline tests for the Seattle(ish) sampler (eval/sample_seattle.py).

Synthetic metadata fixture (three stills, one video, multi-label
dan+dog) pins the rules that matter: videos excluded, dan wins over
vehicle and over animal labels, empty only when labels == {empty},
determinism across runs, and quotas capping at pool size.
"""

from __future__ import annotations

from eval.sample_seattle import SEED, build_spec


def _meta(entries):
    """entries: list of (file_name, [label names]) → CCT-style metadata."""
    images, annotations = [], []
    cats = {}
    for i, (rel, labels) in enumerate(entries, start=1):
        images.append({"id": f"img{i}", "file_name": rel})
        for j, lab in enumerate(sorted(set(labels)), start=1):
            cats[len(cats) + 1] = lab
            cid = next(c for c, n in cats.items() if n == lab)
            annotations.append({"image_id": f"img{i}", "category_id": cid})
    return {
        "categories": [{"id": c, "name": n} for c, n in cats.items()],
        "images": images,
        "annotations": annotations,
    }


def test_videos_excluded_and_dan_wins():
    meta = _meta([
        ("camera_trap_images/a/dan/IMG_1.JPG", ["dan", "dog"]),
        ("converted_camera_trap_videos/b/dan/DSCF_1.AVI", ["dan"]),
        ("camera_trap_images/c/empty/IMG_2.JPG", ["empty"]),
        ("camera_trap_images/d/coyote/IMG_3.JPG", ["coyote", "dog"]),
    ])
    spec = build_spec(meta)
    # the .AVI is excluded; dan+dog → dan; empty → empty; coyote+dog → animal
    assert spec["files"] == ["camera_trap_images/a/dan/IMG_1.JPG",
                             "camera_trap_images/c/empty/IMG_2.JPG",
                             "camera_trap_images/d/coyote/IMG_3.JPG"]
    assert spec["strata"] == {"dan": 1, "vehicle": 0, "empty": 1, "animal": 1}
    assert spec["labels"]["camera_trap_images/a/dan/IMG_1.JPG"] == ["dan"]
    assert spec["labels"]["camera_trap_images/d/coyote/IMG_3.JPG"] == ["animal"]


def test_empty_only_when_exactly_empty():
    meta = _meta([
        ("x/empty_dog.JPG", ["empty", "dog"]),   # not empty: animal
        ("y/empty.JPG", ["empty"]),
    ])
    spec = build_spec(meta)
    # empty_dog is animal-only → drawn into an animal sub-quota (dog=30 ≥ 1)
    labels = {tuple(v) for v in spec["labels"].values()}
    assert ("empty",) in labels and ("animal",) in labels


def test_deterministic_same_seed():
    meta = _meta([(f"camera_trap_images/f{i}/dan/IMG_{i}.JPG", ["dan"])
                  for i in range(50)])
    a = build_spec(meta)
    b = build_spec(meta)
    assert a["files"] == b["files"] and a["seed"] == SEED


def test_quota_capped_at_pool():
    meta = _meta([("camera_trap_images/v/vehicle/IMG_%d.JPG" % i,
                   ["vehicle"]) for i in range(3)])
    spec = build_spec(meta)
    assert spec["strata"]["vehicle"] == 3   # take all available, no error
