"""Unit tests for the read-only Store (dashboard/store.py).

All against the miniature fixture repo from conftest — pyarrow fixtures
for the parquet artifacts, JSONL traces, a one-case catalog — plus the
image-resolution guards (unknown ids, path escapes, bogus kinds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# cameras
#

def test_cameras_count_and_schematic_disclaimer(store):
    cams = store.cameras
    assert cams["n_cameras"] == 3
    assert "NO GPS" in cams["layout"]
    assert "SYNTHETIC" in cams["clock"]
    by_id = {c["camera_id"]: c for c in cams["cameras"]}
    assert by_id["cam-A"]["images"] == 2
    assert by_id["cam-A"]["zone_restricted"] is True  # aw-0002 restricted
    assert by_id["cam-B"]["zone_restricted"] is False
    # every dot position inside the unit square
    assert all(0 <= c["x"] <= 1 and 0 <= c["y"] <= 1
               for c in cams["cameras"])


# ---------------------------------------------------------------------------
# species
#

def test_species_counts_animals_only(store):
    sp = store.species
    assert sp["counts"] == {"elephant": 2, "zebra": 2}  # 'empty' row filtered
    assert sp["n_animal_records"] == 4
    assert sp["n_images_with_animal"] == 3


# ---------------------------------------------------------------------------
# approval queue
#

def test_queue_person_first_with_crossref(store):
    q = store.queue
    assert q["n"] == 2  # auto_clear action is not queued
    first = q["items"][0]
    assert first["event_id"] == "aw-0001"  # person event sorted first
    assert first["person_evidence"] is True
    assert first["face_blur_required"] is True
    assert first["person_max_conf"] == 0.88
    assert first["baseline_action"] == "escalate"  # G4 crossref attached
    assert first["image_url"] == "/image?src=frame:aw-0001"
    assert "committed G4 comparison" in q["baseline_note"]


def test_queue_non_person_event(store):
    other = store.queue["items"][1]
    assert other["event_id"] == "mo-0001"
    assert other["person_evidence"] is False
    assert other["face_blur_required"] is False
    assert other["person_max_conf"] is None
    # rationale truncated to 3 entries even though the trace has 5
    assert other["rationale"] == ["r1", "r2", "r3"]


# ---------------------------------------------------------------------------
# failure gallery
#

def test_failures_catalog_parsed(store):
    f = store.failures
    assert f["n"] == 1
    case = f["cases"][0]
    assert case["case"] == 1
    assert case["title"] == "Tiny animal missed"
    assert case["image"] == "fn-tiny.jpg"
    assert "0.11 conf" in case["key"]
    assert case["image_url"] == "/image?src=failure:fn-tiny.jpg"


# ---------------------------------------------------------------------------
# overview
#

def test_overview_counts(store):
    o = store.overview()
    assert o["cameras"] == 3
    assert o["manifest_images"] == 4
    assert o["queue_size"] == 2
    assert o["queue_person"] == 1
    assert o["failure_cases"] == 1


# ---------------------------------------------------------------------------
# image resolution — the single path the server serves images through
#

def test_resolve_frame_returns_path_and_stored_boxes(store, repo):
    path, boxes = store.resolve_image("frame:aw-0001")
    assert path == (repo / "data" / "aw_cam_a" / "0001.jpg").resolve()
    assert boxes == [[0.2, 0.3, 0.6, 0.9, 0.88]]


def test_resolve_frame_without_person_returns_no_boxes(store):
    _path, boxes = store.resolve_image("frame:mo-0001")
    assert boxes == []  # the live MDv6 pass at serve time covers gaps


def test_resolve_frame_unknown_id(store):
    with pytest.raises(KeyError):
        store.resolve_image("frame:does-not-exist")


def test_resolve_frame_rel_path_escape_rejected(store):
    # manifest row whose rel_path leaves data/ → ValueError (→ 404)
    with pytest.raises(ValueError, match="escapes"):
        store.resolve_image("frame:bad-0001")


def test_resolve_failure_strips_path_components(store, repo):
    # any path components in the failure spec collapse to the basename
    path, boxes = store.resolve_image("failure:sub/dir/fn-tiny.jpg")
    assert path == (repo / "eval" / "failures" / "images"
                    / "fn-tiny.jpg").resolve()
    assert boxes == []


def test_resolve_failure_unknown_file(store):
    with pytest.raises(KeyError):
        store.resolve_image("failure:nope.jpg")


def test_resolve_unknown_kind_rejected(store):
    with pytest.raises(ValueError, match="unknown image spec kind"):
        store.resolve_image("bogus:x.jpg")
