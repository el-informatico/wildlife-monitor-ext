"""Offline tests for scripts/g1_plan.py — the batch driver's pure planning.

The planning module and handler/motion_app.py must agree on S3 layouts and
DynamoDB addresses; several tests here cross-check the two implementations
directly so they cannot drift apart.
"""

from scripts import g1_plan
from handler import motion_app


def make_rows():
    """Two cameras, deliberately shuffled order, odd timestamps."""
    base = [
        # (camera, image_id, minute offset, rel file)
        ("african_wildlife-L01", "aw-00010", 30, "b.jpg"),
        ("african_wildlife-L00", "aw-00001", 20, "z.jpg"),
        ("african_wildlife-L01", "aw-00011", 10, "a.jpg"),
        ("african_wildlife-L00", "aw-00000", 0, "y.jpg"),
        ("african_wildlife-L00", "aw-00002", 50, "x.jpg"),
    ]
    return [
        {
            "image_id": image_id,
            "source_dataset": "african_wildlife",
            "rel_path": f"african-wildlife/images/test/{fname}",
            "species": "zebra",
            "location_id": camera,
            "camera_ts": f"2026-03-02T11:{offset:02d}:00",
            "is_night": offset >= 50,
        }
        for camera, image_id, offset, fname in base
    ]


def test_select_records_filters_and_slices():
    rows = make_rows()
    assert len(g1_plan.select_records(rows)) == 5
    assert all(
        r["source_dataset"] == "missouri"
        for r in g1_plan.select_records(rows, dataset="missouri")
    )
    only_l01 = g1_plan.select_records(
        rows, cameras=["african_wildlife-L01"]
    )
    assert [r["image_id"] for r in only_l01] == ["aw-00010", "aw-00011"]
    assert len(g1_plan.select_records(rows, limit=2)) == 2
    # limit=0 / None mean "no limit"
    assert len(g1_plan.select_records(rows, limit=0)) == 5


def test_build_plan_groups_sorts_and_names():
    jobs = g1_plan.build_plan(make_rows())
    assert [j["camera_id"] for j in jobs] == [
        "african_wildlife-L00",
        "african_wildlife-L01",
    ]
    first = jobs[0]
    assert first["job_key"] == "motion-jobs/african_wildlife/african_wildlife-L00.json"
    # frames in capture order, media keys derived from rel_path
    assert [f["image_id"] for f in first["frames"]] == [
        "aw-00000", "aw-00001", "aw-00002"
    ]
    assert first["frames"][0]["key"] == "raw/african-wildlife/images/test/y.jpg"
    assert first["media"][0] == {
        "key": "raw/african-wildlife/images/test/y.jpg",
        "rel_path": "african-wildlife/images/test/y.jpg",
    }
    assert first["dataset"] == "african_wildlife"


def test_build_plan_splits_long_cameras_with_sequence_suffix():
    jobs = g1_plan.build_plan(make_rows(), max_frames_per_job=2)
    l00 = [j for j in jobs if j["camera_id"] == "african_wildlife-L00"]
    assert [j["job_key"] for j in l00] == [
        "motion-jobs/african_wildlife/african_wildlife-L00-0000.json",
        "motion-jobs/african_wildlife/african_wildlife-L00-0001.json",
    ]
    assert [len(j["frames"]) for j in l00] == [2, 1]
    # a camera under the cap keeps the unsuffixed name
    l01 = [j for j in jobs if j["camera_id"] == "african_wildlife-L01"]
    assert l01[0]["job_key"].endswith("african_wildlife-L01.json")


def test_timestamp_and_id_parity_with_handler():
    for value in [
        "2026-03-02T11:00:00",             # naive (manifest form) -> UTC
        "2026-03-02T11:00:00Z",
        "2026-03-02T11:00:00.250Z",
        "2026-03-02T13:00:00+02:00",       # offset shifts to UTC
    ]:
        assert g1_plan.canon_ts(value) == motion_app._canon_ts(value)
    assert g1_plan.canon_ts("2026-03-02T11:00:00") == "2026-03-02T11:00:00.000Z"
    assert (
        g1_plan.canon_ts("2026-03-02T13:00:00+02:00")
        == "2026-03-02T11:00:00.000Z"
    )
    key = "raw/african-wildlife/images/test/1 (103).jpg"
    assert g1_plan.frame_id(key) == motion_app._frame_id(key)
    assert (
        g1_plan.idempotency_pk("b", "k", "v1", None)
        == motion_app._idempotency_key("b", "k", "v1", None)
    )
    # versionId wins over etag; both empty is its own key
    assert g1_plan.idempotency_pk("b", "k", None, "abc") == g1_plan.idempotency_pk(
        "b", "k", "abc", None
    )


def test_item_address_matches_handler_row():
    camera = "african_wildlife-L00"
    frame = {
        "key": "raw/african-wildlife/images/test/y.jpg",
        "image_id": "aw-00000",
        "captured_at": "2026-03-02T11:00:00",
        "is_night": False,
    }
    item = motion_app._frame_item(
        camera, frame, fg_ratio=0.5, gate="passed",
        dims=(800, 600), cv2_version="5.0.0", now="2026-09-16T00:00:00.000Z",
    )
    address = g1_plan.item_address(camera, frame["key"], frame["captured_at"])
    assert (item["pk"], item["sk"]) == (address["pk"], address["sk"])


def test_job_documents_satisfy_handler_validation():
    for job in g1_plan.build_plan(make_rows()):
        cleaned = motion_app.validate_job(g1_plan.job_document(job))
        assert cleaned["camera_id"] == job["camera_id"]
        assert len(cleaned["frames"]) == len(job["frames"])


def test_expected_addresses_unique_and_complete():
    jobs = g1_plan.build_plan(make_rows())
    addresses = g1_plan.expected_addresses(jobs)
    assert len(addresses) == 5
    assert len({(a["pk"], a["sk"]) for a in addresses}) == 5
    assert all(a["source_key"].startswith("raw/") for a in addresses)


def test_gate_for_mirrors_benchmark_rule():
    # warmup covers the first two frames regardless of ratio
    assert motion_app.gate_for(0, 1.0) == "warmup"
    assert motion_app.gate_for(1, 0.0) == "warmup"
    # threshold is strictly greater than 0.02
    assert motion_app.gate_for(2, 0.02) == "filtered"
    assert motion_app.gate_for(2, 0.0200001) == "passed"
    assert motion_app.gate_for(7, 0.0) == "filtered"
