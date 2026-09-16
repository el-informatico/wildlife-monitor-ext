"""Endpoint tests: a real ThreadingHTTPServer on an ephemeral port.

Covers every route, the blur enforcement end-to-end over HTTP (stored
boxes; injected live detector; byte-identity when no person evidence),
the detector-degrade path (no model in the fixture repo → the header
says ``live=unavailable``, stored-box blur still applies), and the 404
guards (unknown ids, escaping paths, bogus kinds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import FakeDetector  # noqa: E402 — same-dir test helper


# ---------------------------------------------------------------------------
# page + JSON routes
#

def test_index_html(get):
    status, headers, body = get("/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert "ops dashboard" in body.decode()


def test_api_overview(get):
    status, _headers, body = get("/api/overview")
    assert status == 200
    o = json.loads(body)
    assert o["cameras"] == 3
    assert o["manifest_images"] == 4
    assert o["queue_size"] == 2
    assert o["queue_person"] == 1
    assert o["failure_cases"] == 1


def test_api_cameras(get):
    status, _headers, body = get("/api/cameras")
    assert status == 200
    cams = json.loads(body)
    assert cams["n_cameras"] == 3
    assert len(cams["cameras"]) == 3
    assert "NO GPS" in cams["layout"]


def test_api_species(get):
    status, _headers, body = get("/api/species")
    assert status == 200
    assert json.loads(body)["counts"] == {"elephant": 2, "zebra": 2}


def test_api_queue(get):
    status, _headers, body = get("/api/queue")
    assert status == 200
    q = json.loads(body)
    assert q["n"] == 2
    assert q["items"][0]["event_id"] == "aw-0001"
    assert q["items"][0]["face_blur_required"] is True


def test_api_failures(get):
    status, _headers, body = get("/api/failures")
    assert status == 200
    f = json.loads(body)
    assert f["n"] == 1
    assert f["cases"][0]["image"] == "fn-tiny.jpg"


# ---------------------------------------------------------------------------
# the image path — blur enforcement
#

def test_person_frame_blur_applied_and_bytes_differ(get, repo):
    status, headers, body = get("/image?src=frame:aw-0001")
    assert status == 200
    assert headers["content-type"] == "image/jpeg"
    assert headers["x-face-blur"].startswith("applied=")
    on_disk = (repo / "data" / "aw_cam_a" / "0001.jpg").read_bytes()
    assert body != on_disk  # re-encoded, destructively blurred


def test_no_person_frame_served_verbatim(get, repo):
    status, headers, body = get("/image?src=frame:mo-0001")
    assert status == 200
    assert headers["x-face-blur"].startswith("none")
    on_disk = (repo / "data" / "mo_cam_b" / "0001.jpg").read_bytes()
    assert body == on_disk  # byte-identity: no re-encode


def test_live_pass_blurs_frame_with_no_stored_person(get, server, repo):
    # the safety net: a frame whose stored pass found nothing still gets
    # blurred when the LIVE detector sees a person
    _base, httpd = server
    pool = httpd.RequestHandlerClass.detectors
    # a person box covering the left half of the 48x32 fixture frame
    fake = FakeDetector([{"bbox": [4, 4, 24, 28], "conf": 0.9, "cls": 1}])
    saved = (pool._detector, pool._error)
    pool._detector, pool._error = fake, None
    try:
        status, headers, body = get("/image?src=frame:mo-0001")
        assert status == 200
        assert headers["x-face-blur"].startswith("applied=1;live")
        assert body != (repo / "data" / "mo_cam_b" / "0001.jpg").read_bytes()
    finally:
        pool._detector, pool._error = saved


def test_detector_degrades_to_live_unavailable(get, server):
    # the fixture repo has no models/ → the real detector build fails;
    # the server must say so in the header and still serve stored blur
    _base, httpd = server
    status, headers, _body = get("/image?src=frame:aw-0001")
    assert status == 200
    assert "live=unavailable" in headers["x-face-blur"]
    assert "unavailable" in httpd.RequestHandlerClass.detectors.status


def test_failure_gallery_image_served(get, repo):
    status, headers, body = get("/image?src=failure:fn-tiny.jpg")
    assert status == 200
    assert headers["content-type"] == "image/jpeg"
    assert "x-face-blur" in headers  # every image states its outcome


# ---------------------------------------------------------------------------
# 404 guards
#

def test_unknown_frame_404(get):
    status, _headers, body = get("/image?src=frame:does-not-exist")
    assert status == 404
    assert "unknown image" in json.loads(body)["error"]


def test_frame_path_escape_404(get):
    status, _headers, body = get("/image?src=frame:bad-0001")
    assert status == 404
    assert "escapes" in json.loads(body)["error"]


def test_failure_traversal_404(get):
    status, _headers, body = get(
        "/image?src=failure:../../../../../etc/passwd")
    assert status == 404


def test_unknown_spec_kind_404(get):
    status, _headers, body = get("/image?src=bogus:x.jpg")
    assert status == 404
    assert "unknown image spec kind" in json.loads(body)["error"]


def test_missing_src_404(get):
    status, _headers, body = get("/image")
    assert status == 404
    assert "missing" in json.loads(body)["error"]


def test_unknown_route_404(get):
    status, _headers, body = get("/etc/passwd")
    assert status == 404
    assert "no route" in json.loads(body)["error"]
