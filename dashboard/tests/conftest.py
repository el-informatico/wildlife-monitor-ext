"""Offline test bootstrap + the shared fixture repo and live server.

The fixture repo is a miniature of the real artifact tree (manifest,
union batches/records, agent traces, G4 comparison, failure catalog,
two tiny real JPEGs under ``data/``) built under a tmp dir, so endpoint
tests exercise the real ``Store``/handler code with no dependence on
the big artifacts and no model file (the live detector then degrades
to ``live=unavailable`` — itself a tested property).
"""

from __future__ import annotations

import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pick_host() -> str:
    """First loopback address that can actually be connected to.

    Some WSL2 mirrored-networking setups silently DROP IPv4-loopback
    connects (bind works, SYN never returns) while ``::1`` is fine —
    probe both instead of assuming 127.0.0.1.
    """
    for host, fam in (("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)):
        srv = socket.socket(fam, socket.SOCK_STREAM)
        try:
            srv.bind((host, 0))
            srv.listen(1)
            cli = socket.socket(fam, socket.SOCK_STREAM)
            cli.settimeout(1.0)
            try:
                cli.connect((host, srv.getsockname()[1]))
                return host
            except OSError:
                pass
            finally:
                cli.close()
        except OSError:
            pass
        finally:
            srv.close()
    pytest.skip("no connectable loopback address (tried 127.0.0.1, ::1)")


def _jpeg(path: Path, w: int = 48, h: int = 32, seed: int = 0) -> bytes:
    """Write a real decodable JPEG with per-seed content; return bytes."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (h, w, 3), dtype="uint8"))
    assert ok
    path.write_bytes(buf.tobytes())
    return buf.tobytes()


class _FakeDets:
    def __init__(self, items):
        self._items = items

    def to_json(self):
        return self._items


class FakeDetector:
    """Duck-typed MDv6OpenCV returning fixed pixel-space detections.

    Used by test_blur (live_person_boxes) and test_server (injecting a
    live pass into the pool without loading the real model).
    """

    def __init__(self, detections):
        self._detections = detections

    def detect(self, bgr, conf_threshold=0.10):
        return _FakeDets(self._detections), {}, 0.0


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    """Miniature artifact tree the Store and server run against."""
    root = tmp_path_factory.mktemp("dashrepo")

    import pyarrow as pa
    import pyarrow.parquet as pq

    proc = root / "processed"
    proc.mkdir()

    # manifest: 4 images, 3 cameras, one row whose rel_path escapes data/
    pq.write_table(pa.table({
        "image_id": ["aw-0001", "aw-0002", "mo-0001", "bad-0001"],
        "source_dataset": ["african_wildlife", "african_wildlife",
                           "missouri", "missouri"],
        "rel_path": ["aw_cam_a/0001.jpg", "aw_cam_a/0002.jpg",
                     "mo_cam_b/0001.jpg", "../../escape.jpg"],
        "location_id": ["cam-A", "cam-A", "cam-B", "cam-C"],
        "camera_ts": ["2026-03-01T10:00:00", "2026-03-02T10:00:00",
                      "2026-03-01T11:00:00", "2026-03-01T12:00:00"],
        "zone_restricted": [False, True, False, False],
    }), proc / "manifest.parquet")

    # union batches: one person det + one animal det on aw-0001 only
    (proc / "e2e_union_v0_batches.jsonl").write_text("\n".join([
        '{"source": {"image_id": "aw-0001", "width": 48, "height": 32},'
        ' "detections": [{"bbox": [0.2, 0.3, 0.6, 0.9], "label": "person",'
        ' "confidence": 0.88}, {"bbox": [0.1, 0.1, 0.4, 0.5],'
        ' "label": "animal", "confidence": 0.7}]}',
        '{"source": {"image_id": "mo-0001", "width": 48, "height": 32},'
        ' "detections": [{"bbox": [0.1, 0.1, 0.5, 0.6], "label": "animal",'
        ' "confidence": 0.9}]}',
    ]) + "\n")

    # union records: species counts + a non-animal row that must be filtered
    pq.write_table(pa.table({
        "image_id": ["aw-0001", "aw-0001", "aw-0002", "mo-0001", "aw-0002"],
        "label": ["animal", "animal", "animal", "animal", "empty"],
        "species": ["elephant", "zebra", "zebra", "elephant", ""],
    }), proc / "e2e_union_v0_records.parquet")

    # S4 agent traces: two queue actions + one non-queue action
    (proc / "s4_corpus_traces_agent.jsonl").write_text("\n".join([
        '{"event_id": "mo-0001", "camera_id": "cam-B",'
        ' "captured_at": "2026-03-01T11:00:00", "action":'
        ' {"type": "approval_queue", "reason": "low light", "rationale":'
        ' ["r1", "r2", "r3", "r4", "r5"]}}',
        '{"event_id": "aw-0001", "camera_id": "cam-A",'
        ' "captured_at": "2026-03-01T10:00:00", "action":'
        ' {"type": "approval_queue", "reason": "person in open zone",'
        ' "rationale": ["p1"]}}',
        '{"event_id": "aw-0002", "camera_id": "cam-A",'
        ' "captured_at": "2026-03-02T10:00:00", "action":'
        ' {"type": "auto_clear", "reason": "empty frame"}}',
    ]) + "\n")

    crossref = root / "agent" / "artifacts" / "corpus_comparison.json"
    crossref.parent.mkdir(parents=True)
    crossref.write_text(
        '{"person_events": [{"event_id": "aw-0001", "baseline":'
        ' {"action": "escalate"}}], "escalation_load":'
        ' {"agent_person_events_queued": 1, "baseline_escalates": 1}}')

    # G3 failure catalog with one case
    gallery = root / "eval" / "failures" / "images"
    gallery.mkdir(parents=True)
    _jpeg(gallery / "fn-tiny.jpg", seed=7)
    (root / "eval" / "failures" / "CATALOG.md").write_text(
        "# Failure catalog\n\n"
        "## 1. Tiny animal missed\n\n"
        "![frame]( images/fn-tiny.jpg )\n\n"
        "**Key** — small box at 0.11 conf under the deploy floor.\n")

    # data frames referenced by the manifest
    _jpeg(root / "data" / "aw_cam_a" / "0001.jpg", seed=1)
    _jpeg(root / "data" / "aw_cam_a" / "0002.jpg", seed=2)
    _jpeg(root / "data" / "mo_cam_b" / "0001.jpg", seed=3)
    return root


@pytest.fixture(scope="module")
def store(repo):
    from dashboard.store import Store

    return Store(repo)


@pytest.fixture(scope="module")
def server(repo):
    """Live HTTP server on an ephemeral port; yields (base_url, httpd)."""
    from dashboard.server import serve

    host = pick_host()
    httpd = serve(host, 0, repo_root=repo)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    base = (f"http://[{host}]:{port}" if ":" in host
            else f"http://{host}:{port}")
    yield base, httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def get(server):
    """GET helper flattening HTTPError into (status, headers, body)."""
    base_url, _httpd = server

    def _get(path: str):
        try:
            r = urllib.request.urlopen(base_url + path, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        return (r.status, {k.lower(): v for k, v in r.headers.items()},
                r.read())

    return _get
