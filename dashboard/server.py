"""The dashboard HTTP server: stdlib-only core, one image path, blur always.

``ThreadingHTTPServer`` + ``BaseHTTPRequestHandler`` — zero web
dependencies, matching the repo's stdlib-first convention. Read-only:
every route is a GET that reads files; nothing the server can do
mutates the artifacts.

Routes:

- ``/``                    — the single HTML page (map, species counts,
  approval queue, failure gallery; inline CSS/JS, fetches the APIs);
- ``/api/overview`` …      — JSON views over ``Store``;
- ``/image?src=<spec>``    — the ONLY image route. Every byte served
  passes through ``_serve_image``: stored person boxes ∪ a live MDv6
  person pass (floor 0.10) → destructive head-band blur when any
  person evidence exists; ``X-Face-Blur`` states the outcome. Images
  with no person evidence are served verbatim (no re-encode).

The live detector is built lazily on the first image request (model
load ~0.5 s) and shared under a lock — cv2.dnn forward is not
documented thread-safe. If the model or the pipeline venv pieces are
missing the server still runs, serving stored-boxes-only blur and
saying so in the header (``live=unavailable``) — it never serves an
unblurred person frame because of a missing model without stating it.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import blur
from .store import REPO_ROOT, Store

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


class DetectorPool:
    """Lazily-built shared MDv6 detector + the lock serializing forward."""

    def __init__(self, repo_root=REPO_ROOT) -> None:
        self._root = repo_root
        self._detector: Any = None
        self._error: str | None = None
        self.lock = threading.Lock()

    @property
    def status(self) -> str:
        if self._detector is not None:
            return "live"
        if self._error is not None:
            return f"unavailable ({self._error})"
        return "pending (built on first image request)"

    def get(self) -> Any:
        """Build once on first use; None (with reason) if impossible."""
        if self._detector is not None or self._error is not None:
            return self._detector
        try:
            from pipeline import e2e

            mdv6 = e2e.import_wildlife("wildlife_monitor.detect.mdv6")
            onnx = self._root / "models" / "MDV6-yolov10-c.onnx"
            self._detector = mdv6.MDv6OpenCV(onnx, engine="new")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._error = str(exc)[:200]
        return self._detector


class DashboardHandler(BaseHTTPRequestHandler):
    store: Store
    detectors: DetectorPool
    server_version = "wfm-dashboard/1"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        # one concise line per request; the default writes to stderr
        print(f"[dashboard] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, (json.dumps(payload, indent=1) + "\n").encode(),
                   "application/json")

    def _404(self, msg: str) -> None:
        self._json({"error": msg}, 404)

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        route = urlparse(self.path)
        path = route.path
        try:
            if path == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/overview":
                self._json(self.store.overview())
            elif path == "/api/cameras":
                self._json(self.store.cameras)
            elif path == "/api/species":
                self._json(self.store.species)
            elif path == "/api/queue":
                self._json(self.store.queue)
            elif path == "/api/failures":
                self._json(self.store.failures)
            elif path == "/image":
                self._serve_image(parse_qs(route.query).get("src", [""])[0])
            else:
                self._404(f"no route: {path}")
        except KeyError as exc:
            self._404(f"unknown image: {exc.args[0]}")
        except ValueError as exc:
            self._404(f"rejected: {exc}")

    # -- the single image path ------------------------------------------------

    def _serve_image(self, spec: str) -> None:
        if not spec:
            self._404("missing ?src=")
            return
        path, stored_boxes = self.store.resolve_image(spec)
        raw = path.read_bytes()

        import cv2
        import numpy as np

        bgr = cv2.imdecode(np.frombuffer(raw, dtype="uint8"),
                           cv2.IMREAD_COLOR)
        if bgr is None:
            self._404(f"undecodable image: {spec}")
            return

        person_norm = [b[:4] for b in stored_boxes]
        detector = self.detectors.get()
        if detector is not None:
            with self.detectors.lock:
                live = blur.live_person_boxes(bgr, detector)
            person_norm = person_norm + live
            live_note = "live"
        else:
            live_note = f"live=unavailable;{self.detectors.status}"

        h, w = bgr.shape[:2]
        bands = [blur.head_band(b, w, h) for b in person_norm]
        n = blur.blur_regions(bgr, bands)

        if n:
            ok, buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:  # pragma: no cover — imencode on valid arrays
                self._404(f"encode failed: {spec}")
                return
            data = buf.tobytes()
            header = {"X-Face-Blur": f"applied={n};{live_note}"}
        else:
            data = raw  # verbatim — byte-identity for non-person images
            header = {"X-Face-Blur": f"none;{live_note}"}
        self._send(200, data, "image/jpeg", header)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          repo_root=REPO_ROOT) -> ThreadingHTTPServer:
    """Build the server (store + detector pool bound) without serving."""
    handler = type("BoundHandler", (DashboardHandler,), {})
    handler.store = Store(repo_root)
    handler.detectors = DetectorPool(repo_root)
    cls = ThreadingHTTPServer
    if ":" in host:
        # IPv6 literal (e.g. ``::1``): some WSL2 mirrored-networking
        # setups drop IPv4-loopback connects while ::1 works fine.
        cls = type("ThreadingHTTPServer6", (ThreadingHTTPServer,),
                   {"address_family": socket.AF_INET6})
    httpd = cls((host, port), handler)
    httpd.daemon_threads = True
    return httpd


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wildlife-monitor — ops dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto;
         max-width: 1100px; padding: 1rem; }
  h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
  .muted { opacity: .7; font-size: .85rem; }
  .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
  .card { border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
          border-radius: 8px; padding: .6rem 1rem; min-width: 9rem; }
  .card b { font-size: 1.4rem; display: block; }
  #map { position: relative; height: 340px; border: 1px dashed
         color-mix(in srgb, currentColor 30%, transparent); border-radius: 8px; }
  .cam { position: absolute; transform: translate(-50%,-50%); border-radius: 50%;
         width: 14px; height: 14px; cursor: pointer; }
  .cam.aw { background: #2a7; } .cam.mo { background: #c66; }
  .cam.zone { box-shadow: 0 0 0 3px #dc3; }
  .cam .tip { display: none; position: absolute; left: 16px; top: -8px;
              white-space: nowrap; background: #000c; color: #fff;
              border-radius: 4px; padding: 2px 6px; font-size: .75rem; }
  .cam:hover .tip { display: block; }
  table { border-collapse: collapse; width: 100%; }
  td, th { padding: .25rem .5rem; text-align: left;
           border-bottom: 1px solid
           color-mix(in srgb, currentColor 15%, transparent); }
  img { max-width: 100%; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill,
          minmax(240px, 1fr)); gap: 1rem; }
  figure { margin: 0; } figcaption { font-size: .8rem; }
  .tag { font-size: .75rem; border-radius: 4px; padding: 0 4px;
         border: 1px solid currentColor; }
</style>
</head>
<body>
<h1>wildlife-monitor — ops dashboard</h1>
<p class="muted">Read-only views over the local pipeline artifacts
(manifest · e2e union records · S4 agent traces · G3 failure catalog).
Every image is served through the blur path — see the
<code>X-Face-Blur</code> header on any image.</p>

<h2>Overview</h2>
<div class="cards" id="overview"></div>

<h2>Camera map</h2>
<p class="muted" id="map-note"></p>
<div id="map"></div>
<div class="muted"><span class="tag" style="color:#2a7">african_wildlife</span>
<span class="tag" style="color:#c66">missouri</span> yellow ring = restricted
zone · hover a camera for stats</div>

<h2>Species counts</h2>
<table id="species"><thead><tr><th>species</th><th>deployed detections</th>
</tr></thead><tbody></tbody></table>
<p class="muted" id="species-note"></p>

<h2>Approval queue</h2>
<p class="muted" id="queue-note"></p>
<table id="queue"><thead><tr><th>event</th><th>camera</th><th>captured</th>
<th>person</th><th>reason</th><th>baseline</th><th></th></tr></thead>
<tbody></tbody></table>

<h2>Failure gallery (G3 catalog)</h2>
<div class="grid" id="failures"></div>

<script>
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

fetch("/api/overview").then(r => r.json()).then(o => {
  $("#overview").innerHTML = [
    ["cameras", o.cameras], ["manifest images", o.manifest_images],
    ["queue size", o.queue_size], ["queue w/ person", o.queue_person],
    ["failure cases", o.failure_cases],
  ].map(([k, v]) => `<div class="card"><b>${v}</b>${k}</div>`).join("");
});

fetch("/api/cameras").then(r => r.json()).then(c => {
  $("#map-note").textContent = c.layout + " — " + c.clock + ".";
  $("#map").innerHTML = c.cameras.map(cam =>
    `<div class="cam ${cam.dataset === "african_wildlife" ? "aw" : "mo"}"
      ${cam.zone_restricted ? "zone" : ""}
      style="left:${cam.x * 100}%;top:${cam.y * 100}%">
      <span class="tip">${esc(cam.camera_id)} · ${cam.images} imgs ·
      ${cam.fp_sample ?? "?"} FP (sample) ·
      ${cam.person_dets_sample ?? "?"} person dets (sample)</span>
    </div>`).join("");
});

fetch("/api/species").then(r => r.json()).then(s => {
  $("#species tbody").innerHTML = Object.entries(s.counts).map(([k, v]) =>
    `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("");
  $("#species-note").textContent = s.source + ".";
});

fetch("/api/queue").then(r => r.json()).then(q => {
  $("#queue-note").textContent = (q.baseline_note ?? "") +
    ` — ${q.n} events in the queue; person evidence always carries ` +
    `face_blur_required.`;
  $("#queue tbody").innerHTML = q.items.map(it =>
    `<tr><td>${esc(it.event_id)}</td><td>${esc(it.camera_id)}</td>
     <td>${esc(it.captured_at)}</td>
     <td>${it.person_evidence ? `yes (max ${it.person_max_conf})` : "no"}</td>
     <td>${esc(it.reason)}</td><td>${esc(it.baseline_action ?? "—")}</td>
     <td><a href="${it.image_url}" target="_blank">image</a></td></tr>`
  ).join("") || `<tr><td colspan="7">queue empty</td></tr>`;
});

fetch("/api/failures").then(r => r.json()).then(f => {
  $("#failures").innerHTML = f.cases.map(c =>
    `<figure><a href="${c.image_url}" target="_blank">
     <img src="${c.image_url}" loading="lazy" alt="${esc(c.title)}"></a>
     <figcaption><b>Case ${c.case}: ${esc(c.title)}</b><br>
     ${esc(c.key ?? "")}</figcaption></figure>`).join("");
});
</script>
</body>
</html>
"""
