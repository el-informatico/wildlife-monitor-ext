# dashboard — read-only ops endpoint (S5)

A stdlib-only HTTP server (no FastAPI/flask — the repo stays
dependency-light; cv2/numpy are used only on the image path) serving
the pipeline's local artifacts. Every route is a GET; nothing the
server does mutates the artifacts.

## Run

    ../wildlife-monitor/.venv/bin/python -m dashboard            # 127.0.0.1:8787
    ../wildlife-monitor/.venv/bin/python -m dashboard --port 9000
    ../wildlife-monitor/.venv/bin/python -m dashboard --host ::1 # see note

WSL2 note (this machine): mirrored networking can silently DROP
IPv4-loopback connects (bind works, `curl 127.0.0.1` hangs). `::1`
works — the server accepts IPv6 host literals for that reason. A
Windows-side browser at `http://127.0.0.1:8787` may still work (the
Windows→WSL direction is the one mirrored mode guarantees).

## Routes

| route | what it serves |
|---|---|
| `/` | the single page: camera map, species counts, approval queue, failure gallery |
| `/api/overview` | headline counts |
| `/api/cameras` | per-camera stats (images, first/last `camera_ts`, restricted-zone flag, FP + person-det samples from the S5 final report) + schematic layout |
| `/api/species` | deployed-animal species counts from the e2e union records |
| `/api/queue` | every `approval_queue` action in the S4 agent traces, person evidence attached, cross-referenced with the committed G4 baseline comparison |
| `/api/failures` | the committed G3 failure catalog parsed into gallery entries |
| `/image?src=<spec>` | **the only image route** — `frame:<image_id>` (manifest `rel_path`) or `failure:<file>` (committed gallery) |

## Face blur — always, on the single image path

There is exactly one image-serving path (`DashboardHandler._serve_image`),
and every byte it serves passes the blur policy:

1. stored person boxes from the e2e union batches, **united with** a live
   MDv6 person pass at conf floor 0.10 run at serve time (a frame whose
   stored pass found nothing is still protected if the live pass sees a
   person);
2. each person box contributes a destructive head band — top 28% of the
   box, widened 6% per side. No pixel-level face model exists offline
   (the OpenCV wheel ships no Haar data; `models/` has none), so the
   band is deliberately generous;
3. regions are blurred in place (3× Gaussian k=31) and the frame is
   re-encoded — there is no un-blur;
4. images with zero person evidence are served as the original bytes,
   verbatim (no re-encode).

The `X-Face-Blur` response header always states the outcome:
`applied=<n>;live`, `none;live`, or `…;live=unavailable;<reason>` when
the model could not be built (the server then still applies
stored-box blur rather than failing or serving blind).

## Data sources

`processed/manifest.parquet` · `processed/e2e_union_v0_batches.jsonl`
· `processed/e2e_union_v0_records.parquet`
· `processed/s4_corpus_traces_agent.jsonl`
· `agent/artifacts/corpus_comparison.json`
· `eval/failures/CATALOG.md` + `eval/failures/images/`.

The map is a **schematic** two-cluster grid: no GPS/lat-lon exists in
any local metadata, and `camera_ts` is the ETL synthetic per-camera
clock — both stated in the `/api/cameras` payload itself.

## Tests

    ../wildlife-monitor/.venv/bin/python -m pytest dashboard/tests -q

38 tests: blur geometry/destructiveness, Store aggregation and the
image-resolution guards, and live-HTTP endpoint tests (ephemeral port,
loopback-address probe, injected live detector, byte-identity, 404s).
