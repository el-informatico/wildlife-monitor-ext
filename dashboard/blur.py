"""Destructive face blur for every image the dashboard serves.

Policy (stated, not implied):

- A person-labeled detection contributes a **head band** — the top
  ``HEAD_FRACTION`` of its box, widened by ``WIDEN`` on each side.
  Pixel-level face localization is NOT available offline: this
  OpenCV 5 wheel ships no Haar cascade data (checked —
  ``cv2.data`` absent, no ``haarcascade*.xml`` anywhere in the
  package), and no face model exists in ``models/``. The band is
  deliberately generous for that reason.
- Regions are blurred destructively (repeated strong Gaussian on the
  array slice) — there is no un-blur.
- The serving path unions the STORED person boxes (e2e union batches)
  with a LIVE MDv6 person pass at floor 0.10, so a frame whose stored
  pass found nothing is still protected if the live pass sees a person.
- An image with zero regions is served as the original bytes, verbatim
  (no re-encode) — byte-identity for non-person images is a property,
  and the ``X-Face-Blur`` header always states what happened:

      applied=<n>            n ≥ 1 regions blurred
      none                   no person evidence, bytes untouched
"""

from __future__ import annotations

from typing import Any

#: top fraction of a person box treated as the head region.
HEAD_FRACTION = 0.28
#: horizontal widening of the band, as a fraction of box width per side.
WIDEN = 0.06
#: blur passes × kernel — heavy enough that no structure survives.
BLUR_PASSES = 3
BLUR_KERNEL = 31


def head_band(box_xyxy: list[float], img_w: int, img_h: int
              ) -> tuple[int, int, int, int]:
    """Normalized [x1,y1,x2,y2] person box → clamped pixel head band."""
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    bx1 = (x1 - WIDEN * w) * img_w
    bx2 = (x2 + WIDEN * w) * img_w
    by1 = y1 * img_h
    by2 = min(y1 + HEAD_FRACTION * max(y2 - y1, 0.0), 1.0) * img_h
    x1p, y1p = max(int(bx1), 0), max(int(by1), 0)
    x2p, y2p = min(int(bx2) + 1, img_w), min(int(by2) + 1, img_h)
    return x1p, y1p, max(x2p, x1p + 1), max(y2p, y1p + 1)


def blur_regions(bgr: Any, bands: list[tuple[int, int, int, int]]) -> int:
    """Destructively blur each pixel band in place; return the count done.

    cv2 is imported here (the blur path is the only cv2 consumer of the
    server) so stdlib-only callers stay importable without it.
    """
    if not bands:
        return 0
    import cv2

    h, w = bgr.shape[:2]
    for x1, y1, x2, y2 in bands:
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        region = bgr[y1:y2, x1:x2]
        for _ in range(BLUR_PASSES):
            region = cv2.GaussianBlur(region, (BLUR_KERNEL, BLUR_KERNEL), 0)
        bgr[y1:y2, x1:x2] = region
    return len(bands)


def live_person_boxes(bgr: Any, detector: Any, conf_floor: float = 0.10
                      ) -> list[list[float]]:
    """One live MDv6 pass over the decoded frame → normalized person boxes.

    The detector is the sibling's ``MDv6OpenCV`` (injected — no import
    side effects here); its ``detect`` returns boxes in original-frame
    pixels, normalized here so the caller can union them with the stored
    normalized boxes before a single ``head_band`` pass.
    """
    h, w = bgr.shape[:2]
    dets, _params, _forward_s = detector.detect(bgr, conf_threshold=conf_floor)
    out = []
    for d in dets.to_json():
        if int(d["cls"]) == 1:  # 1 = person (pipeline.e2e.DET_LABELS)
            x1, y1, x2, y2 = d["bbox"]
            out.append([x1 / w, y1 / h, x2 / w, y2 / h])
    return out
