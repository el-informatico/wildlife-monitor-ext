"""Unit tests for the blur primitives (dashboard/blur.py).

Geometry of the head band, destructiveness of the region blur, and the
live-pass box extraction against a duck-typed detector — no server, no
model, no files.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import FakeDetector  # noqa: E402 — same-dir test helper
from dashboard import blur  # noqa: E402


# ---------------------------------------------------------------------------
# head-band geometry
#

def test_head_band_centered_box():
    # box (0.25, 0.2, 0.75, 0.8) on 100x200: width 0.5 → ±0.03 widened,
    # height band = top 28% of 0.6 → y2 = 0.368
    assert blur.head_band([0.25, 0.2, 0.75, 0.8], 100, 200) == (22, 40, 79, 74)


def test_head_band_clamps_outside_frame():
    # box partially outside the frame clamps to 0 on the top-left
    x1, y1, x2, y2 = blur.head_band([-0.1, -0.1, 0.5, 0.5], 100, 100)
    assert (x1, y1) == (0, 0)
    assert x2 > 0 and y2 > 0


def test_head_band_degenerate_box_still_nonempty():
    # zero-width box must still produce a >=1px band, never an empty slice
    x1, y1, x2, y2 = blur.head_band([0.5, 0.5, 0.5, 0.9], 100, 100)
    assert x2 >= x1 + 1 and y2 >= y1 + 1


# ---------------------------------------------------------------------------
# region blur
#

def test_blur_regions_empty_is_identity():
    import numpy as np

    img = np.arange(300, dtype="uint8").reshape(10, 10, 3) % 255
    before = img.copy()
    assert blur.blur_regions(img, []) == 0
    assert (img == before).all()


def test_blur_regions_destructive_and_counted():
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 40, 3), dtype="uint8")
    before = img.copy()
    assert blur.blur_regions(img, [(0, 0, 20, 20), (25, 25, 30, 30)]) == 2
    # structure inside the band collapses (flat regions are trivially
    # invariant under their own Gaussian — noise is not)
    band, orig = (img[0:20, 0:20].astype("float32"),
                  before[0:20, 0:20].astype("float32"))
    assert not np.array_equal(band, orig)
    assert band.std() < 0.5 * orig.std()
    # and the untouched corner is untouched
    assert (img[30:40, 30:40] == before[30:40, 30:40]).all()


def test_blur_regions_clamps_oversized_band():
    import numpy as np

    img = np.full((10, 10, 3), 255, dtype="uint8")
    # band larger than the frame must not raise on slicing
    assert blur.blur_regions(img, [(-5, -5, 50, 50)]) == 1


# ---------------------------------------------------------------------------
# live pass
#

def test_live_person_boxes_filters_and_normalizes():
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype="uint8")
    det = FakeDetector([
        {"bbox": [100, 40, 160, 90], "conf": 0.9, "cls": 1},   # person
        {"bbox": [10, 10, 50, 50], "conf": 0.8, "cls": 0},     # animal
        {"bbox": [0, 0, 30, 30], "conf": 0.7, "cls": 2},       # vehicle
    ])
    boxes = blur.live_person_boxes(frame, det)
    assert boxes == [[100 / 200, 40 / 100, 160 / 200, 90 / 100]]


def test_live_person_boxes_no_person_is_empty():
    import numpy as np

    frame = np.zeros((64, 64, 3), dtype="uint8")
    det = FakeDetector([{"bbox": [1, 1, 10, 10], "conf": 0.9, "cls": 0}])
    assert blur.live_person_boxes(frame, det) == []
