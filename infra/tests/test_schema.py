"""Unit tests for infra/schema.py — the detection-payload contract."""

import pytest

from schema import (
    BBOX_CLAMP_TOLERANCE,
    SchemaError,
    bbox_px_to_norm,
    validate_batch,
    validate_detection,
)


def valid_detection(**overrides):
    det = {
        "label": "animal",
        "species": "zebra",
        "confidence": 0.5,
        "bbox": [0.1, 0.2, 0.8, 0.9],
    }
    det.update(overrides)
    return det


def valid_batch(**overrides):
    doc = {
        "schema_version": 1,
        "source": {
            "key": "raw/cam-01/a.jpg",
            "image_id": "img-1",
            "width": 550,
            "height": 412,
        },
        "model": {"name": "MDV6-yolov10-c"},
        "detections": [valid_detection()],
    }
    doc.update(overrides)
    return doc


# --- batch-level (structural: poison policy) ------------------------------


def test_valid_batch_normalizes():
    batch = validate_batch(valid_batch())
    assert batch.schema_version == 1
    assert batch.source["key"] == "raw/cam-01/a.jpg"
    assert batch.model["name"] == "MDV6-yolov10-c"
    assert len(batch.detections) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(schema_version="1"),
        lambda d: d.pop("source"),
        lambda d: d["source"].pop("key"),
        lambda d: d["source"].update(key="/absolute/path.jpg"),
        lambda d: d["source"].update(key="../escape.jpg"),
        lambda d: d["source"].pop("image_id"),
        lambda d: d["source"].update(width=0),
        lambda d: d["source"].update(height=-3),
        lambda d: d["source"].update(width=550.0),
        lambda d: d["source"].update(is_night="yes"),
        lambda d: d.pop("model"),
        lambda d: d["model"].pop("name"),
        lambda d: d.pop("detections"),
        lambda d: d.update(detections={"not": "a list"}),
    ],
)
def test_structural_violations_raise(mutate):
    doc = valid_batch()
    mutate(doc)
    with pytest.raises(SchemaError):
        validate_batch(doc)


def test_payload_not_a_mapping():
    with pytest.raises(SchemaError):
        validate_batch([1, 2, 3])


def test_empty_detections_is_valid():
    batch = validate_batch(valid_batch(detections=[]))
    assert batch.detections == ()


def test_bad_captured_at_raises():
    doc = valid_batch()
    doc["source"]["captured_at"] = "16/09/2026 10:15"
    with pytest.raises(SchemaError):
        validate_batch(doc)


def test_naive_captured_at_raises():
    doc = valid_batch()
    doc["source"]["captured_at"] = "2026-09-16T10:15:00"  # no timezone
    with pytest.raises(SchemaError):
        validate_batch(doc)


def test_captured_accepts_z_and_offset():
    for ts in ("2026-09-16T10:15:00Z", "2026-09-16T10:15:00+02:00"):
        doc = valid_batch()
        doc["source"]["captured_at"] = ts
        assert validate_batch(doc).source["captured_at"] == ts


# --- detection-level (skip policy) ----------------------------------------


def test_detection_roundtrip():
    det = validate_detection(valid_detection())
    assert det.label == "animal"
    assert det.species == "zebra"
    assert det.confidence == 0.5
    assert det.bbox == (0.1, 0.2, 0.8, 0.9)


@pytest.mark.parametrize("label", ["person", "vehicle"])
def test_non_animal_labels_valid_without_species(label):
    det = validate_detection(valid_detection(label=label, species=None))
    assert det.species is None


def test_species_on_non_animal_rejected():
    with pytest.raises(SchemaError, match="only valid with label 'animal'"):
        validate_detection(valid_detection(label="person"))


@pytest.mark.parametrize("species", ["Zebra", "zebra!", "", "x" * 40, 7])
def test_bad_species_token_rejected(species):
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(species=species))


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, 0.5689])
def test_confidence_bounds_accepted(confidence):
    assert validate_detection(valid_detection(confidence=confidence)).confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.0001, "0.5", None, True])
def test_bad_confidence_rejected(confidence):
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(confidence=confidence))


def test_unknown_label_rejected():
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(label="machine"))


def test_bbox_must_have_four_numbers():
    for bbox in ([0.1, 0.2, 0.8], "0.1", [0.1, "x", 0.8, 0.9], None):
        with pytest.raises(SchemaError):
            validate_detection(valid_detection(bbox=bbox))


def test_bbox_inverted_coords_rejected():
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(bbox=[0.8, 0.2, 0.1, 0.9]))
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(bbox=[0.1, 0.9, 0.8, 0.2]))


def test_bbox_zero_area_rejected():
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(bbox=[0.5, 0.2, 0.5, 0.9]))


def test_bbox_slight_overshoot_is_clamped():
    det = validate_detection(valid_detection(bbox=[-0.01, 0.0, 1.01, 1.0]))
    assert det.bbox == (0.0, 0.0, 1.0, 1.0)


def test_bbox_far_overshoot_rejected():
    beyond = BBOX_CLAMP_TOLERANCE + 0.05
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(bbox=[-beyond, 0.0, 1.0, 1.0]))
    with pytest.raises(SchemaError):
        validate_detection(valid_detection(bbox=[0.0, 0.0, 1.0 + beyond, 1.0]))


# --- producer helper -------------------------------------------------------


def test_bbox_px_to_norm_from_s2_sample():
    # Real S2 output for aw-00007 (550x412): px [136.61, 90.36, 362.0, 338.98]
    norm = bbox_px_to_norm([136.61, 90.36, 362.0, 338.98], 550, 412)
    assert norm == pytest.approx([0.24838, 0.21932, 0.65818, 0.82277], abs=1e-4)


def test_bbox_px_to_norm_rejects_bad_frames():
    with pytest.raises(SchemaError):
        bbox_px_to_norm([0, 0, 10, 10], 0, 100)
    with pytest.raises(SchemaError):
        bbox_px_to_norm([0, 0, 10], 100, 100)
