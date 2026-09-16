"""Offline tests for the G3 evaluation glue (no cv2, models, or datasets).

eval.metrics is numpy-only by design, so everything here runs in any
environment with numpy+pytest. Expectations are hand-computed on
synthetic vectors — the point is pinning the DEFINITIONS (micro vs
macro, held-out leak guards, the parity matcher's argmax rule, the
trigger/selection ladder), not the datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import metrics


# ---------------------------------------------------------------------------
# Day/night IR flag + agreement
# ---------------------------------------------------------------------------


def test_ir_flag_threshold_boundary():
    assert metrics.ir_flag(0.8) is True      # IR monochrome night frame
    assert metrics.ir_flag(24.999) is True
    assert metrics.ir_flag(25.0) is False    # boundary is day
    assert metrics.ir_flag(122.0) is False


def test_agreement_rate_chance_level_for_random_flags():
    rng = np.random.default_rng(0)
    a = rng.random(1000) < 0.5
    b = rng.random(1000) < 0.5
    assert abs(metrics.agreement_rate(a, b) - 0.5) < 0.06  # ~chance
    assert metrics.agreement_rate([True, False], [True, False]) == 1.0
    assert metrics.agreement_rate(a, a) == 1.0  # numpy arrays fine
    with pytest.raises(ValueError):
        metrics.agreement_rate([True], [])


# ---------------------------------------------------------------------------
# top-k metrics: micro vs macro on synthetic vectors
# ---------------------------------------------------------------------------


def test_topk_micro_and_macro_disagree_exactly_when_class_imbalanced():
    # 8 buffalo all correct, 2 rhino all wrong: micro 0.8, macro 0.5.
    labels = [0] * 8 + [1] * 2
    ranked = [[0]] * 8 + [[0, 1]] * 2
    t = metrics.topk_metrics(labels, ranked, n_classes=3, k=2)
    assert t["n"] == 10
    assert t["top1_micro"] == 0.8
    assert t["top1_macro"] == pytest.approx(0.5)
    # both rhino rankings contain the truth within k=2 -> top-2 rescues them
    assert t["top2_micro"] == 1.0
    assert t["degenerate_topk"] is False


def test_topk_macro_ignores_absent_classes():
    # class 2 never appears as a label: macro averages present classes only.
    t = metrics.topk_metrics([0, 1], [[0], [1]], n_classes=3, k=1)
    assert t["top1_macro"] == 1.0
    assert set(t["per_class"]) == {"0", "1", "2"}
    assert t["per_class"]["2"]["n"] == 0
    assert t["per_class"]["2"]["top1_acc"] is None


def test_topk_degenerate_when_k_reaches_n_classes():
    # classifier v0's regime: 4 classes, k=5 -> top-4 always contains truth.
    labels = [0, 0, 1, 1]
    ranked = [[3, 2, 1, 0]] * 4  # truth ranked LAST every time
    t = metrics.topk_metrics(labels, ranked, n_classes=4, k=5)
    assert t["k_effective"] == 4
    assert t["degenerate_topk"] is True
    assert t["top1_micro"] == 0.0
    assert t["top5_micro"] == 1.0  # by construction — the flag says so


def test_topk_rejects_mismatched_and_empty_inputs():
    with pytest.raises(ValueError):
        metrics.topk_metrics([0], [[0], [1]], n_classes=2)
    with pytest.raises(ValueError):
        metrics.topk_metrics([0], [[]], n_classes=2)


def test_confusion_matrix_counts_and_accuracy():
    conf = metrics.confusion_matrix(
        [0, 0, 1], [0, 1, 1], class_names={0: "buffalo", 1: "zebra"}
    )
    # (0→0), (0→1), (1→1)
    assert conf["matrix"] == [[1, 1], [0, 1]]
    assert conf["per_class"]["buffalo"] == {"n": 2, "accuracy": 0.5}
    assert conf["per_class"]["zebra"] == {"n": 1, "accuracy": 1.0}


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


def _entries(**overrides):
    base = [
        {"id": f"test/img__{i:02d}", "split": "test", "class_name": "zebra",
         "source_image": f"african-wildlife/images/test/img{i}.jpg"}
        for i in range(3)
    ]
    for key, value in overrides.items():
        if callable(value):
            base = [{**e, key: value(e)} for e in base]
        else:
            for e in base:
                e[key] = value
    return base


def _sources(split: str) -> set[str]:
    return {f"african-wildlife/images/{split}/img{i}.jpg" for i in range(3)}


def test_split_integrity_passes_and_summarizes():
    summary = metrics.check_split_integrity(
        _entries(),
        allowed_splits={"test"},
        expected_counts={"test": 3},
        vocab={"zebra", "buffalo"},
        heldout_sources=_sources("test"),
        other_split_sources=_sources("train") | _sources("val"),
    )
    assert summary["n"] == 3
    assert summary["n_source_images"] == 3
    assert summary["by_class"] == {"zebra": 3}
    assert "trainer count match" in summary["checks"]
    assert "no train/val overlap" in summary["checks"]


def test_split_integrity_rejects_wrong_split():
    entries = _entries(split="train")
    with pytest.raises(metrics.SplitIntegrityError, match="split 'train'"):
        metrics.check_split_integrity(entries, allowed_splits={"test"})


def test_split_integrity_rejects_duplicate_ids():
    entries = _entries()
    entries[1]["id"] = entries[0]["id"]
    with pytest.raises(metrics.SplitIntegrityError, match="duplicate ids"):
        metrics.check_split_integrity(entries, allowed_splits={"test"})


def test_split_integrity_rejects_out_of_vocabulary_class():
    entries = _entries(class_name="capybara")
    with pytest.raises(metrics.SplitIntegrityError, match="vocabulary"):
        metrics.check_split_integrity(entries, allowed_splits={"test"},
                                      vocab={"zebra"})


def test_split_integrity_rejects_count_drift_vs_trainer_index():
    with pytest.raises(metrics.SplitIntegrityError, match="count|index"):
        metrics.check_split_integrity(
            _entries(), allowed_splits={"test"}, expected_counts={"test": 375}
        )


def test_split_integrity_rejects_train_image_leak():
    # a test-labelled crop sourced from a train image — the leak guard
    entries = _entries(
        source_image=lambda e: e["source_image"].replace("/test/", "/train/")
    )
    with pytest.raises(metrics.SplitIntegrityError, match="TRAIN/VAL LEAK"):
        metrics.check_split_integrity(
            entries,
            allowed_splits={"test"},
            other_split_sources=_sources("train"),
        )


def test_split_integrity_rejects_source_outside_heldout_split():
    entries = _entries(
        source_image=lambda e: e["source_image"].replace("/test/", "/val/")
    )
    with pytest.raises(metrics.SplitIntegrityError, match="not in the held-out"):
        metrics.check_split_integrity(
            entries, allowed_splits={"test"}, heldout_sources=_sources("test")
        )


def test_split_integrity_rejects_empty_set():
    with pytest.raises(metrics.SplitIntegrityError, match="empty"):
        metrics.check_split_integrity([], allowed_splits={"test"})


# ---------------------------------------------------------------------------
# Detection matching (parity protocol) + PR counts
# ---------------------------------------------------------------------------


def test_iou_xyxy_basic_and_degenerate():
    assert metrics.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert metrics.iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(5 / 15)
    assert metrics.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert metrics.iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0  # zero-area


def test_greedy_match_happy_path():
    # gt is xywh, dets are xyxy — per-detection GT index comes back
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    dets = [[0, 0, 10, 10], [100, 100, 110, 110]]
    assert metrics.greedy_match(gt, dets, [0.5, 0.9]) == [0, 1]
    counts = metrics.pr_counts([True, True], n_gt=2, n_images=1)
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "n_images": 1, "n_gt": 2,
                      "recall": 1.0, "precision": 1.0, "fp_per_image": 0.0}


def test_greedy_match_below_threshold_is_fp():
    gt = [[0, 0, 10, 10]]
    dets = [[0, 0, 4, 4]]  # IoU 0.16 < 0.5
    assert metrics.greedy_match(gt, dets, [0.9]) == [-1]


def test_greedy_match_consumed_gt_does_not_fall_back():
    """The parity rule: a det whose argmax GT is taken is an FP — it must
    NOT claim a free GT box it overlaps less (this exact case produced a
    1-TP divergence until the matcher was pinned to parity's rule)."""
    gt = [[0, 0, 10, 10], [2, 0, 10, 10]]  # heavy overlap with each other
    dets = [[0, 0, 10, 10], [1, 0, 10, 10]]  # both overlap gt0 most
    mapping = metrics.greedy_match(gt, dets, [0.9, 0.8])
    assert mapping[0] == 0        # higher score takes its argmax
    assert mapping[1] == -1       # argmax gt0 consumed -> FP, no fallback


def test_greedy_match_tie_resolves_to_lowest_gt_and_blocks_fallback():
    gt = [[0, 0, 10, 10], [0, 0, 10, 10]]  # identical boxes
    # equal scores: stable order keeps det order; the argmax tie goes to
    # gt0, so the second det's argmax (gt0) is consumed -> unmatched
    assert metrics.greedy_match(gt, [[0, 0, 10, 10], [0, 0, 10, 10]],
                                [0.5, 0.5]) == [0, -1]


def test_greedy_match_no_gt_all_fp():
    assert metrics.greedy_match([], [[0, 0, 5, 5]], [0.9]) == [-1]


def test_pr_counts_fp_per_image_and_empty():
    counts = metrics.pr_counts([False, False, True], n_gt=4, n_images=2)
    assert counts["tp"] == 1 and counts["fp"] == 2 and counts["fn"] == 3
    assert counts["recall"] == 0.25
    assert counts["precision"] == pytest.approx(1 / 3, abs=1e-4)  # 4dp rounding
    assert counts["fp_per_image"] == 1.0
    empty = metrics.pr_counts([], n_gt=0, n_images=0)
    assert empty["recall"] is None and empty["fp_per_image"] is None


# ---------------------------------------------------------------------------
# Trigger gate + max-per-trigger selection + rollup
# ---------------------------------------------------------------------------


def test_gate_pass_warmup_never_triggers():
    assert metrics.gate_pass(0.9, warmup=True) is False
    assert metrics.gate_pass(0.9, warmup=False) is True
    assert metrics.gate_pass(0.02, warmup=False) is False  # strictly greater
    assert metrics.gate_pass(0.0201, warmup=False) is True


def _frame(gate, confs=(), matched=()):
    return {"gate": gate,
            "dets": [{"conf": c, "matched": m in matched}
                     for m, c in enumerate(sorted(confs, reverse=True))]}


def test_select_max_per_trigger_picks_highest_conf_gated_frame():
    frames = [
        _frame(False, [0.9]),          # not gated: conf ignored
        _frame(True, [0.4, 0.3]),
        _frame(True, [0.8]),           # winner
        _frame(False),
    ]
    idx, reason = metrics.select_max_per_trigger(frames)
    assert (idx, reason) == (2, "selected")


def test_select_max_per_trigger_no_gate_and_no_detection():
    idx, reason = metrics.select_max_per_trigger([_frame(False, [0.9])])
    assert (idx, reason) == (None, "no_gate_pass")
    idx, reason = metrics.select_max_per_trigger([_frame(True), _frame(True)])
    assert (idx, reason) == (None, "no_detection")


def test_sequence_rollup_ladder():
    """union > gated > selected recall ladder on one synthetic sequence."""
    frames = [
        {"image_id": "f0", "warmup": True, "gate": False, "has_gt": False,
         "matched": False, "animal_dets": [], "animal_fp": []},
        {"image_id": "f1", "warmup": False, "gate": True, "has_gt": True,
         "matched": True, "animal_dets": [{"conf": 0.4, "matched": True}],
         "animal_fp": [], "selected": False},
        {"image_id": "f2", "warmup": False, "gate": False, "has_gt": True,
         "matched": True, "animal_dets": [], "animal_fp": []},
        {"image_id": "f3", "warmup": False, "gate": True, "has_gt": True,
         "matched": False, "animal_dets": [{"conf": 0.7, "matched": False}],
         "animal_fp": [0.7], "selected": True},  # winner frame, det unmatched
    ]
    r = metrics.sequence_rollup(frames, "selected")
    assert r["frames"] == 4
    assert r["frames_warmup"] == 1
    assert r["frames_gated"] == 2
    assert r["gt_frames"] == 3 and r["gt_frames_gated"] == 2
    assert r["union_matched"] is True          # f2 matched (ungated)
    assert r["gated_union_matched"] is True    # f1 matched on a gated frame
    assert r["selected_matched"] is False      # the chosen frame missed
    assert r["animal_dets_analyzed"] == 2
    assert r["animal_fp_analyzed"] == 1
    assert r["animal_fp_selected"] == 1


def test_aggregate_sequences_scopes_and_rates():
    def _seq(dataset, day_night, **fields):
        base = {"frames": 10, "frames_warmup": 2, "frames_gated": 4,
                "has_gt": True, "gt_frames": 3, "gt_frames_gated": 2,
                "union_matched": True, "gated_union_matched": True,
                "selected_matched": False, "selected_reason": "selected",
                "animal_dets_analyzed": 1, "animal_fp_analyzed": 1,
                "animal_fp_selected": 1, "dataset": dataset,
                "day_night": day_night}
        base.update(fields)
        return base

    records = [
        _seq("missouri", "night"),
        _seq("missouri", "day", selected_matched=True),
        _seq("african_wildlife", "day"),
    ]
    agg = metrics.aggregate_sequences(
        records, scopes={"dataset": ["missouri", "african_wildlife"],
                         "day_night": ["day", "night"]},
    )
    assert agg["all"]["sequences"] == 3
    assert agg["all"]["gt_sequences"] == 3
    assert agg["all"]["seq_union_recall"] == 1.0
    assert agg["all"]["seq_gated_recall"] == 1.0
    assert agg["all"]["seq_selected_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert agg["all"]["selection_loss"] == pytest.approx(2 / 3, abs=1e-4)
    assert agg["all"]["frame_gate_rate"] == pytest.approx(12 / 24)
    assert agg["all"]["inference_saving"] == pytest.approx(18 / 30)  # skipped/run
    mo = agg["dataset=missouri|day_night=night"]
    assert mo["sequences"] == 1 and mo["seq_selected_recall"] == 0.0
    day = agg["dataset=missouri|day_night=day"]
    assert day["seq_selected_recall"] == 1.0
    aw = agg["dataset=african_wildlife|day_night=day"]
    assert aw["gt_sequences"] == 1 and aw["seq_selected_recall"] == 0.0


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------


def test_rows_to_csv_quotes_and_column_order():
    csv = metrics.rows_to_csv([
        {"family": "clf", "metric": "top1_micro", "value": 0.9867, "n": 375},
        {"family": "det@0.25", "metric": "recall, all", "value": None},
    ])
    lines = csv.splitlines()
    assert lines[0] == "family,metric,value,n"
    assert lines[1] == "clf,top1_micro,0.9867,375"
    assert lines[2] == 'det@0.25,"recall, all",,'


def test_markdown_table_renders_none_as_dash_and_escapes_pipes():
    md = metrics.markdown_table(["a", "b"], [[1, None]])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[2] == "| 1 | - |"
    # scope labels carry pipes: they must not split the cell
    md = metrics.markdown_table(["scope"], [["missouri|night"]])
    assert md.splitlines()[2] == "| missouri\\|night |"


def test_emitters_empty():
    assert metrics.rows_to_csv([]) == ""
