import numpy as np
import pytest

from src.evaluation.metrics import (
    classification_topk, coco_ap, ece, end_to_end_topk, localization, nll, ratio,
    rejection_contingency, set_metrics,
)


def test_plan_ece_fixture_is_count_weighted():
    r = ece(np.array([0.51, 0.59, 0.81, 0.85, 0.89]), np.array([1, 0, 1, 1, 0]))
    assert r["ece"] == pytest.approx(0.13, abs=1e-9)
    occupied = [b for b in r["bins"] if b["count"]]
    assert [b["count"] for b in occupied] == [2, 3]
    assert occupied[0]["mean_confidence"] == pytest.approx(0.55) and occupied[0]["accuracy"] == 0.5
    assert all(b["accuracy"] is None for b in r["bins"] if b["count"] == 0)


def test_confidence_one_goes_in_last_bin():
    r = ece(np.array([1.0]), np.array([1]))
    assert r["bins"][-1]["count"] == 1 and r["ece"] == 0.0


def test_plan_localization_fixture():
    gt = [[0, 0, 10, 10]] * 3
    r = localization([[0, 0, 5, 10], [10, 0, 20, 10], None], gt)
    assert r["primary_recall_at_iou"]["value"] == pytest.approx(1 / 3)      # inclusive 0.50
    assert r["mean_iou_valid_selected"]["value"] == pytest.approx(0.25)    # (0.5 + 0) / 2
    assert r["mean_iou_zero_for_failures"]["value"] == pytest.approx(0.5 / 3)
    assert r["missing"] == 1 and r["wrong_target_below_thr"] == 1


def test_dangerous_case_runtime_box_decides():
    # runtime picked a large wrong vehicle; a smaller perfect detection existed but is irrelevant
    r = end_to_end_topk([[50, 50, 100, 100]], [[0, 0, 10, 10]], [[7]], [7], k=1)
    assert r["value"] == 0.0
    ok = end_to_end_topk([[0, 0, 10, 10], None], [[0, 0, 10, 10]] * 2, [[3, 7], None], [7, 7], k=5)
    assert ok["numerator"] == 1 and ok["denominator"] == 2


def test_plan_rejection_fixture():
    r = rejection_contingency(80, 72, 20, 6, 34)
    assert r["known_retention"]["value"] == pytest.approx(0.80)
    assert r["accepted_known_accuracy"]["value"] == pytest.approx(0.90)
    assert r["negative_false_acceptance"]["value"] == pytest.approx(0.15)


def test_zero_denominator_is_not_applicable():
    assert ratio(0, 0)["value"] is None


def test_topk_and_nll():
    z = np.array([[3.0, 2.0, 1.0], [0.0, 0.0, 5.0]])
    assert classification_topk(z, np.array([1, 2]), 1)["value"] == 0.5
    assert classification_topk(z, np.array([1, 2]), 2)["value"] == 1.0
    assert nll(np.array([[0.5, 0.5]]), np.array([0])) == pytest.approx(np.log(2))


def test_set_metrics_counts_failures():
    r = set_metrics([[1], [2, 3], [], list(range(196))], [1, 4, 0, 5], [True, True, False, False])
    assert r["coverage"]["value"] == 0.5
    assert r["singleton_accuracy"]["value"] == 1.0 and r["empty"]["numerator"] == 1
    assert r["specific_answer_rate"]["value"] == 0.5


def test_coco_ap_perfect_and_wrong_category():
    gts = [{"image_id": i, "category_id": 1, "box": [0, 0, 50, 50]} for i in range(1, 6)]
    perfect = [dict(g, score=0.9) for g in gts]
    r = coco_ap(gts, perfect, [1, 2])
    assert r["AP50"] == pytest.approx(1.0) and r["AP50_95"] == pytest.approx(1.0)
    wrong = [dict(g, category_id=2 if g["image_id"] == 1 else 1, score=0.9) for g in gts]
    assert coco_ap(gts, wrong, [1, 2])["AP50"] < 1.0
