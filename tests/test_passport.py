import numpy as np
import pytest

from src.passport.conformal import (
    ConformalQuantile, calibrate, calibration_scores, candidate_set, failure_set,
)
from src.passport.decision import make_passport
from src.passport.hierarchy import Hierarchy

H = Hierarchy.load()


def oid(*orig):  # original one-based ids -> application ids
    return [o - 1 for o in orig]


# ---------- conformal order statistic (plan boundary fixtures)
def test_n815_uses_one_based_position_735():
    cq = calibrate(np.arange(1, 816) / 1000.0, 0.10)
    assert cq.k == 735 and cq.q == pytest.approx(0.735) and cq.quantile_mode == "order_statistic"


def test_n9_and_n8():
    cq9 = calibrate(np.arange(1, 10) / 10.0, 0.10)
    assert cq9.k == 9 and cq9.q == pytest.approx(0.9)
    cq8 = calibrate(np.arange(1, 9) / 10.0, 0.10)
    assert cq8.k == 9 and cq8.quantile_mode == "full_set" and cq8.q is None
    assert cq8.to_dict()["q"] is None   # serialised as null, never Infinity


def test_equality_is_included_and_empty_set_kept():
    full = ConformalQuantile(0.1, 10, 9, "order_statistic", 0.75)
    assert candidate_set(np.array([0.50, 0.25, 0.25]), full) == [0, 1, 2]
    empty = ConformalQuantile(0.1, 10, 9, "order_statistic", 0.20)
    assert candidate_set(np.array([0.60, 0.25, 0.15]), empty) == []


def test_tie_ordering_by_class_id_and_full_set_mode():
    cq = ConformalQuantile(0.1, 10, 9, "order_statistic", 0.9)
    assert candidate_set(np.array([0.1, 0.4, 0.4, 0.1]), cq) == [1, 2, 0, 3]
    fs = ConformalQuantile(0.1, 8, 9, "full_set", None)
    assert candidate_set(np.array([0.7, 0.2, 0.1]), fs) == [0, 1, 2]


def test_invalid_inputs_fail_safely():
    for bad in ([], [0.1, float("nan")]):
        with pytest.raises(ValueError):
            calibrate(np.array(bad), 0.1)
    for a in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            calibrate(np.array([0.1, 0.2]), a)
    with pytest.raises(ValueError):
        candidate_set(np.array([0.5, np.inf]), ConformalQuantile(0.1, 2, 2, "order_statistic", 0.5))


def test_calibration_scores():
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert calibration_scores(p, np.array([0, 0])).tolist() == pytest.approx([0.3, 0.8])


# ---------- hierarchy (plan illustrative outcomes)
def test_hierarchy_cases():
    assert H.specificity(oid(34)).level == "category"
    assert H.specificity(oid(34)).label == "BMW M3 Coupe 2012"
    s = H.specificity(oid(23, 24))
    assert (s.level, s.label) == ("family", "Audi S4 Sedan") and "2007" in s.unresolved[0]
    assert H.specificity(oid(42, 43)).label == "Bentley Continental GT Coupe"
    s = H.specificity(oid(21, 22))
    assert (s.level, s.label) == ("make", "Audi")                 # body-style families not merged
    assert H.specificity(oid(23, 34)).level == "root"


def test_sixth_candidate_from_other_make_forces_root():
    audi = oid(12, 13, 14, 15, 16)                               # five Audi categories
    assert H.specificity(audi).level == "make"
    assert H.specificity(audi + oid(34)).level == "root"


def test_every_category_maps_once_and_multiword_makes():
    assert sorted(H.classes) == list(range(196))
    assert H.path(0)[0] == "AM General" and H.path(oid(8)[0])[0] == "Aston Martin"


# ---------- decision states
CAL = {"bundle_id": "test", "kind": "development"}


def _probs(top: int, p_top: float, second: int | None = None, p_second: float = 0.0):
    p = np.full(196, 0.0)
    p[top] = p_top
    if second is not None:
        p[second] = p_second
    rest = 1.0 - p.sum()
    idx = [i for i in range(196) if p[i] == 0]
    p[idx] = rest / len(idx)
    return p


def test_singleton_passing_policy_is_exact_candidate():
    p = _probs(oid(34)[0], 0.95)
    cq = ConformalQuantile(0.1, 815, 735, "order_statistic", 0.5)
    out = make_passport(p, True, CAL, cq, 0.3, H)
    assert out["decision_state"] == "exact_candidate" and out["accepted_identity"] == oid(34)[0]
    assert out["candidate_set"] == oid(34)


def test_reject_overrides_singleton_but_keeps_set():
    p = _probs(oid(34)[0], 0.95)
    cq = ConformalQuantile(0.1, 815, 735, "order_statistic", 0.5)
    out = make_passport(p, True, CAL, cq, 0.99, H)
    assert out["decision_state"] == "rejected" and out["accepted_identity"] is None
    assert out["candidate_set"] == oid(34)


def test_family_answer_is_ambiguous_with_family_specificity():
    p = _probs(oid(23)[0], 0.48, oid(24)[0], 0.47)
    cq = ConformalQuantile(0.1, 815, 735, "order_statistic", 0.6)
    out = make_passport(p, True, CAL, cq, 0.3, H)
    assert out["decision_state"] == "ambiguous" and out["specificity"]["level"] == "family"
    assert out["accepted_identity"] is None


def test_cross_make_set_is_inadequate():
    p = _probs(oid(23)[0], 0.48, oid(34)[0], 0.47)
    cq = ConformalQuantile(0.1, 815, 735, "order_statistic", 0.6)
    out = make_passport(p, True, CAL, cq, 0.3, H)
    assert out["decision_state"] == "inadequate" and "no_shared_make" in out["reason_codes"]


def test_empty_set_is_inadequate():
    p = _probs(5, 0.3)
    cq = ConformalQuantile(0.1, 815, 735, "order_statistic", 0.2)
    out = make_passport(p, True, CAL, cq, 0.1, H)
    assert out["decision_state"] == "inadequate" and out["candidate_set"] == []
    assert "empty_candidate_set" in out["reason_codes"]


def test_detection_failure_full_set_no_probabilities():
    out = make_passport(None, False, CAL, None, 0.3, H)
    assert out["decision_state"] == "detection_failed" and out["candidate_count"] == 196
    assert out["candidate_set"] == failure_set() and out["candidate_set_is_failure_convention"]
    assert out["max_probability"] is None and out["accepted_identity"] is None


def test_comparison_mode_borrows_no_calibration():
    out = make_passport(_probs(1, 0.9), True, None, None, 0.3, H)
    assert out["decision_state"] == "comparison_only" and out["candidate_set"] == []
    assert out["calibration"] is None


def test_nonfinite_probabilities_rejected():
    p = _probs(1, 0.9)
    p[3] = np.nan
    with pytest.raises(ValueError):
        make_passport(p, True, CAL, ConformalQuantile(0.1, 9, 9, "order_statistic", 0.5), 0.3, H)
