"""Fixture tests for P5 freeze / final calibration / failure selection (no held-out data used)."""
import numpy as np
import pytest

from src.evaluation.analysis import select_failures, subgroup_table
from src.passport.calibration import choose_reject_threshold, final_conformal_scores, fit_temperature, model_fingerprint


def test_fit_temperature_recovers_known_scale():
    rng = np.random.RandomState(0)
    n, c, T_true = 4000, 10, 2.5
    base = rng.randn(n, c) * 2.0
    p = np.exp(base) / np.exp(base).sum(1, keepdims=True)
    y = np.array([rng.choice(c, p=pi) for pi in p])
    assert fit_temperature(base * T_true, y) == pytest.approx(T_true, rel=0.1)


def test_reject_threshold_without_and_with_negatives():
    known = np.array([0.9] * 90 + [0.12] * 10)              # 90% of known at 0.9
    grid = [0.1, 0.2, 0.5, 0.95]
    r = choose_reject_threshold(known, grid, 0.90)
    assert r["threshold"] == 0.5 and r["feasible"] and not r["negatives_used"]
    neg = np.array([0.15, 0.6, 0.6, 0.6])
    r2 = choose_reject_threshold(known, grid, 0.90, neg)
    assert r2["threshold"] == 0.2 or r2["threshold"] == 0.5
    fa = {row["threshold"]: row["negative_false_acceptance"] for row in r2["table"]}
    assert fa[r2["threshold"]] == min(fa[t] for t in (0.1, 0.2, 0.5))
    r3 = choose_reject_threshold(np.array([0.05] * 10), grid, 0.90)
    assert not r3["feasible"]


def test_final_scores_use_uniform_fallback_for_failures():
    z = [0.0] * 196
    z[3] = 10.0
    rows = [{"y": 3, "sel_box": [0, 0, 1, 1], "z_hybrid": z}, {"y": 5, "sel_box": None}]
    s, failures = final_conformal_scores(rows, "z_hybrid", 1.0)
    assert failures == 1 and s[1] == pytest.approx(1 - 1 / 196) and s[0] < 0.01


def test_model_fingerprint_changes_with_heads_and_runtime():
    m = {"assets": [{"name": "yolo11n.pt", "sha256": "a"}, {"name": "head_hybrid.joblib", "sha256": "b"},
                    {"name": "encoder_fp32.onnx", "sha256": "c"}, {"name": "calibration_final.json", "sha256": "d"}]}
    f1 = model_fingerprint(m, "hybrid", "torch")
    m2 = {"assets": [dict(a, sha256="x") if a["name"] == "head_hybrid.joblib" else a for a in m["assets"]]}
    assert model_fingerprint(m2, "hybrid", "torch") != f1
    m3 = {"assets": [dict(a, sha256="y") if a["name"].endswith(".json") else a for a in m["assets"]]}
    assert model_fingerprint(m3, "hybrid", "torch") == f1          # calibration files do not change the model
    assert model_fingerprint(m, "hybrid", "onnx_fp32") != f1


def _z(top, n=196, val=8.0, second=None):
    z = [0.0] * n
    z[top] = val
    if second is not None:
        z[second] = val - 1
    return z


def test_failure_rules_and_subgroups():
    classes = {i: {"make": "A" if i < 100 else "B", "name": f"c{i}"} for i in range(196)}
    gts = {f"k{i}": [0, 0, 10, 10] for i in range(5)}
    rows = [
        {"image_key": "k0", "y": 1, "sel_box": None},                                      # no box
        {"image_key": "k1", "y": 1, "sel_box": [0, 0, 10, 10], "z_hybrid": _z(2, val=9)},    # same-make error
        {"image_key": "k2", "y": 1, "sel_box": [0, 0, 10, 10], "z_hybrid": _z(150, val=12)}, # cross-make error
        {"image_key": "k3", "y": 4, "sel_box": [0, 0, 10, 10], "z_hybrid": _z(4)},           # correct
        {"image_key": "k4", "y": 4, "sel_box": [20, 20, 30, 30], "z_hybrid": _z(4)},         # wrong box
    ]
    f = select_failures(rows, gts, classes, "z_hybrid")
    assert [x["row"]["image_key"] for x in f] == ["k0", "k1", "k2"]
    t = subgroup_table(rows, gts, {"k1": "side", "k3": "side", "k4": "front"}, "z_hybrid")
    assert t["side"]["count"] == 2 and t["side"]["top1"] == 1 and t["front"]["top1"] == 0
