import numpy as np
import pytest

from src.passport.challenge import grid_cells, make_variants, run_challenge, summarize


def test_grid_bounds_floor_and_cover():
    cells = grid_cells(10, 5)
    assert cells == [(0, 0, 3, 2), (3, 0, 6, 2), (6, 0, 10, 2), (0, 2, 3, 5), (3, 2, 6, 5), (6, 2, 10, 5)]
    with pytest.raises(ValueError):
        grid_cells(2, 10)
    with pytest.raises(ValueError):
        grid_cells(10, 1)


def test_variants_are_independent_and_original_based():
    rng = np.random.RandomState(0)
    crop = rng.randint(0, 255, (60, 90, 3)).astype(np.uint8)
    variants, cells = make_variants(crop)
    assert len(variants) == 6
    for v, (l, t, r, b) in zip(variants, cells):
        diff = np.any(v != crop, axis=2)
        outside = diff.copy()
        outside[t:b, l:r] = False
        assert not outside.any()           # only its own cell changed
        assert diff[t:b, l:r].any()        # and that cell did change


def _fixed_logits(mrs, a=0, b=1, n=4):
    z = np.zeros((6, n))
    for i, m in enumerate(mrs):
        z[i, a], z[i, b] = m, 0.0
    return z


def test_margin_arithmetic_and_fixed_pair():
    crop = np.random.RandomState(1).randint(0, 255, (40, 60, 3)).astype(np.uint8)
    orig = np.array([4.0, 0.0, 0.0, 0.0])              # m0 = 4
    mrs = [1.0, 7.0, 4.0, 4.0, 4.0, 4.0]
    z = _fixed_logits(mrs)
    z[2, 3] = 99.0                                     # a variant changes the global winner
    res = run_challenge(crop, lambda vs: z, 0, 1, orig)
    deltas = [c["delta"] for c in res["cells"]]
    assert deltas[:3] == [3.0, -3.0, 0.0]
    assert res["class_a"] == 0 and res["class_b"] == 1   # pair stays fixed
    assert res["cells"][2]["new_top1"] == 3
    assert res["selected_cell"] == 0


def test_no_positive_drop_says_no_supporting_region():
    sel, text = summarize([0.0, -1.0, -0.5, 0.0, -2.0, 0.0])
    assert sel is None and text == "No supporting region found by this test."


def test_ties_choose_lower_grid_index():
    assert summarize([0.5, 2.0, 2.0, 0.1, 0.0, 0.0])[0] == 1


def test_invalid_scoring_output_fails():
    crop = np.zeros((40, 60, 3), np.uint8)
    with pytest.raises(RuntimeError):
        run_challenge(crop, lambda vs: np.zeros((5, 4)), 0, 1, np.zeros(4))
    with pytest.raises(ValueError):
        run_challenge(crop, lambda vs: np.zeros((6, 4)), 1, 1, np.zeros(4))
