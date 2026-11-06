import numpy as np
import pytest

from src.evaluation.stress import apply_condition, stratified_subset


def _img():
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (200, 300, 3)).astype(np.uint8)


def test_brightness_clips_and_scales():
    img = np.full((4, 4, 3), 200, np.uint8)
    assert apply_condition(img, {"type": "brightness", "factor": 1.5}, [0, 0, 4, 4], "k", 1, "b").max() == 255
    assert apply_condition(img, {"type": "brightness", "factor": 0.5}, [0, 0, 4, 4], "k", 1, "b").max() == 100


def test_downsample_restores_size():
    out = apply_condition(_img(), {"type": "downsample", "factor": 0.25}, [0, 0, 300, 200], "k", 1, "d")
    assert out.shape == (200, 300, 3)


def test_occlusion_area_inside_box_and_deterministic():
    img = _img()
    box = [50, 40, 250, 160]
    a = apply_condition(img, {"type": "occlusion", "area_frac": 0.30}, box, "test/1.jpg", 7, "occl_0.30")
    b = apply_condition(img, {"type": "occlusion", "area_frac": 0.30}, box, "test/1.jpg", 7, "occl_0.30")
    assert np.array_equal(a, b)
    changed = np.any(a != img, axis=2)
    ys, xs = np.nonzero(changed)
    assert xs.min() >= 50 and xs.max() < 250 and ys.min() >= 40 and ys.max() < 160
    assert changed.sum() / (200 * 120) == pytest.approx(0.30, abs=0.03)
    c = apply_condition(img, {"type": "occlusion", "area_frac": 0.30}, box, "test/2.jpg", 7, "occl_0.30")
    assert not np.array_equal(a, c)        # placement depends on the image key


def test_clean_is_identity_and_unknown_fails():
    img = _img()
    assert apply_condition(img, {"type": "none"}, [0, 0, 1, 1], "k", 1, "c") is img
    with pytest.raises(ValueError):
        apply_condition(img, {"type": "rotate"}, [0, 0, 1, 1], "k", 1, "x")


def test_stratified_subset_two_per_class_and_seeded():
    keys = [f"test/{i:05d}.jpg" for i in range(30)]
    labels = [i % 5 for i in range(30)]
    s1 = stratified_subset(keys, labels, 2, 3)
    assert len(s1) == 10 and s1 == stratified_subset(keys, labels, 2, 3)
    assert sorted({labels[keys.index(k)] for k in s1}) == [0, 1, 2, 3, 4]
