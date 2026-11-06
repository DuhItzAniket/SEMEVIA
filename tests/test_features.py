import numpy as np
import pytest

from src.vision.features import SLICES, extract_classical
from src.vision.proposals import propose


def _car_like(h=200, w=400, seed=0):
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), 230, np.uint8)
    img[60:160, 40:360] = (180, 30, 30)              # body
    img[30:70, 120:280] = (60, 60, 80)               # cabin
    for cx in (100, 300):                            # wheels
        yy, xx = np.ogrid[:h, :w]
        img[(yy - 160) ** 2 + (xx - cx) ** 2 < 30 ** 2] = (20, 20, 20)
    return np.clip(img.astype(int) + rng.randint(-5, 5, img.shape), 0, 255).astype(np.uint8)


def test_length_layout_and_finite():
    r = extract_classical(_car_like(), box=[10, 20, 410, 220], image_size=(800, 600))
    v = r.vector
    assert v.shape == (1816,) and v.dtype == np.float32 and np.isfinite(v).all()
    assert v[SLICES["hog"]].shape == (1764,)
    assert v[SLICES["lbp"]].sum() == pytest.approx(1.0, abs=1e-5)
    assert v[SLICES["hsv"]].sum() == pytest.approx(3.0, abs=1e-4)      # three normalised channels
    assert r.measurements["crop_aspect"] == pytest.approx(2.0)          # measured before square resize
    assert r.measurements["empty_contour"] is False


def test_blank_crop_returns_zeros_and_flags():
    r = extract_classical(np.full((120, 300, 3), 128, np.uint8))
    v = r.vector
    assert np.isfinite(v).all()
    assert r.measurements["empty_contour"] is True and r.measurements["near_uniform"] is True
    g = v[SLICES["geometry"]]
    assert g[1] == 0 and g[2] == 0 and g[3] == 0 and g[4] == 0          # area, solidity, extent, edges
    assert list(v[SLICES["flags"]]) == [1.0, 1.0]


def test_tiny_and_extreme_crops():
    for shape in ((1, 1, 3), (2, 500, 3), (500, 3, 3)):
        r = extract_classical(np.zeros(shape, np.uint8))
        assert r.vector.shape == (1816,) and np.isfinite(r.vector).all()


def test_deterministic():
    a = extract_classical(_car_like(seed=3)).vector
    b = extract_classical(_car_like(seed=3)).vector
    assert np.array_equal(a, b)


def test_views_requested():
    r = extract_classical(_car_like(), want_views=True)
    assert set(r.views) == {"gray", "edges", "contours"}
    assert r.views["edges"].dtype == np.uint8


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        extract_classical(np.zeros((10, 10), np.uint8))


def test_proposals_find_object_and_fail_cleanly_on_blank():
    img = np.full((400, 800, 3), 235, np.uint8)
    img[100:340, 120:700] = _car_like(240, 580)
    props = propose(img)
    assert props and 0 <= props[0].score <= 1
    from src.vision.coordinates import iou

    # the synthetic vehicle (body, cabin, wheels) occupies about x 160..480, y 130..290
    assert iou(props[0].box, [160, 130, 480, 290]) > 0.7
    assert propose(np.full((400, 800, 3), 128, np.uint8)) == []
