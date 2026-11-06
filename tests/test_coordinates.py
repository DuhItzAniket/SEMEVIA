import pytest

from src.vision.coordinates import (
    box_valid, crop_bounds, expand_box, iou, letterbox_to_original, stanford_to_canonical,
)


def test_plan_letterbox_fixture():
    # 800x400 image, scale 0.8 -> 640x320, padded vertically to 640x640 (160 px top padding)
    box = letterbox_to_original([80, 200, 560, 440], scale=0.8, pad_x=0, pad_y=160)
    assert box == pytest.approx([100, 50, 700, 350])
    assert box[2] - box[0] == pytest.approx(600) and box[3] - box[1] == pytest.approx(300)


def test_iou_hand_fixtures():
    gt = [0, 0, 10, 10]
    assert iou(gt, [0, 0, 5, 10]) == pytest.approx(0.5)
    assert iou(gt, [10, 0, 20, 10]) == 0.0          # touching -> zero
    assert iou(gt, gt) == pytest.approx(1.0)


def test_stanford_conversion_is_half_open():
    # 1-based inclusive pixels 1..10 cover ten pixels -> [0, 10)
    assert stanford_to_canonical(1, 1, 10, 20) == [0.0, 0.0, 10.0, 20.0]


def test_crop_bounds_clip_every_border():
    assert crop_bounds([-5.2, -1, 50.1, 30.9], 40, 30) == (0, 0, 40, 30)
    assert crop_bounds([10.7, 3.2, 20.1, 9.9], 40, 30) == (10, 3, 21, 10)
    assert crop_bounds([39.5, 0, 45, 10], 40, 30) == (39, 0, 40, 10)
    assert crop_bounds([41, 0, 45, 10], 40, 30) is None


def test_expand_box_clips():
    assert expand_box([10, 10, 30, 20], 0.05, 100, 100) == pytest.approx([9, 9.5, 31, 20.5])
    assert expand_box([0, 0, 100, 100], 0.05, 100, 100) == [0, 0, 100, 100]


def test_box_valid():
    assert box_valid([0, 0, 10, 10], 10, 10)
    assert not box_valid([0, 0, 11, 10], 10, 10)
    assert not box_valid([5, 5, 5, 8], 10, 10)
    assert not box_valid([0, 0, float("nan"), 3], 10, 10)
