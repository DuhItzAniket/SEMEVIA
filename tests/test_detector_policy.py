from src.models.detector import Detection, select_target


def _d(box, score):
    return Detection(box=box, score=score, coco_class="car", vehicle=True)


def test_largest_box_above_threshold_wins():
    small_conf = _d([0, 0, 10, 10], 0.95)
    big = _d([0, 0, 50, 50], 0.40)
    below = _d([0, 0, 100, 100], 0.10)
    sel, i = select_target([small_conf, big, below], threshold=0.25)
    assert sel is big and i == 1


def test_no_candidate_above_threshold():
    sel, i = select_target([_d([0, 0, 10, 10], 0.2)], threshold=0.25)
    assert sel is None and i is None


def test_tie_on_area_prefers_higher_score():
    a, b = _d([0, 0, 10, 10], 0.5), _d([5, 5, 15, 15], 0.6)
    assert select_target([a, b], 0.25)[0] is b
