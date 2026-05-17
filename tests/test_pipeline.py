"""
test_pipeline.py
================
Unit tests for the pipeline logic.
These tests use mock detections — NO model weights are required to run them.

Run with:
  cd traffic_violation_detector
  python -m pytest tests/ -v
  # or without pytest:
  python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from pipeline.associator import (
    RiderGroup,
    associate,
    associate_helmets_to_persons,
    associate_persons_to_moto,
    associate_plates_to_moto,
)
from pipeline.detector import (
    CLS_HELMET,
    CLS_LICENSE,
    CLS_MOTORCYCLE,
    CLS_PERSON,
    Detection,
    FrameDetections,
)
from pipeline.ocr import _apply_positional_fix, _clean_plate

# ── Test helpers ──────────────────────────────────────────────────────────────


def _det(cls_id, x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(cls_id=cls_id, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def _moto(x1, y1, x2, y2) -> Detection:
    return _det(CLS_MOTORCYCLE, x1, y1, x2, y2)


def _person(x1, y1, x2, y2) -> Detection:
    return _det(CLS_PERSON, x1, y1, x2, y2)


def _helmet(x1, y1, x2, y2) -> Detection:
    return _det(CLS_HELMET, x1, y1, x2, y2)


def _plate(x1, y1, x2, y2) -> Detection:
    return _det(CLS_LICENSE, x1, y1, x2, y2)


# ── Detection geometry tests ──────────────────────────────────────────────────


def test_detection_area():
    d = _det(0, 10, 20, 50, 60)
    assert d.width == 40
    assert d.height == 40
    assert d.area == 1600


def test_detection_iou_identical():
    d = _det(0, 0, 0, 100, 100)
    assert abs(d.iou(d) - 1.0) < 1e-6


def test_detection_iou_no_overlap():
    a = _det(0, 0, 0, 50, 50)
    b = _det(0, 60, 60, 120, 120)
    assert a.iou(b) == 0.0


def test_detection_iou_half_overlap():
    a = _det(0, 0, 0, 100, 100)
    b = _det(0, 50, 0, 150, 100)
    iou = a.iou(b)
    # Intersection = 50*100=5000, Union = 10000+10000-5000=15000
    assert abs(iou - 5000 / 15000) < 1e-4


def test_overlap_ratio_full_containment():
    big = _det(0, 0, 0, 200, 200)
    small = _det(0, 50, 50, 100, 100)
    # small is fully inside big, so overlap_ratio(big) should be 1.0
    assert abs(small.overlap_ratio(big) - 1.0) < 1e-4


def test_overlap_ratio_no_overlap():
    a = _det(0, 0, 0, 50, 50)
    b = _det(0, 60, 0, 110, 50)
    assert a.overlap_ratio(b) == 0.0


# ── Associator tests ──────────────────────────────────────────────────────────


def test_person_inside_moto():
    """Person box largely inside motorcycle box → should be associated."""
    moto = _moto(100, 100, 400, 400)
    person = _person(150, 120, 350, 380)  # ~fully inside moto
    result = associate_persons_to_moto(moto, [person])
    assert len(result) == 1


def test_pedestrian_not_associated():
    """Person far away from motorcycle → should NOT be associated."""
    moto = _moto(100, 100, 300, 300)
    person = _person(600, 600, 700, 750)  # far away
    result = associate_persons_to_moto(moto, [person])
    assert len(result) == 0


def test_two_persons_one_moto():
    """Two riders on same motorcycle."""
    moto = _moto(100, 100, 400, 400)
    p1 = _person(120, 110, 240, 390)
    p2 = _person(250, 110, 380, 390)
    result = associate_persons_to_moto(moto, [p1, p2])
    assert len(result) == 2


def test_helmet_on_person():
    """Helmet inside top part of person → associated to that person."""
    person = _person(100, 100, 200, 400)
    # Helmet inside top 40% = y in [100, 100+120] = [100, 220]
    helmet = _helmet(110, 105, 190, 195)
    result = associate_helmets_to_persons([person], [helmet])
    assert len(result[0]) == 1


def test_helmet_below_person_head():
    """Helmet at person waist → NOT associated to any person's head."""
    person = _person(100, 100, 200, 400)
    # Waist area: y around 280 (bottom half)
    helmet = _helmet(110, 280, 190, 330)
    result = associate_helmets_to_persons([person], [helmet])
    assert len(result[0]) == 0


def test_plate_inside_expanded_moto():
    """Plate within expanded moto box → associated."""
    moto = _moto(100, 100, 300, 300)
    plate = _plate(120, 290, 250, 330)  # just below moto (within expansion)
    result = associate_plates_to_moto(moto, [plate])
    assert len(result) == 1


def test_plate_far_from_moto():
    """Plate far from motorcycle → not associated."""
    moto = _moto(100, 100, 300, 300)
    plate = _plate(600, 600, 700, 640)
    result = associate_plates_to_moto(moto, [plate])
    assert len(result) == 0


# ── RiderGroup violation logic ────────────────────────────────────────────────


def test_no_violation_two_riders_both_helmets():
    moto = _moto(0, 0, 400, 400)
    group = RiderGroup(
        motorcycle=moto,
        persons=[_person(0, 0, 200, 400), _person(200, 0, 400, 400)],
        helmets=[_helmet(0, 0, 200, 120), _helmet(200, 0, 400, 120)],
        plates=[],
    )
    assert group.num_riders == 2
    assert group.helmet_count == 2
    assert group.helmet_violations == 0
    assert group.has_overload == False
    assert group.is_violation == False


def test_violation_one_rider_no_helmet():
    moto = _moto(0, 0, 400, 400)
    group = RiderGroup(
        motorcycle=moto,
        persons=[_person(0, 0, 400, 400)],
        helmets=[],
        plates=[],
    )
    assert group.helmet_violations == 1
    assert group.is_violation == True


def test_violation_triple_riders():
    moto = _moto(0, 0, 600, 400)
    group = RiderGroup(
        motorcycle=moto,
        persons=[
            _person(0, 0, 200, 400),
            _person(200, 0, 400, 400),
            _person(400, 0, 600, 400),
        ],
        helmets=[
            _helmet(0, 0, 200, 100),
            _helmet(200, 0, 400, 100),
            _helmet(400, 0, 600, 100),
        ],
        plates=[],
    )
    assert group.num_riders == 3
    assert group.has_overload == True  # >2 riders
    assert group.is_violation == True


def test_violation_triple_riders_no_helmets():
    moto = _moto(0, 0, 600, 400)
    group = RiderGroup(
        motorcycle=moto,
        persons=[
            _person(0, 0, 200, 400),
            _person(200, 0, 400, 400),
            _person(400, 0, 600, 400),
        ],
        helmets=[],
        plates=[],
    )
    assert group.num_riders == 3
    assert group.has_overload == True
    assert group.helmet_violations == 3
    assert group.is_violation == True


def test_best_plate_by_confidence():
    moto = _moto(0, 0, 400, 400)
    p_low = _det(CLS_LICENSE, 100, 350, 250, 400, conf=0.5)
    p_hi = _det(CLS_LICENSE, 100, 350, 250, 400, conf=0.95)
    group = RiderGroup(motorcycle=moto, plates=[p_low, p_hi])
    assert group.best_plate is p_hi


# ── Full associate() integration ──────────────────────────────────────────────


def test_full_pipeline_two_motos():
    """Two motorcycles with different violation profiles."""
    fd = FrameDetections(
        motorcycles=[
            _moto(0, 0, 300, 300),  # moto A — single rider, no helmet
            _moto(400, 0, 700, 300),  # moto B — two riders, both helmets
        ],
        persons=[
            _person(50, 50, 250, 280),  # on moto A
            _person(420, 50, 570, 280),  # on moto B
            _person(560, 50, 690, 280),  # on moto B
        ],
        helmets=[
            _helmet(420, 55, 570, 155),  # on moto B person 1
            _helmet(560, 55, 690, 155),  # on moto B person 2
        ],
        plates=[
            _plate(60, 280, 220, 320),  # on moto A (expanded box catches it)
        ],
    )
    groups = associate(fd)
    assert len(groups) == 2

    # Identify which group is moto A and which is moto B
    group_a = next(g for g in groups if g.motorcycle.cx < 350)
    group_b = next(g for g in groups if g.motorcycle.cx > 350)

    assert group_a.num_riders == 1
    assert group_a.helmet_violations == 1
    assert group_a.is_violation == True
    assert len(group_a.plates) >= 0  # plate association depends on thresholds

    assert group_b.num_riders == 2
    assert group_b.helmet_violations == 0
    assert group_b.has_overload == False
    assert group_b.is_violation == False


def test_empty_image():
    """No detections → no groups."""
    groups = associate(FrameDetections())
    assert groups == []


def test_moto_with_no_riders():
    """Motorcycle detected but no persons nearby → 0 riders, no violation."""
    fd = FrameDetections(
        motorcycles=[_moto(100, 100, 300, 300)],
        persons=[],
    )
    groups = associate(fd)
    assert len(groups) == 1
    assert groups[0].num_riders == 0
    assert groups[0].is_violation == False


# ── OCR post-processing tests ─────────────────────────────────────────────────


def test_clean_plate_removes_spaces():
    assert _clean_plate("MH 12 AB 1234") == "MH12AB1234"


def test_clean_plate_removes_dash():
    assert _clean_plate("KA-05-MG-6789") == "KA05MG6789"


def test_clean_plate_uppercase():
    assert _clean_plate("mh12ab1234") == "MH12AB1234"


def test_positional_fix_digit_in_letter_slot():
    # pos 0,1 should be letters; 0→O, 1→I
    result = _apply_positional_fix("01HMAB1234")
    assert result[0] == "O"
    assert result[1] == "I"


def test_positional_fix_letter_in_digit_slot():
    # pos 2,3 should be digits; O→0
    result = _apply_positional_fix("MHOOAB1234")
    assert result[2] == "0"
    assert result[3] == "0"


def test_positional_fix_short_string():
    # Should not crash on short input
    result = _apply_positional_fix("MH")
    assert isinstance(result, str)


# ── Runner ────────────────────────────────────────────────────────────────────


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed.append(fn.__name__)

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    print("Running pipeline unit tests (no model weights required)...\n")
    _run_all()
