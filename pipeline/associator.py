"""
associator.py
=============
Associates persons, helmets, and license plates with their parent motorcycle
using lightweight geometric heuristics only. No tracking, no pose estimation.

Association strategy (applied in priority order):
  1. Containment  — person/object is largely inside or overlapping the moto box
  2. Proximity    — bottom-center of person is close to the moto box
  3. IoU fallback — broad IoU check when containment fails (e.g. partially visible bikes)

Helmet-to-person association:
  Helmets are associated to the person whose upper-body region they overlap
  the most, since a helmet always sits on the head (upper ~30% of person box).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pipeline.detector import Detection, FrameDetections

logger = logging.getLogger(__name__)


# ── Tunable thresholds ────────────────────────────────────────────────────────
# These values were set empirically for images at 640px resolution.
# Person inside moto: at least this fraction of person area inside moto box
PERSON_IN_MOTO_RATIO = 0.50
# Person proximity: bottom-center of person within PROX_FACTOR * moto_height
PERSON_PROX_FACTOR = 0.30
# Broad IoU fallback threshold
PERSON_MOTO_IOU_THRESH = 0.15
# Helmet upper-body overlap: helmet inside top HELMET_HEAD_FRAC of person box
HELMET_HEAD_FRAC = 0.40
HELMET_PERSON_RATIO = 0.30  # at least this much of helmet inside person upper body
# Plate inside moto: at least this fraction of plate inside the expanded moto box
PLATE_IN_MOTO_RATIO = 0.30
PLATE_EXPAND_FACTOR = 1.30  # expand moto box for plate lookup (plates near bottom)


@dataclass
class RiderGroup:
    """All objects belonging to one motorcycle."""

    motorcycle: Detection
    persons: list[Detection] = field(default_factory=list)
    helmets: list[Detection] = field(default_factory=list)
    plates: list[Detection] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def num_riders(self) -> int:
        return len(self.persons)

    @property
    def helmet_count(self) -> int:
        return len(self.helmets)

    @property
    def has_overload(self) -> bool:
        """True when more than 2 persons are on the bike."""
        return self.num_riders > 2

    @property
    def helmet_violations(self) -> int:
        """
        Number of riders without a helmet.
        Strategy: count helmets vs riders. If there are more riders than helmets,
        the difference is the number of violators.
        Cap at num_riders to avoid negatives if detection is noisy.
        """
        return max(0, self.num_riders - self.helmet_count)

    @property
    def is_violation(self) -> bool:
        return self.has_overload or self.helmet_violations > 0

    @property
    def best_plate(self) -> Optional[Detection]:
        if not self.plates:
            return None
        return max(self.plates, key=lambda p: p.conf)


def _expand_box(det: Detection, factor: float) -> tuple[float, float, float, float]:
    """Expand a bounding box by factor around its center."""
    cx, cy = det.cx, det.cy
    hw = det.width * factor / 2
    hh = det.height * factor / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def _dummy_det_from_xyxy(x1, y1, x2, y2) -> Detection:
    """Create a dummy Detection for geometric ops on a raw box."""
    from pipeline.detector import CLS_MOTORCYCLE

    return Detection(cls_id=CLS_MOTORCYCLE, conf=1.0, x1=x1, y1=y1, x2=x2, y2=y2)


def associate_persons_to_moto(
    moto: Detection,
    persons: list[Detection],
) -> list[Detection]:
    """
    Return all persons that belong to this motorcycle.
    A person belongs if ANY of these conditions hold:
      1. At least PERSON_IN_MOTO_RATIO of the person's area is inside the moto box
      2. The bottom-center of the person is within PROX_FACTOR * moto_height
         below the moto box center (handles riders on top of moto)
      3. IoU > PERSON_MOTO_IOU_THRESH (broad catch-all)
    """
    associated = []
    for person in persons:
        # 1. Containment
        ratio = person.overlap_ratio(moto)
        if ratio >= PERSON_IN_MOTO_RATIO:
            associated.append(person)
            continue

        # 2. Proximity — bottom-center of person near moto
        prox_thresh = moto.height * PERSON_PROX_FACTOR
        dx = abs(person.bottom_cx - moto.cx)
        dy = abs(person.bottom_cy - moto.cy)
        if dx < moto.width * 0.6 and dy < prox_thresh:
            associated.append(person)
            continue

        # 3. IoU fallback
        if person.iou(moto) > PERSON_MOTO_IOU_THRESH:
            associated.append(person)

    return associated


def associate_helmets_to_persons(
    persons: list[Detection],
    helmets: list[Detection],
) -> dict[int, list[Detection]]:
    """
    Map person index → list of helmets on that person.
    A helmet is associated to a person if it overlaps significantly with the
    upper HELMET_HEAD_FRAC region of the person's bounding box.
    """
    # Build upper-body boxes for each person (top HELMET_HEAD_FRAC of height)
    person_helmets: dict[int, list[Detection]] = {i: [] for i in range(len(persons))}
    used_helmets = set()

    for helmet in helmets:
        best_idx = -1
        best_score = 0.0

        for i, person in enumerate(persons):
            # Upper body region: top HELMET_HEAD_FRAC of person box
            head_y2 = person.y1 + person.height * HELMET_HEAD_FRAC
            head_box = _dummy_det_from_xyxy(person.x1, person.y1, person.x2, head_y2)
            overlap = helmet.overlap_ratio(head_box)

            if overlap > best_score:
                best_score = overlap
                best_idx = i

        if best_idx >= 0 and best_score >= HELMET_PERSON_RATIO:
            person_helmets[best_idx].append(helmet)
            used_helmets.add(id(helmet))

    return person_helmets


def associate_plates_to_moto(
    moto: Detection,
    plates: list[Detection],
) -> list[Detection]:
    """
    Return plates that belong to this motorcycle.
    Uses an expanded version of the moto box to catch plates near the bottom.
    """
    ex1, ey1, ex2, ey2 = _expand_box(moto, PLATE_EXPAND_FACTOR)
    expanded = _dummy_det_from_xyxy(ex1, ey1, ex2, ey2)

    associated = []
    for plate in plates:
        if plate.overlap_ratio(expanded) >= PLATE_IN_MOTO_RATIO:
            associated.append(plate)
    return associated


def associate(detections: FrameDetections) -> list[RiderGroup]:
    """
    Main entry point. Takes FrameDetections and returns a RiderGroup per motorcycle.

    Algorithm:
    1. For each motorcycle, find associated persons.
    2. For associated persons, find their helmets.
    3. For each motorcycle, find associated license plates.

    Persons not associated with any motorcycle are ignored (they are pedestrians).
    """
    groups: list[RiderGroup] = []

    if not detections.motorcycles:
        logger.debug("[Associator] No motorcycles detected")
        return groups

    # Track which persons are already claimed (avoid double-counting)
    claimed_persons: set[int] = set()

    for moto in detections.motorcycles:
        # 1. Associate persons
        candidates = [p for p in detections.persons if id(p) not in claimed_persons]
        persons = associate_persons_to_moto(moto, candidates)

        for p in persons:
            claimed_persons.add(id(p))

        # 2. Associate helmets to those persons
        person_helmet_map = associate_helmets_to_persons(persons, detections.helmets)
        all_helmets = [h for lst in person_helmet_map.values() for h in lst]

        # 3. Associate license plates
        plates = associate_plates_to_moto(moto, detections.plates)

        group = RiderGroup(
            motorcycle=moto,
            persons=persons,
            helmets=all_helmets,
            plates=plates,
        )
        groups.append(group)

        logger.debug(
            f"[Associator] Moto @ ({moto.cx:.0f},{moto.cy:.0f}): "
            f"{len(persons)} riders, {len(all_helmets)} helmets, "
            f"{len(plates)} plates  → violation={group.is_violation}"
        )

    return groups
