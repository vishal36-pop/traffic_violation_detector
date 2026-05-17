"""
detector.py
===========
Thin wrapper around Ultralytics YOLO that runs inference and returns
structured detection results as plain dataclasses.

Class IDs (MUST match combined_dataset.yaml):
  0 = motorcycle
  1 = person
  2 = helmet
  3 = license_plate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Class ID constants ────────────────────────────────────────────────────────
CLS_MOTORCYCLE = 0
CLS_PERSON = 1
CLS_HELMET = 2
CLS_LICENSE = 3
CLASS_NAMES = {0: "motorcycle", 1: "person", 2: "helmet", 3: "license_plate"}


@dataclass
class Detection:
    """Single bounding box detection."""

    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    # ── Derived geometry ──────────────────────────────────────────────────────
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def bottom_cx(self) -> float:
        return self.cx

    @property
    def bottom_cy(self) -> float:
        return self.y2

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def iou(self, other: "Detection") -> float:
        """Intersection over Union with another detection."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / (self.area + other.area - inter)

    def intersection_area(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def overlap_ratio(self, other: "Detection") -> float:
        """What fraction of self's area overlaps with other."""
        inter = self.intersection_area(other)
        if self.area == 0:
            return 0.0
        return inter / self.area

    def name(self) -> str:
        return CLASS_NAMES.get(self.cls_id, str(self.cls_id))


@dataclass
class FrameDetections:
    """All detections for a single image."""

    motorcycles: list[Detection] = field(default_factory=list)
    persons: list[Detection] = field(default_factory=list)
    helmets: list[Detection] = field(default_factory=list)
    plates: list[Detection] = field(default_factory=list)

    def all_detections(self) -> list[Detection]:
        return self.motorcycles + self.persons + self.helmets + self.plates


class Detector:
    """
    Loads YOLO model once in __init__ and exposes a stateless detect() call.

    Parameters
    ----------
    model_path   : Path to .pt weights file
    conf_thresh  : Minimum confidence threshold
    iou_thresh   : NMS IoU threshold
    imgsz        : Inference resolution (use 640 for combined, 320 for fast)
    device       : "cpu", "0", "cuda:0", etc.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        from ultralytics import YOLO  # import here so detector.py is importable
        # even before ultralytics is installed

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[Detector] Model weights not found: {model_path}\n"
                f"  Run training first:  python training/train.py --task combined"
            )

        logger.info(f"[Detector] Loading {model_path.name} on {device}")
        self._model = YOLO(str(model_path))
        self._conf = conf_thresh
        self._iou = iou_thresh
        self._imgsz = imgsz
        self._device = device

        # Warm-up inference to pre-allocate GPU memory / JIT compile
        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
            logger.info("[Detector] Warm-up complete")
        except Exception as e:
            logger.warning(f"[Detector] Warm-up failed (non-fatal): {e}")

    def detect(self, image: np.ndarray) -> FrameDetections:
        """
        Run inference on a single BGR or RGB uint8 image array.

        Parameters
        ----------
        image : H x W x 3 numpy array (BGR from cv2.imread or RGB from PIL)

        Returns
        -------
        FrameDetections with separated lists per class
        """
        if image is None or image.size == 0:
            logger.warning("[Detector] Empty image passed to detect()")
            return FrameDetections()

        try:
            results = self._model.predict(
                image,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"[Detector] Inference failed: {e}")
            return FrameDetections()

        return self._parse_results(results[0])

    def _parse_results(self, result) -> FrameDetections:
        fd = FrameDetections()

        if result.boxes is None or len(result.boxes) == 0:
            return fd

        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] x1y1x2y2 in pixels
        confs = result.boxes.conf.cpu().numpy()  # [N]
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            det = Detection(
                cls_id=cls_ids[i],
                conf=float(confs[i]),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
            if det.area < 4:  # degenerate box guard
                continue
            if det.cls_id == CLS_MOTORCYCLE:
                fd.motorcycles.append(det)
            elif det.cls_id == CLS_PERSON:
                fd.persons.append(det)
            elif det.cls_id == CLS_HELMET:
                fd.helmets.append(det)
            elif det.cls_id == CLS_LICENSE:
                fd.plates.append(det)

        logger.debug(
            f"[Detector] Detected: {len(fd.motorcycles)} bikes, "
            f"{len(fd.persons)} persons, {len(fd.helmets)} helmets, "
            f"{len(fd.plates)} plates"
        )
        return fd
