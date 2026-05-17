"""
solution.py
===========
Main entry point for Traffic Rule Violation Detection.

Implements the mandatory interface:

    class TrafficViolationDetector:
        def __init__(self, model_dir="./models"):
            ...
        def predict(self, image_path) -> dict:
            ...

Output format:
    {
        "violations": [
            {
                "num_riders"       : int,
                "helmet_violations": int,
                "license_plate"    : "string"
            }
        ]
    }

Only motorcycles with at least one violation are included in the output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Make project root importable regardless of CWD ───────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline.associator import associate
from pipeline.detector import Detector, Detection, CLS_MOTORCYCLE, CLS_PERSON
from pipeline.ocr import PlateOCR
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = str(_HERE / "models")
DETECTOR_MODEL_NAME = "combined_detector.pt"
CONF_THRESHOLD = 0.10  # Base confidence, filtered later
IOU_THRESHOLD = 0.45  # NMS IoU
INFERENCE_IMGSZ = 1280  # Increased resolution for better small object detection


class TrafficViolationDetector:
    """
    End-to-end traffic violation detector.

    All models are loaded exactly once in __init__.
    predict() is fully stateless — safe to call multiple times in any order.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Load all models.

        Parameters
        ----------
        model_dir : Directory containing:
                      combined_detector.pt   — YOLO for moto/person/helmet/plate
                    (Optional)
                      plate_detector.pt      — standalone plate YOLO (higher accuracy)
        """
        t0 = time.perf_counter()
        model_dir = Path(model_dir)

        # ── Determine device ──────────────────────────────────────────────────
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        self._device = device
        logger.info(f"[TVD] Device: {device}")

        # ── Load main detector ────────────────────────────────────────────────
        main_model = model_dir / DETECTOR_MODEL_NAME
        if not main_model.exists():
            raise FileNotFoundError(
                f"[TVD] Combined detector model not found: {main_model}\n"
                f"  Please train first:  cd training && python train.py --task combined\n"
                f"  Or place pre-trained weights at: {main_model}"
            )

        self._detector = Detector(
            model_path=main_model,
            conf_thresh=CONF_THRESHOLD,
            iou_thresh=IOU_THRESHOLD,
            imgsz=INFERENCE_IMGSZ,
            device=device,
        )

        # ── Load base model for motorcycles ───────────────────────────────────
        base_model_path = Path(model_dir) / "yolo11s.pt"
        if not base_model_path.exists():
            logger.warning(f"[TVD] Base model not found at {base_model_path}. Motorcycle detection might fail.")
            self._base_model = None
        else:
            logger.info("[TVD] Loading base model for motorcycles")
            self._base_model = YOLO(str(base_model_path))

        # ── Optional standalone plate detector ────────────────────────────────
        plate_model = model_dir / "plate_detector.pt"
        if plate_model.exists():
            logger.info("[TVD] Loading standalone plate detector")
            self._plate_detector = Detector(
                model_path=plate_model,
                conf_thresh=0.40,
                iou_thresh=0.45,
                imgsz=320,  # plates are small, 320 is enough
                device=device,
            )
        else:
            logger.info(
                "[TVD] No standalone plate detector; using combined model plates"
            )
            self._plate_detector = None

        # ── Load OCR ──────────────────────────────────────────────────────────
        ocr_gpu = device != "cpu"
        self._ocr = PlateOCR(gpu=ocr_gpu)

        elapsed = time.perf_counter() - t0
        logger.info(f"[TVD] All models loaded in {elapsed:.1f}s")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict[str, Any]:
        """
        Detect traffic violations in a single image.

        Parameters
        ----------
        image_path : Absolute or relative path to an RGB/BGR JPEG or PNG image.

        Returns
        -------
        dict with key "violations", a list of dicts each containing:
            num_riders        : int  — total riders detected on this motorcycle
            helmet_violations : int  — riders without helmets
            license_plate     : str  — OCR result, "" if not readable
        """
        # ── 1. Load image ─────────────────────────────────────────────────────
        image = self._load_image(image_path)
        if image is None:
            logger.error(f"[TVD] Cannot load image: {image_path}")
            return {"violations": []}

        # ── 2. Detect all objects ─────────────────────────────────────────────
        # Custom model detects persons, helmets, plates
        detections = self._detector.detect(image)
        
        # Exclusively rely on the base model for persons
        detections.persons.clear()
        # Strict confidence threshold for helmets and plates to prevent hallucinations
        detections.helmets = [h for h in detections.helmets if h.conf >= 0.40]
        detections.plates = [p for p in detections.plates if p.conf >= 0.40]

        # Base model detects motorcycles and persons
        if self._base_model is not None:
            base_results = self._base_model.predict(image, imgsz=INFERENCE_IMGSZ, conf=0.10, iou=0.45, verbose=False)
            for r in base_results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    if cls_ids[i] == 3:  # COCO motorcycle
                        if float(confs[i]) < 0.40:
                            continue
                        x1, y1, x2, y2 = boxes[i]
                        detections.motorcycles.append(Detection(
                            cls_id=CLS_MOTORCYCLE, conf=float(confs[i]),
                            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                        ))
                    elif cls_ids[i] == 0:  # COCO person
                        x1, y1, x2, y2 = boxes[i]
                        detections.persons.append(Detection(
                            cls_id=CLS_PERSON, conf=float(confs[i]),
                            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                        ))

        # If standalone plate detector exists, boost plate detections
        if self._plate_detector is not None:
            plate_dets = self._plate_detector.detect(image)
            # Merge plates that are not already detected
            existing_plate_boxes = {(p.x1, p.y1, p.x2, p.y2) for p in detections.plates}
            for plate in plate_dets.plates:
                key = (plate.x1, plate.y1, plate.x2, plate.y2)
                if key not in existing_plate_boxes:
                    detections.plates.append(plate)

        # ── 3. Associate objects per motorcycle ───────────────────────────────
        rider_groups = associate(detections)

        # ── 4. Filter to violations only ──────────────────────────────────────
        violations = []
        for group in rider_groups:
            if not group.is_violation:
                continue

            # ── 5. OCR for violating bikes only ───────────────────────────────
            plate_text = ""
            if group.best_plate is not None:
                plate_text = self._ocr.read(image, group.best_plate.bbox)

            violations.append(
                {
                    "num_riders": group.num_riders,
                    "helmet_violations": group.helmet_violations,
                    "license_plate": plate_text,
                }
            )
            logger.info(
                f"[TVD] VIOLATION → riders={group.num_riders}, "
                f"no_helmet={group.helmet_violations}, plate={plate_text!r}"
            )

        result = {"violations": violations}
        logger.info(f"[TVD] Result: {json.dumps(result)}")
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray | None:
        """
        Load image from disk robustly.
        Handles spaces in path, non-ASCII filenames (Windows), and
        returns BGR uint8 array as expected by OpenCV/YOLO.
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"[TVD] File not found: {path}")
            return None

        # cv2.imread fails on non-ASCII paths on Windows → use np.fromfile
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("imdecode returned None")
            logger.debug(f"[TVD] Loaded image {path.name}  shape={img.shape}")
            return img
        except Exception as e:
            logger.error(f"[TVD] Failed to load {path}: {e}")
            return None


# ── Standalone CLI ─────────────────────────────────────────────────────────────


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Violation Detector CLI")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model_dir", default=DEFAULT_MODEL_DIR, help="Directory with model weights"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    args = parser.parse_args()

    detector = TrafficViolationDetector(model_dir=args.model_dir)
    result = detector.predict(args.image)
    indent = 4 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    _cli()
