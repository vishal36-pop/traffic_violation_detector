"""
visualize.py
============
Debug visualization utilities. Draw detections and associations on an image.
NOT used during inference — only for development and debugging.

Usage:
  python utils/visualize.py --image test.jpg --model_dir ./models --out debug.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline.associator import RiderGroup
from pipeline.detector import Detection, FrameDetections

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "motorcycle": (0, 165, 255),  # orange
    "person": (0, 255, 0),  # green
    "helmet": (255, 200, 0),  # gold
    "license_plate": (255, 0, 0),  # blue
    "violation": (0, 0, 255),  # red
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2


def _draw_box(
    img: np.ndarray,
    det: Detection,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 1, y1 - 2),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_detections(
    image: np.ndarray,
    detections: FrameDetections,
) -> np.ndarray:
    """Draw raw detections without association."""
    img = image.copy()
    for det in detections.motorcycles:
        _draw_box(img, det, COLORS["motorcycle"], f"moto {det.conf:.2f}")
    for det in detections.persons:
        _draw_box(img, det, COLORS["person"], f"person {det.conf:.2f}")
    for det in detections.helmets:
        _draw_box(img, det, COLORS["helmet"], f"helmet {det.conf:.2f}")
    for det in detections.plates:
        _draw_box(img, det, COLORS["license_plate"], f"plate {det.conf:.2f}")
    return img


def draw_groups(
    image: np.ndarray,
    rider_groups: list[RiderGroup],
    ocr_results: dict[int, str] | None = None,  # group_idx → plate text
) -> np.ndarray:
    """Draw association groups. Red border = violation."""
    img = image.copy()
    if ocr_results is None:
        ocr_results = {}

    for idx, group in enumerate(rider_groups):
        moto_color = COLORS["violation"] if group.is_violation else COLORS["motorcycle"]
        plate_text = ocr_results.get(idx, "")

        # Motorcycle box
        label = f"#{idx} riders:{group.num_riders} nohelmet:{group.helmet_violations}"
        if plate_text:
            label += f" [{plate_text}]"
        _draw_box(img, group.motorcycle, moto_color, label)

        # Person boxes
        for person in group.persons:
            _draw_box(img, person, COLORS["person"], "")

        # Helmet boxes
        for helmet in group.helmets:
            _draw_box(img, helmet, COLORS["helmet"], "")

        # Plate boxes
        for plate in group.plates:
            _draw_box(
                img,
                plate,
                COLORS["license_plate"],
                plate_text if plate_text else "plate",
            )

    return img


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualise detections")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_dir", default=str(_HERE / "models"))
    parser.add_argument("--out", default="debug_output.jpg")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Import here to avoid circular deps
    from pipeline.associator import associate
    from pipeline.detector import Detector
    from pipeline.ocr import PlateOCR

    MODELS = Path(args.model_dir)
    img = cv2.imread(args.image)
    if img is None:
        buf = np.fromfile(args.image, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: Cannot load {args.image}")
        return

    detector = Detector(
        model_path=MODELS / "combined_detector.pt",
        conf_thresh=0.25,
        imgsz=640,
    )
    ocr = PlateOCR(gpu=False)

    dets = detector.detect(img)
    groups = associate(dets)

    ocr_map: dict[int, str] = {}
    for i, g in enumerate(groups):
        if g.is_violation and g.best_plate:
            ocr_map[i] = ocr.read(img, g.best_plate.bbox)

    vis = draw_groups(img, groups, ocr_map)

    cv2.imwrite(args.out, vis)
    print(f"Saved → {args.out}")
    if args.show:
        cv2.imshow("Violations", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
