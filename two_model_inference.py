import sys
import cv2
import argparse
import logging
from pathlib import Path

# Add the project root to sys.path so we can import from pipeline
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline.detector import Detection, FrameDetections, CLS_MOTORCYCLE, CLS_PERSON, CLS_HELMET, CLS_LICENSE
from pipeline.associator import associate
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Colors for visualization
COLOR_MOTO = (0, 255, 255)   # Yellow
COLOR_PERSON = (0, 0, 255)   # Red
COLOR_HELMET = (0, 255, 0)   # Green
COLOR_PLATE = (255, 0, 0)    # Blue

class TwoModelDetector:
    def __init__(self, base_model_path: str, custom_model_path: str):
        if not Path(base_model_path).exists():
            raise FileNotFoundError(f"Base model not found at {base_model_path}")
        if not Path(custom_model_path).exists():
            raise FileNotFoundError(f"Custom model not found at {custom_model_path}")

        logger.info(f"Loading Base Model for bikes: {base_model_path}")
        self.base_model = YOLO(base_model_path)
        
        logger.info(f"Loading Custom Model for persons/helmets/plates: {custom_model_path}")
        self.custom_model = YOLO(custom_model_path)
        
        try:
            import easyocr
            logger.info("Initializing EasyOCR for license plates...")
            self.reader = easyocr.Reader(['en'], gpu=True)
        except ImportError:
            logger.warning("EasyOCR not installed. Plate OCR will be disabled. Run `pip install easyocr`.")
            self.reader = None

    def process_frame(self, img) -> tuple[list, FrameDetections]:
        """Runs both models on a frame and returns associated rider groups and raw detections."""
        fd = FrameDetections()

        # 1. Base Model -> Motorcycles (COCO 3) and Persons (COCO 0)
        base_results = self.base_model.predict(img, imgsz=1280, conf=0.10, iou=0.45, verbose=False)
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
                    fd.motorcycles.append(Detection(
                        cls_id=CLS_MOTORCYCLE, conf=float(confs[i]),
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                    ))
                elif cls_ids[i] == 0:  # COCO person
                    x1, y1, x2, y2 = boxes[i]
                    fd.persons.append(Detection(
                        cls_id=CLS_PERSON, conf=float(confs[i]),
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                    ))

        # 2. Custom Model -> Helmets and Plates exclusively
        custom_results = self.custom_model.predict(img, imgsz=1280, conf=0.10, iou=0.45, verbose=False)
        for r in custom_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                if float(confs[i]) < 0.40:  # Prevent hallucinations
                    continue
                cls_id = cls_ids[i]
                x1, y1, x2, y2 = boxes[i]
                det = Detection(
                    cls_id=cls_id, conf=float(confs[i]),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                )
                if cls_id == CLS_HELMET:
                    fd.helmets.append(det)
                elif cls_id == CLS_LICENSE:
                    fd.plates.append(det)

        # 3. Associate detections geometrically
        rider_groups = associate(fd)
        
        # 4. Perform OCR on the best plate of each rider group
        if self.reader is not None:
            for group in rider_groups:
                best_plate = group.best_plate
                if best_plate:
                    x1, y1, x2, y2 = map(int, [best_plate.x1, best_plate.y1, best_plate.x2, best_plate.y2])
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        plate_crop = img[y1:y2, x1:x2]
                        # Read text using EasyOCR
                        results = self.reader.readtext(plate_crop, detail=0)
                        text = " ".join(results).strip()
                        if text:
                            # Attach the text dynamically
                            best_plate.ocr_text = text

        return rider_groups, fd

    def annotate_frame(self, img, rider_groups, fd):
        """Draws bounding boxes and stats on the image."""
        # Draw motorcycles and their associated objects
        for group in rider_groups:
            moto = group.motorcycle
            cv2.rectangle(img, (int(moto.x1), int(moto.y1)), (int(moto.x2), int(moto.y2)), COLOR_MOTO, 2)
            
            text = f"Bike: {group.num_riders}p, {group.helmet_count}h"
            cv2.putText(img, text, (int(moto.x1), int(moto.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MOTO, 2)
            
            if group.is_violation:
                cv2.putText(img, "VIOLATION!", (int(moto.x1), int(moto.y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for p in group.persons:
                cv2.rectangle(img, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)), COLOR_PERSON, 2)
            for h in group.helmets:
                cv2.rectangle(img, (int(h.x1), int(h.y1)), (int(h.x2), int(h.y2)), COLOR_HELMET, 2)
            for pl in group.plates:
                cv2.rectangle(img, (int(pl.x1), int(pl.y1)), (int(pl.x2), int(pl.y2)), COLOR_PLATE, 2)
                # Draw OCR text if available
                if hasattr(pl, 'ocr_text') and pl.ocr_text:
                    cv2.putText(img, pl.ocr_text, (int(pl.x1), int(pl.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PLATE, 2)

        # Draw unassociated pedestrians faintly
        claimed_persons = set(id(p) for group in rider_groups for p in group.persons)
        for p in fd.persons:
            if id(p) not in claimed_persons:
                cv2.rectangle(img, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)), (128, 128, 128), 1)
        
        return img


def process_image(detector: TwoModelDetector, input_path: str, output_path: str):
    logger.info(f"Processing image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        logger.error(f"Failed to read image: {input_path}")
        return

    rider_groups, fd = detector.process_frame(img)
    logger.info(f"Detected {len(rider_groups)} motorcycles.")
    
    for i, group in enumerate(rider_groups):
        logger.info(f"  Bike {i+1}: {group.num_riders} persons, {group.helmet_count} helmets. Violation: {group.is_violation}")

    out_img = detector.annotate_frame(img, rider_groups, fd)
    cv2.imwrite(output_path, out_img)
    logger.info(f"Saved annotated image to {output_path}")


def process_video(detector: TwoModelDetector, input_path: str, output_path: str):
    logger.info(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rider_groups, fd = detector.process_frame(frame)
        out_frame = detector.annotate_frame(frame, rider_groups, fd)
        out.write(out_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    logger.info(f"Saved annotated video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Two-Model Inference for Traffic Violations")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--base-model", type=str, default=str(_HERE / "yolo11s.pt"), help="Path to base YOLO model (detects bikes)")
    # Using the newly trained combined model best.pt
    parser.add_argument("--custom-model", type=str, default=str(_HERE.parent / "runs" / "detect" / "models" / "combined_detector_yolo11" / "weights" / "best.pt"), help="Path to custom model (detects person/helmet/plate)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output (use .mp4 for video)")
    
    args = parser.parse_args()

    try:
        detector = TwoModelDetector(args.base_model, args.custom_model)
        
        # Check if video or image
        ext = Path(args.source).suffix.lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            process_video(detector, args.source, args.output)
        else:
            process_image(detector, args.source, args.output)
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
