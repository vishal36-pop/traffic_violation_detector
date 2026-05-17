"""Traffic violation detection pipeline modules."""

from pipeline.associator import RiderGroup, associate
from pipeline.detector import Detection, Detector, FrameDetections
from pipeline.ocr import PlateOCR

__all__ = [
    "Detector",
    "FrameDetections",
    "Detection",
    "associate",
    "RiderGroup",
    "PlateOCR",
]
