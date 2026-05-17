"""
ocr.py
======
License plate OCR using EasyOCR with Indian number plate post-processing.

Indian number plate format:
  [STATE_CODE 2 letters] [DISTRICT_CODE 2 digits] [SERIES 1-2 letters] [NUMBER 4 digits]
  e.g. MH12AB1234, KA05MG6789, DL8CAB1234

Post-processing pipeline:
  1. Run EasyOCR on the cropped plate image
  2. Concatenate all detected text (plates often split across rows)
  3. Clean: remove spaces, lowercase artifacts, normalise O/0 Q/0 I/1 S/5 etc.
  4. Apply regex validation; return raw if nothing matches
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Indian plate regex (lenient) ──────────────────────────────────────────────
# Covers most plate formats:  XX00XX0000  or  XX000000  (old format)
_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
    re.IGNORECASE,
)

# Common OCR character confusions  (wrong → correct)
_CHAR_FIXES = str.maketrans(
    {
        "O": "0",  # The letter O → digit 0  (in the digit positions)
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
        " ": "",
        "-": "",
        ".": "",
        "_": "",
    }
)

# Positional fix table — only apply numeric corrections in digit slots
# (done after initial cleanup, not as a simple translate)


def _clean_plate(raw: str) -> str:
    """
    Remove noise characters and apply common OCR corrections.
    The corrected string is returned for downstream regex validation.
    """
    upper = raw.upper().strip()
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r"[^A-Z0-9]", "", upper)
    return cleaned


def _apply_positional_fix(text: str) -> str:
    """
    Apply character-to-digit (and digit-to-character) corrections based on
    known Indian plate structure:
      pos 0-1:  state code        → letters only
      pos 2-3:  district code     → digits only
      pos 4-5:  series letters    → letters only
      pos 6-9:  vehicle number    → digits only

    This handles cases like 'MH1ZAB1Z34' → 'MH12AB1234'.
    """
    if len(text) < 6:
        return text  # too short to fix safely

    LETTER_FIXES = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"}
    DIGIT_FIXES = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
        "D": "0",
    }

    result = list(text)
    # pos 0,1  → must be letters
    for i in [0, 1]:
        if i < len(result) and result[i] in LETTER_FIXES:
            result[i] = LETTER_FIXES[result[i]]
    # pos 2,3  → must be digits
    for i in [2, 3]:
        if i < len(result) and result[i] in DIGIT_FIXES:
            result[i] = DIGIT_FIXES[result[i]]
    # pos 4,5  → letters
    for i in [4, 5]:
        if i < len(result) and result[i] in LETTER_FIXES:
            result[i] = LETTER_FIXES[result[i]]
    # pos 6 onward → digits
    for i in range(6, len(result)):
        if result[i] in DIGIT_FIXES:
            result[i] = DIGIT_FIXES[result[i]]

    return "".join(result)


def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Enhance a plate crop for better OCR.
    Returns a processed grayscale image.
    """
    # Resize to a standard height while preserving aspect ratio
    h, w = crop.shape[:2]
    target_h = 64
    if h < target_h:
        scale = target_h / h
        new_w = int(w * scale)
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Apply adaptive sharpening instead of harsh Otsu thresholding
    # This preserves edge details which deep learning OCR models prefer over binary pixels.
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return sharpened


class PlateOCR:
    """
    EasyOCR-based license plate reader.
    Loaded once in __init__, stateless read() calls.

    Parameters
    ----------
    gpu      : Use GPU for OCR (set False on CPU-only machines)
    lang     : EasyOCR language list. 'en' is sufficient for Indian plates.
    """

    def __init__(self, gpu: bool = False, lang: list[str] | None = None):
        import easyocr  # deferred import — heavy

        if lang is None:
            lang = ["en"]

        logger.info(f"[PlateOCR] Initialising EasyOCR (gpu={gpu}, lang={lang})")
        try:
            self._reader = easyocr.Reader(lang, gpu=gpu, verbose=False)
            logger.info("[PlateOCR] EasyOCR ready")
        except Exception as e:
            logger.error(f"[PlateOCR] Failed to init EasyOCR: {e}")
            raise

    def read(
        self,
        image: np.ndarray,
        plate_box: tuple[float, float, float, float] | None = None,
    ) -> str:
        """
        Read a license plate from either:
          - a full image + plate_box (x1,y1,x2,y2)
          - a pre-cropped plate image (plate_box=None)

        Returns cleaned plate string, or "" on failure.
        """
        try:
            crop = self._crop(image, plate_box)
            if crop is None or crop.size == 0:
                return ""

            processed = _preprocess_plate_crop(crop)
            # EasyOCR accepts both gray and BGR; pass the processed binary
            raw_results = self._reader.readtext(
                processed,
                detail=0,  # text only
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
                width_ths=0.7,
                mag_ratio=1.5,
            )

            if not raw_results:
                # Retry on original crop if processed fails
                raw_results = self._reader.readtext(
                    crop,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    paragraph=False,
                )

            if not raw_results:
                return ""

            # Join all text fragments (plate may be read in chunks)
            combined = "".join(raw_results).upper()
            cleaned = _clean_plate(combined)
            fixed = _apply_positional_fix(cleaned)

            logger.debug(
                f"[PlateOCR] raw={raw_results!r}  cleaned={cleaned}  fixed={fixed}"
            )

            # Validate against Indian plate pattern
            if _PLATE_PATTERN.match(fixed):
                return fixed

            # If validation fails, still return cleaned text — don't drop it
            # (some plates are damaged, non-standard, or partially visible)
            return cleaned if cleaned else ""

        except Exception as e:
            logger.error(f"[PlateOCR] OCR failed: {e}")
            return ""

    def _crop(
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float] | None,
    ) -> np.ndarray | None:
        if box is None:
            return image

        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        # Clamp and validate
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"[PlateOCR] Degenerate crop box: ({x1},{y1},{x2},{y2})")
            return None

        # Add small padding (5%) to avoid cutting off edge characters
        pad_x = max(2, int((x2 - x1) * 0.05))
        pad_y = max(2, int((y2 - y1) * 0.05))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2]
