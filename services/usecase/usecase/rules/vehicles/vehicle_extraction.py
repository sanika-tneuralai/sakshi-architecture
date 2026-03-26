"""
Vehicle Detail Extraction Rule
================================
Decodes snapshot_b64, crops the car bbox, and attempts OCR for license plate.
car_model extraction is a stub — wire in your LLM or classifier.

Triggered standalone or after a parking_intime event.
"""
import base64
import logging
import threading
from typing import Any, ClassVar, Dict, List

import cv2
import numpy as np

from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)


def _decode_image(snapshot_b64: str):
    try:
        data = base64.b64decode(snapshot_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.error("[VEHICLE] Failed to decode snapshot: %s", exc)
        return None


def _crop_bbox(image: np.ndarray, bbox: dict):
    h, w = image.shape[:2]
    x1 = max(0, int(bbox.get("x1", 0)))
    y1 = max(0, int(bbox.get("y1", 0)))
    x2 = min(w, int(bbox.get("x2", w)))
    y2 = min(h, int(bbox.get("y2", h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _ocr_plate(crop: np.ndarray) -> str:
    """Run pytesseract OCR on the cropped image. Returns empty string if unavailable."""
    try:
        import pytesseract
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )
        return text.strip()
    except ImportError:
        logger.debug("[VEHICLE] pytesseract not installed — plate OCR skipped")
        return ""
    except Exception as exc:
        logger.warning("[VEHICLE] OCR error: %s", exc)
        return ""


def _classify_model(crop: np.ndarray) -> str:
    # TODO: replace with LLM call or ResNet classifier
    return "unknown"


class VehicleExtractionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "vehicle_extraction"
    _lock = threading.Lock()

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        cars = [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") == "car"
        ]
        if not cars:
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        snapshot_b64 = detection_output.get("snapshot_b64")
        image = _decode_image(snapshot_b64) if snapshot_b64 else None
        vehicle_details: List[dict] = []

        for car in cars:
            track_id = car.get("track_id", "unknown")
            plate, model = "", "unknown"

            if image is not None:
                crop = _crop_bbox(image, car["bbox"])
                if crop is not None and crop.size > 0:
                    plate = _ocr_plate(crop)
                    model = _classify_model(crop)

            vehicle_details.append({
                "track_id": track_id,
                "car_number": plate or "unreadable",
                "car_model": model,
                "confidence": car.get("confidence", 0.0),
            })
            logger.info(
                "[VEHICLE] track=%s plate=%s model=%s",
                track_id, plate or "unreadable", model,
            )

        return {
            "triggered": True,
            "matched_objects": cars,
            "vehicle_details": vehicle_details,
        }
