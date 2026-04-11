"""
Vehicle Detail Extraction Rule
================================
Downloads the frame snapshot from S3, crops the car bbox, and calls the
Gemini Vision API to extract license plate number and car model.

Design (locked):
- Extraction is slot-anchored — Gemini is called AT MOST ONCE per car
  lifecycle. "Once per lifecycle" is enforced via slot.extracted in the shared
  slot state (``slot:{camera_id}:{slot_id}``).
- The spatial-key / 200px-grid cache is removed. The slot lifecycle is the
  cache boundary: extraction attempts stop as soon as slot.extracted=True and
  restart only when the slot resets (car exits, reset_slot_state is called).
- Retry schedule (frame-based, using slot.frame_counter):
    Attempt 1: immediately when slot becomes occupied (backoff_until = frame + 10)
    Attempt 2: frame_counter >= backoff_until (backoff_until = frame + 20)
    Attempt 3: frame_counter >= backoff_until (backoff_until = frame + 9999)
  After attempt 3: extracted=True, fields stay null — no further calls.
- Success condition: at least one of car_number / car_model is not
  "unreadable" / "unknown". Partial success counts.
- Blind-spot case: after 3 failures (~30 frames / ~30 s), extraction stops
  permanently for this car. Session records null plate/model.

Per-camera tracker state continues to come from parking_detection via
``detection_output["tracked_cars"]``.
"""
import base64
import json
import logging
import os
from typing import Any, ClassVar, Dict, List

import boto3
import cv2
import numpy as np

from shared.common.roi import which_rois
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_slot_state, set_slot_state

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Backoff increments (in frames) after each failed attempt
_BACKOFF = [10, 20, 9999]

DETECTION_W = int(os.getenv("DETECTION_WIDTH",  "1920"))
DETECTION_H = int(os.getenv("DETECTION_HEIGHT", "1080"))


# ---------------------------------------------------------------------------
# Image utilities (unchanged from previous version)
# ---------------------------------------------------------------------------

def _download_image(snapshot_url: str):
    """Download a frame from a private S3 URL using boto3 (authenticated)."""
    try:
        url_path = snapshot_url.split(".amazonaws.com/", 1)
        if len(url_path) != 2:
            raise ValueError(f"Unrecognised S3 URL format: {snapshot_url}")
        key    = url_path[1]
        bucket = snapshot_url.split("//")[1].split(".s3.")[0]

        logger.info("[VEHICLE] Downloading S3 image: bucket=%s key=%s", bucket, key)
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        logger.info(
            "[VEHICLE] AWS credentials present: access_key=%s secret=%s",
            "yes" if aws_key else "NO",
            "yes" if os.getenv("AWS_SECRET_ACCESS_KEY") else "NO",
        )
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
            aws_access_key_id=aws_key,
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        logger.info("[VEHICLE] S3 download OK: %d bytes", len(data))
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("[VEHICLE] cv2.imdecode returned None — image data may be corrupt")
        else:
            logger.info("[VEHICLE] Image decoded: shape=%s", img.shape)
        return img
    except Exception as exc:
        logger.error("[VEHICLE] Failed to download snapshot from %s: %s", snapshot_url, exc)
        return None


def _crop_bbox(image: np.ndarray, bbox: dict):
    h, w = image.shape[:2]
    sx = w / DETECTION_W
    sy = h / DETECTION_H
    x1 = max(0, int(bbox.get("x1", 0) * sx))
    y1 = max(0, int(bbox.get("y1", 0) * sy))
    x2 = min(w, int(bbox.get("x2", w) * sx))
    y2 = min(h, int(bbox.get("y2", h) * sy))
    logger.info(
        "[VEHICLE] Scaled bbox: x1=%d y1=%d x2=%d y2=%d (image=%dx%d detection=%dx%d)",
        x1, y1, x2, y2, w, h, DETECTION_W, DETECTION_H,
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _crop_to_b64(crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _query_gemini(crop: np.ndarray) -> Dict[str, str]:
    """Send cropped car image to Gemini. Returns car_number + car_model."""
    if not GEMINI_API_KEY:
        logger.warning("[VEHICLE] GEMINI_API_KEY not set — skipping Gemini extraction")
        return {"car_number": "unreadable", "car_model": "unknown"}

    logger.info("[VEHICLE] Gemini API key present (len=%d), model=%s", len(GEMINI_API_KEY), GEMINI_MODEL)
    logger.info("[VEHICLE] Crop shape before Gemini call: %s", crop.shape)

    try:
        from google import genai
        from google.genai import types

        client     = genai.Client(api_key=GEMINI_API_KEY)
        image_b64  = _crop_to_b64(crop)
        logger.info("[VEHICLE] Crop encoded to base64: %d chars", len(image_b64))

        prompt = (
            "You are a vehicle recognition assistant. "
            "Look at this image of a vehicle and return ONLY a JSON object with two keys:\n"
            '  "car_number": the license plate number exactly as it appears in the image (e.g. "MH12AB1234"). '
            'If the plate is not clearly visible, partially obscured, blurry, or you are not 100% certain, '
            'you MUST return "unreadable". Do NOT guess or infer — only return a plate you can directly read.\n'
            '  "car_model": the make and model of the vehicle (e.g. "Toyota Innova"). '
            'If you cannot clearly identify it, return "unknown". Do NOT guess.\n'
            "Return only valid JSON, no explanation."
        )

        logger.info("[VEHICLE] Sending request to Gemini...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"),
            ],
        )
        text = response.text.strip()
        logger.info("[VEHICLE] Gemini raw response: %s", text)

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            logger.info("[VEHICLE] Gemini response after stripping fences: %s", text)

        result = json.loads(text)
        logger.info("[VEHICLE] Gemini parsed result: %s", result)
        return {
            "car_number": str(result.get("car_number", "unreadable")),
            "car_model":  str(result.get("car_model",  "unknown")),
        }

    except Exception as exc:
        logger.warning("[VEHICLE] Gemini extraction failed: %s", exc, exc_info=True)
        return {"car_number": "unreadable", "car_model": "unknown"}


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

class VehicleExtractionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "vehicle_extraction"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id    = detection_output.get("camera_id", "unknown")
        rois         = detection_output.get("rois") or {}
        snapshot_url = detection_output.get("snapshot_url")

        # Tracked cars from parking_detection (injected into slim payload by engine)
        tracked_cars = detection_output.get("tracked_cars") or [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") == "car"
        ]

        print(f"[VEHICLE] camera={camera_id} | tracked_cars={len(tracked_cars)} | snapshot={'yes' if snapshot_url else 'no'}")
        logger.info("[VEHICLE] snapshot_url=%s", snapshot_url)

        if not tracked_cars:
            print("[VEHICLE] no cars — skipping")
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        # Download frame once for all cars (lazy — only if needed below)
        image      = None
        image_tried = False

        vehicle_details: List[dict] = []

        for car in tracked_cars:
            bbox     = car.get("bbox", {})
            track_id = car.get("track_id")

            # Resolve slot_id for this car
            matched_rois = which_rois(bbox, rois) if rois else []
            slot_id      = matched_rois[0] if matched_rois else None

            if not slot_id:
                # Car not in any defined ROI — skip
                continue

            slot = get_slot_state(camera_id, slot_id)

            # ── Trigger check ─────────────────────────────────────────────
            # Conditions from design (all must be true to call Gemini):
            #   1. slot is occupied
            #   2. not yet extracted
            #   3. attempts < 3
            #   4. frame_counter >= backoff_until
            should_extract = (
                slot["occupied"]
                and not slot["extracted"]
                and slot["extraction_attempts"] < 3
                and slot["frame_counter"] >= slot["extraction_backoff_until"]
            )

            if should_extract:
                attempt_number = slot["extraction_attempts"] + 1
                print(f"[VEHICLE] slot={slot_id} attempt={attempt_number} | calling Gemini")

                # Set backoff before the call so a crash/exception still counts the attempt
                backoff_increment = _BACKOFF[slot["extraction_attempts"]]  # 10, 20, or 9999
                slot["extraction_attempts"]     += 1
                slot["extraction_backoff_until"] = slot["frame_counter"] + backoff_increment

                car_number, car_model = "unreadable", "unknown"

                # Download image lazily — only on first extraction attempt this frame
                if not image_tried:
                    image      = _download_image(snapshot_url) if snapshot_url else None
                    image_tried = True
                    logger.info("[VEHICLE] image download result: %s", "OK" if image is not None else "FAILED/None")

                if image is not None:
                    crop = _crop_bbox(image, bbox)
                    if crop is None or crop.size == 0:
                        logger.warning(
                            "[VEHICLE] Skipping Gemini — crop is empty for bbox=%s image_shape=%s",
                            bbox, image.shape,
                        )
                    else:
                        logger.info("[VEHICLE] Crop OK: shape=%s | calling Gemini", crop.shape)
                        extracted  = _query_gemini(crop)
                        car_number = extracted["car_number"]
                        car_model  = extracted["car_model"]
                        logger.info(
                            "[VEHICLE] Gemini result: plate=%s model=%s", car_number, car_model
                        )
                else:
                    logger.warning("[VEHICLE] Skipping Gemini — image is None")

                # Success: at least one field is not the fallback value
                success = not (car_number == "unreadable" and car_model == "unknown")

                if success:
                    slot["extracted"]   = True
                    slot["car_number"]  = car_number
                    slot["car_model"]   = car_model
                    logger.info(
                        "[VEHICLE] Extraction success: slot=%s plate=%s model=%s",
                        slot_id, car_number, car_model,
                    )
                elif slot["extraction_attempts"] >= 3:
                    # 3 failures — stop permanently for this lifecycle
                    slot["extracted"] = True
                    logger.warning(
                        "[VEHICLE] 3 attempts exhausted for slot=%s — marking extracted, fields null",
                        slot_id,
                    )
                else:
                    logger.warning(
                        "[VEHICLE] Attempt %d failed for slot=%s — will retry at frame %d",
                        attempt_number, slot_id, slot["extraction_backoff_until"],
                    )

                set_slot_state(camera_id, slot_id, slot)

            # Always include in vehicle_details using whatever is stored in slot
            # (may be null on first attempt; will be populated after success)
            vehicle_details.append({
                "track_id":  track_id,
                "slot_id":   slot_id,
                "car_number": slot["car_number"],
                "car_model":  slot["car_model"],
                "confidence": car.get("confidence", 0.0),
            })

        if not vehicle_details:
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        print(
            f"[VEHICLE] result: triggered=True | vehicles="
            f"{[(v['slot_id'], v['car_number'], v['car_model']) for v in vehicle_details]}"
        )
        return {
            "triggered":      True,
            "matched_objects": tracked_cars,
            "vehicle_details": vehicle_details,
        }
