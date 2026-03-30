"""
Vehicle Detail Extraction Rule
================================
Decodes snapshot_b64, crops the car bbox, and sends the crop to the
Gemini Vision API to extract license plate number and car model.

Gemini is called ONCE per unique car. A car is identified by a stable
spatial key (camera + bbox snapped to a 50px grid). Once extracted,
the result is cached in Redis. When the car leaves (bbox disappears
for EVICT_FRAMES consecutive frames) the cache entry is cleared so a
new car parking in the same spot gets extracted fresh.

Per-camera state is persisted in Redis so that Celery workers (separate
processes) share state across frames. The Redis key per camera is:
    ``vehicle_cache:<camera_id>``

Requires:
    GEMINI_API_KEY env variable set to your Google AI Studio key.

If Gemini is unavailable or the key is missing, both fields fall back
to "unreadable" / "unknown".
"""
import base64
import hashlib
import json
import logging
import os
from typing import Any, ClassVar, Dict, List

import cv2
import numpy as np

from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # free-tier model

# Number of consecutive frames a car must be absent before its cache is cleared
EVICT_FRAMES = 5


def _car_key(camera_id: str, bbox: dict) -> str:
    """
    Stable identity key for a car detection.
    Snaps bbox to a 50px grid so minor jitter across frames doesn't
    create duplicate keys for the same parked car.
    """
    raw = (
        f"{camera_id}_"
        f"{int(bbox.get('x1', 0) // 50)}_"
        f"{int(bbox.get('y1', 0) // 50)}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:8]


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


def _crop_to_b64(crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _query_gemini(crop: np.ndarray) -> Dict[str, str]:
    """
    Send the cropped car image to Gemini and extract car_number and car_model.
    Falls back to defaults on any error.
    """
    if not GEMINI_API_KEY:
        logger.warning("[VEHICLE] GEMINI_API_KEY not set — skipping Gemini extraction")
        return {"car_number": "unreadable", "car_model": "unknown"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        image_b64 = _crop_to_b64(crop)

        prompt = (
            "You are a vehicle recognition assistant. "
            "Look at this image of a vehicle and return ONLY a JSON object with two keys:\n"
            '  "car_number": the license plate number as a string (e.g. "MH12AB1234"), '
            'or "unreadable" if not visible.\n'
            '  "car_model": the make and model of the vehicle (e.g. "Toyota Innova"), '
            'or "unknown" if not identifiable.\n'
            "Return only valid JSON, no explanation."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"),
            ],
        )
        text = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return {
            "car_number": str(result.get("car_number", "unreadable")),
            "car_model": str(result.get("car_model", "unknown")),
        }

    except Exception as exc:
        logger.warning("[VEHICLE] Gemini extraction failed: %s", exc)
        return {"car_number": "unreadable", "car_model": "unknown"}


class VehicleExtractionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "vehicle_extraction"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        cars = [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") == "car"
        ]

        snapshot_b64 = detection_output.get("snapshot_b64")
        print(f"[VEHICLE] camera={camera_id} | cars_detected={len(cars)} | snapshot={'yes' if snapshot_b64 else 'no'}")
        image = _decode_image(snapshot_b64) if snapshot_b64 else None
        vehicle_details: List[dict] = []

        # --- Load state from Redis ---
        # cache: { spatial_key: {car_number, car_model, absent_frames} }
        redis_key = f"vehicle_cache:{camera_id}"
        cache: Dict[str, dict] = get_state(redis_key)

        # --- Run existing logic (unchanged) ---
        # Track which keys are seen this frame to evict gone cars
        seen_keys = set()

        for car in cars:
            bbox = car.get("bbox", {})
            key = _car_key(camera_id, bbox)
            seen_keys.add(key)

            if key in cache:
                # Already extracted — return cached result, skip Gemini
                cached = cache[key]
                cached["absent_frames"] = 0
                print(f"[VEHICLE] cache hit: key={key} plate={cached['car_number']} model={cached['car_model']}")
                car_number = cached["car_number"]
                car_model = cached["car_model"]
            else:
                # New car — call Gemini once
                print(f"[VEHICLE] new car detected: key={key} | calling Gemini")
                car_number, car_model = "unreadable", "unknown"
                if image is not None:
                    crop = _crop_bbox(image, bbox)
                    if crop is not None and crop.size > 0:
                        extracted = _query_gemini(crop)
                        car_number = extracted["car_number"]
                        car_model = extracted["car_model"]

                cache[key] = {
                    "car_number": car_number,
                    "car_model": car_model,
                    "absent_frames": 0,
                }
                logger.info(
                    "[VEHICLE] New car extracted: key=%s plate=%s model=%s",
                    key, car_number, car_model,
                )

            vehicle_details.append({
                "track_id": car.get("track_id") or key,
                "car_number": car_number,
                "car_model": car_model,
                "confidence": car.get("confidence", 0.0),
            })

        # Increment absent counter for cars not seen this frame
        for key in list(cache.keys()):
            if key not in seen_keys:
                cache[key]["absent_frames"] += 1
                if cache[key]["absent_frames"] >= EVICT_FRAMES:
                    logger.info("[VEHICLE] Car left, clearing cache for key=%s", key)
                    del cache[key]

        # --- Save updated state back to Redis ---
        set_state(redis_key, cache)

        if not cars:
            print(f"[VEHICLE] no cars — skipping")
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        print(f"[VEHICLE] result: triggered=True | vehicles={[(v['car_number'], v['car_model']) for v in vehicle_details]}")
        return {
            "triggered": True,
            "matched_objects": cars,
            "vehicle_details": vehicle_details,
        }
