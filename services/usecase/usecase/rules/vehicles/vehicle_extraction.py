"""
Vehicle Detail Extraction Rule
================================
Downloads the frame snapshot from S3, crops the car bbox, and calls an
LLM Vision API (OpenAI or Gemini) to extract license plate number and car model.

Provider selection (env-driven):
- If OPENAI_API_KEY is set → uses OpenAI (gpt-4o by default, override with OPENAI_MODEL)
- Otherwise falls back to Gemini (gemini-2.5-flash by default, override with GEMINI_MODEL)

Design (locked):
- Extraction is slot-anchored — the LLM is called AT MOST ONCE per car
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

# Backoff increments (in frames) after each failed attempt
_BACKOFF = [10, 20, 9999]
_MODEL_RETRY_DELAY = int(os.getenv("VEHICLE_MODEL_RETRY_DELAY_FRAMES", "45"))
_MIN_CROP_W = int(os.getenv("VEHICLE_MIN_CROP_W", "140"))
_MIN_CROP_H = int(os.getenv("VEHICLE_MIN_CROP_H", "90"))
_MIN_SHARPNESS = float(os.getenv("VEHICLE_MIN_CROP_SHARPNESS", "60.0"))

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


def _is_crop_quality_ok(crop: np.ndarray) -> bool:
    """
    Cheap quality gate to avoid wasting LLM calls on tiny/blurry crops.
    """
    h, w = crop.shape[:2]
    if w < _MIN_CROP_W or h < _MIN_CROP_H:
        logger.info(
            "[VEHICLE] Crop rejected by size gate: w=%d h=%d (min=%dx%d)",
            w, h, _MIN_CROP_W, _MIN_CROP_H,
        )
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < _MIN_SHARPNESS:
        logger.info(
            "[VEHICLE] Crop rejected by sharpness gate: %.2f < %.2f",
            sharpness, _MIN_SHARPNESS,
        )
        return False
    return True


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

        client    = genai.Client(api_key=GEMINI_API_KEY)
        image_b64 = _crop_to_b64(crop)
        logger.info("[VEHICLE] Crop encoded to base64: %d chars", len(image_b64))

        prompt = (
            "You are a vehicle recognition system. Analyse this image.\n\n"
            "RULES — follow exactly, violations cause system errors:\n"
            '1. "car_number": Copy the license plate characters EXACTLY as printed on the plate. '
            "Every character must be directly visible and legible in the image. "
            'If ANY character is unclear, obscured, blurry, cut off, or you have ANY doubt, '
            'return the exact string "unreadable". NEVER guess, infer, autocomplete, or construct '
            "a plausible-looking plate — return only what you can literally read pixel-by-pixel.\n"
            '2. "car_model": State the make and model only if you are highly confident (e.g. "Toyota Innova"). '
            'If the model is unclear or you are guessing, return the exact string "unknown".\n\n'
            "Return ONLY a JSON object with these two keys. No explanation, no markdown."
        )

        response_schema = {
            "type": "object",
            "properties": {
                "car_number": {"type": "string"},
                "car_model":  {"type": "string"},
            },
            "required": ["car_number", "car_model"],
        }

        logger.info("[VEHICLE] Sending request to Gemini...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0,
            ),
        )
        text = response.text.strip()
        logger.info("[VEHICLE] Gemini raw response: %s", text)

        result = json.loads(text)
        car_number = str(result.get("car_number", "unreadable")).strip()
        car_model  = str(result.get("car_model",  "unknown")).strip()

        # Reject suspiciously short plates (real plates have ≥4 chars)
        if car_number != "unreadable" and len(car_number.replace(" ", "")) < 4:
            logger.warning(
                "[VEHICLE] Plate '%s' rejected — too short to be real, marking unreadable", car_number
            )
            car_number = "unreadable"

        logger.info("[VEHICLE] Gemini parsed result: car_number=%s car_model=%s", car_number, car_model)
        return {"car_number": car_number, "car_model": car_model}

    except Exception as exc:
        logger.warning("[VEHICLE] Gemini extraction failed: %s", exc, exc_info=True)
        return {"car_number": "unreadable", "car_model": "unknown"}


def _query_openai(crop: np.ndarray) -> Dict[str, str]:
    """Send cropped car image to OpenAI vision. Returns car_number + car_model."""
    if not OPENAI_API_KEY:
        logger.warning("[VEHICLE] OPENAI_API_KEY not set — skipping OpenAI extraction")
        return {"car_number": "unreadable", "car_model": "unknown"}

    logger.info("[VEHICLE] OpenAI API key present (len=%d), model=%s", len(OPENAI_API_KEY), OPENAI_MODEL)
    logger.info("[VEHICLE] Crop shape before OpenAI call: %s", crop.shape)

    try:
        from openai import OpenAI

        client    = OpenAI(api_key=OPENAI_API_KEY)
        image_b64 = _crop_to_b64(crop)
        logger.info("[VEHICLE] Crop encoded to base64: %d chars", len(image_b64))

        prompt = (
            "You are a vehicle recognition system. Analyse this image.\n\n"
            "RULES — follow exactly, violations cause system errors:\n"
            '1. "car_number": Copy the license plate characters EXACTLY as printed on the plate. '
            "Every character must be directly visible and legible in the image. "
            'If ANY character is unclear, obscured, blurry, cut off, or you have ANY doubt, '
            'return the exact string "unreadable". NEVER guess, infer, autocomplete, or construct '
            "a plausible-looking plate — return only what you can literally read pixel-by-pixel.\n"
            '2. "car_model": State the make and model only if you are highly confident (e.g. "Toyota Innova"). '
            'If the model is unclear or you are guessing, return the exact string "unknown".\n\n'
            "Return ONLY a JSON object with exactly two keys: car_number and car_model. No explanation, no markdown."
        )

        logger.info("[VEHICLE] Sending request to OpenAI...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content.strip()
        logger.info("[VEHICLE] OpenAI raw response: %s", text)

        result     = json.loads(text)
        car_number = str(result.get("car_number", "unreadable")).strip()
        car_model  = str(result.get("car_model",  "unknown")).strip()

        # Reject suspiciously short plates (real plates have ≥4 chars)
        if car_number != "unreadable" and len(car_number.replace(" ", "")) < 4:
            logger.warning(
                "[VEHICLE] Plate '%s' rejected — too short to be real, marking unreadable", car_number
            )
            car_number = "unreadable"

        logger.info("[VEHICLE] OpenAI parsed result: car_number=%s car_model=%s", car_number, car_model)
        return {"car_number": car_number, "car_model": car_model}

    except Exception as exc:
        logger.warning("[VEHICLE] OpenAI extraction failed: %s", exc, exc_info=True)
        return {"car_number": "unreadable", "car_model": "unknown"}


def _query_llm(crop: np.ndarray) -> Dict[str, str]:
    """Dispatch to OpenAI if OPENAI_API_KEY is set, otherwise fall back to Gemini."""
    if OPENAI_API_KEY:
        logger.info("[VEHICLE] LLM provider: OpenAI (model=%s)", OPENAI_MODEL)
        return _query_openai(crop)
    logger.info("[VEHICLE] LLM provider: Gemini (model=%s)", GEMINI_MODEL)
    return _query_gemini(crop)


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
                print(f"[VEHICLE] slot={slot_id} attempt={attempt_number} | calling LLM")

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
                            "[VEHICLE] Skipping LLM — crop is empty for bbox=%s image_shape=%s",
                            bbox, image.shape,
                        )
                    elif not _is_crop_quality_ok(crop):
                        logger.info("[VEHICLE] Skipping LLM — crop quality below threshold")
                    else:
                        logger.info("[VEHICLE] Crop OK: shape=%s | calling LLM", crop.shape)
                        extracted  = _query_llm(crop)
                        car_number = extracted["car_number"]
                        car_model  = extracted["car_model"]
                        logger.info(
                            "[VEHICLE] LLM result: plate=%s model=%s", car_number, car_model
                        )
                else:
                    logger.warning("[VEHICLE] Skipping Gemini — image is None")

                plate_ok = car_number != "unreadable"
                model_ok = car_model != "unknown"
                success = plate_ok or model_ok

                if success and model_ok:
                    slot["extracted"]   = True
                    slot["car_number"]  = car_number
                    slot["car_model"]   = car_model
                    logger.info(
                        "[VEHICLE] Extraction success (full): slot=%s plate=%s model=%s",
                        slot_id, car_number, car_model,
                    )
                elif success and plate_ok and not model_ok:
                    # Keep low-call policy: allow only one deferred model-only retry.
                    slot["car_number"] = car_number
                    slot["car_model"] = slot.get("car_model")
                    if not slot.get("model_retry_done", False):
                        slot["model_retry_done"] = True
                        slot["extraction_backoff_until"] = max(
                            slot["extraction_backoff_until"],
                            slot["frame_counter"] + _MODEL_RETRY_DELAY,
                        )
                        logger.info(
                            "[VEHICLE] Plate-only extraction: scheduling one model retry at frame %d",
                            slot["extraction_backoff_until"],
                        )
                    else:
                        slot["extracted"] = True
                        logger.info(
                            "[VEHICLE] Plate-only extraction finalized (model retry already used) slot=%s",
                            slot_id,
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
