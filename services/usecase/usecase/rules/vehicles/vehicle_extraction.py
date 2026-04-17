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
_MIN_CONFIDENCE = float(os.getenv("VEHICLE_MIN_CONFIDENCE", "0.35"))

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
    x1 = int(bbox.get("x1", 0) * sx)
    y1 = int(bbox.get("y1", 0) * sy)
    x2 = int(bbox.get("x2", w) * sx)
    y2 = int(bbox.get("y2", h) * sy)

    # Add 15% padding so plate edges and model badges aren't clipped
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    logger.info(
        "[VEHICLE] Scaled bbox (with 15%% padding): x1=%d y1=%d x2=%d y2=%d (image=%dx%d detection=%dx%d)",
        x1, y1, x2, y2, w, h, DETECTION_W, DETECTION_H,
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _crop_to_b64(crop: np.ndarray) -> str:
    # Use high JPEG quality to preserve plate characters and model badges
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
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


def _annotate_frame(image: np.ndarray, bbox: dict) -> np.ndarray:
    """Draw a bright green rectangle on a copy of the frame to highlight the target car."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    sx = w / DETECTION_W
    sy = h / DETECTION_H
    x1 = max(0, int(bbox.get("x1", 0) * sx))
    y1 = max(0, int(bbox.get("y1", 0) * sy))
    x2 = min(w, int(bbox.get("x2", w) * sx))
    y2 = min(h, int(bbox.get("y2", h) * sy))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return annotated


def _query_gemini(full_frame: np.ndarray, bbox: dict) -> Dict[str, str]:
    """Send full frame + crop to Gemini. Returns car_number + car_model."""
    if not GEMINI_API_KEY:
        logger.warning("[VEHICLE] GEMINI_API_KEY not set — skipping Gemini extraction")
        return {"car_number": "unreadable", "car_model": "unknown"}

    logger.info("[VEHICLE] Gemini API key present (len=%d), model=%s", len(GEMINI_API_KEY), GEMINI_MODEL)
    logger.info("[VEHICLE] Full frame shape: %s, bbox=%s", full_frame.shape, bbox)

    try:
        from google import genai
        from google.genai import types

        client    = genai.Client(api_key=GEMINI_API_KEY)
        frame_b64 = _crop_to_b64(full_frame)
        crop      = _crop_bbox(full_frame, bbox)
        crop_b64  = _crop_to_b64(crop) if crop is not None and crop.size > 0 else None
        logger.info("[VEHICLE] Full frame encoded: %d chars, crop encoded: %s chars",
                    len(frame_b64), len(crop_b64) if crop_b64 else "N/A")

        prompt = (
            "You are a strict vehicle recognition assistant.\n"
            "Two images are provided:\n"
            "  1. The full scene showing a parking/charging area\n"
            "  2. A close-up crop of the specific vehicle to analyze\n\n"
            "Analyze the vehicle shown in the close-up (second image) and extract verifiable information. "
            "Use the full scene (first image) for additional context if needed.\n\n"

            "Return a JSON object with EXACTLY these two keys:\n"
            '  "car_number": string\n'
            '  "car_model": string\n\n'

            "Guardrails for license plate (car_number):\n"
            "- Output MUST be valid JSON. No extra text, no explanation, no comments.\n"
            "- If the license plate is clipped, cut off, or extends beyond the image edge, return \"unreadable\".\n"
            "- If ANY characters on the plate are obscured, blurry, or ambiguous, return \"unreadable\".\n"
            "- Do NOT guess, infer, complete, or hallucinate missing characters.\n"
            "- A valid Indian plate has the form: 2 letters + 1-2 digits + 1-3 letters + 4 digits (e.g. KL01AB1234).\n"
            "- If you cannot read ALL characters of the plate with high confidence, return \"unreadable\".\n"
            "- Partial plates like \"NZ04\" or \"KL0\" MUST be returned as \"unreadable\".\n\n"

            "Guardrails for car model:\n"
            "- Identify make AND model from body shape, headlights, grille, badges (e.g. \"BYD Atto 3\", \"Hyundai Creta\").\n"
            "- If the vehicle is clearly visible but the exact model is uncertain, provide the make + best-guess model.\n"
            "- Only return \"unknown\" if you genuinely cannot identify make or model from the visible car.\n"
            "- Do NOT fabricate brands/models not supported by visible evidence.\n\n"

            "Output rules:\n"
            "- Do NOT include anything outside the JSON object.\n"
            "- Ensure correct JSON formatting (double quotes, no trailing commas).\n\n"

            "Example output:\n"
            '{ "car_number": "KL01AB1234", "car_model": "Hyundai Creta" }\n\n'

            "Now analyze the vehicle and return ONLY the JSON."
        )

        response_schema = {
            "type": "object",
            "properties": {
                "car_number": {"type": "string"},
                "car_model":  {"type": "string"},
            },
            "required": ["car_number", "car_model"],
        }

        # Build content parts: full frame + crop
        content_parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=base64.b64decode(frame_b64), mime_type="image/jpeg"),
        ]
        if crop_b64:
            content_parts.append(
                types.Part.from_bytes(data=base64.b64decode(crop_b64), mime_type="image/jpeg")
            )

        logger.info("[VEHICLE] Sending request to Gemini (full frame + crop)...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_parts,
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

        # Reject suspiciously short plates (Indian plates have 8-10 chars: 2L + 1-2D + 1-3L + 4D)
        plate_clean = car_number.replace(" ", "").replace("-", "")
        if car_number != "unreadable" and len(plate_clean) < 8:
            logger.warning(
                "[VEHICLE] Plate '%s' rejected — too short (%d chars) to be a full plate, marking unreadable",
                car_number, len(plate_clean),
            )
            car_number = "unreadable"

        logger.info("[VEHICLE] Gemini parsed result: car_number=%s car_model=%s", car_number, car_model)
        return {"car_number": car_number, "car_model": car_model}

    except Exception as exc:
        logger.warning("[VEHICLE] Gemini extraction failed: %s", exc, exc_info=True)
        return {"car_number": "unreadable", "car_model": "unknown"}


def _query_openai(full_frame: np.ndarray, bbox: dict) -> Dict[str, str]:
    """Send full frame + crop to OpenAI vision. Returns car_number + car_model."""
    if not OPENAI_API_KEY:
        logger.warning("[VEHICLE] OPENAI_API_KEY not set — skipping OpenAI extraction")
        return {"car_number": "unreadable", "car_model": "unknown"}

    logger.info("[VEHICLE] OpenAI API key present (len=%d), model=%s", len(OPENAI_API_KEY), OPENAI_MODEL)
    logger.info("[VEHICLE] Full frame shape: %s, bbox=%s", full_frame.shape, bbox)

    try:
        from openai import OpenAI

        client    = OpenAI(api_key=OPENAI_API_KEY)
        frame_b64 = _crop_to_b64(full_frame)
        crop      = _crop_bbox(full_frame, bbox)
        crop_b64  = _crop_to_b64(crop) if crop is not None and crop.size > 0 else None
        logger.info("[VEHICLE] Full frame encoded: %d chars, crop encoded: %s chars",
                    len(frame_b64), len(crop_b64) if crop_b64 else "N/A")

        prompt = (
            "You are a strict vehicle recognition assistant.\n"
            "Two images are provided:\n"
            "  1. The full scene showing a parking/charging area\n"
            "  2. A close-up crop of the specific vehicle to analyze\n\n"
            "Analyze the vehicle shown in the close-up (second image) and extract verifiable information. "
            "Use the full scene (first image) for additional context if needed.\n\n"

            "Return a JSON object with EXACTLY these two keys:\n"
            '  "car_number": string\n'
            '  "car_model": string\n\n'

            "Guardrails for license plate (car_number):\n"
            "- Output MUST be valid JSON. No extra text, no explanation, no comments.\n"
            "- If the license plate is clipped, cut off, or extends beyond the image edge, return \"unreadable\".\n"
            "- If ANY characters on the plate are obscured, blurry, or ambiguous, return \"unreadable\".\n"
            "- Do NOT guess, infer, complete, or hallucinate missing characters.\n"
            "- A valid Indian plate has the form: 2 letters + 1-2 digits + 1-3 letters + 4 digits (e.g. KL01AB1234).\n"
            "- If you cannot read ALL characters of the plate with high confidence, return \"unreadable\".\n"
            "- Partial plates like \"NZ04\" or \"KL0\" MUST be returned as \"unreadable\".\n\n"

            "Guardrails for car model:\n"
            "- Identify make AND model from body shape, headlights, grille, badges (e.g. \"BYD Atto 3\", \"Hyundai Creta\").\n"
            "- If the vehicle is clearly visible but the exact model is uncertain, provide the make + best-guess model.\n"
            "- Only return \"unknown\" if you genuinely cannot identify make or model from the visible car.\n"
            "- Do NOT fabricate brands/models not supported by visible evidence.\n\n"

            "Output rules:\n"
            "- Do NOT include anything outside the JSON object.\n"
            "- Ensure correct JSON formatting (double quotes, no trailing commas).\n\n"

            "Example output:\n"
            '{ "car_number": "KL01AB1234", "car_model": "Hyundai Creta" }\n\n'

            "Now analyze the vehicle and return ONLY the JSON."
        )

        # Build image content: full frame + crop
        image_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"},
            },
        ]
        if crop_b64:
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"},
            })

        logger.info("[VEHICLE] Sending request to OpenAI (full frame + crop)...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": image_content,
                }
            ],
        )

        text = response.choices[0].message.content.strip()
        logger.info("[VEHICLE] OpenAI raw response: %s", text)

        result     = json.loads(text)
        car_number = str(result.get("car_number", "unreadable")).strip()
        car_model  = str(result.get("car_model",  "unknown")).strip()

        # Reject suspiciously short plates (Indian plates have 8-10 chars: 2L + 1-2D + 1-3L + 4D)
        plate_clean = car_number.replace(" ", "").replace("-", "")
        if car_number != "unreadable" and len(plate_clean) < 8:
            logger.warning(
                "[VEHICLE] Plate '%s' rejected — too short (%d chars) to be a full plate, marking unreadable",
                car_number, len(plate_clean),
            )
            car_number = "unreadable"

        logger.info("[VEHICLE] OpenAI parsed result: car_number=%s car_model=%s", car_number, car_model)
        return {"car_number": car_number, "car_model": car_model}

    except Exception as exc:
        logger.warning("[VEHICLE] OpenAI extraction failed: %s", exc, exc_info=True)
        return {"car_number": "unreadable", "car_model": "unknown"}


def _query_llm(full_frame: np.ndarray, bbox: dict) -> Dict[str, str]:
    """
    Try OpenAI first (if available). If it gets the plate but not the model,
    ask Gemini specifically for the model — Gemini is better at car identification.
    """
    result = {"car_number": "unreadable", "car_model": "unknown"}

    if OPENAI_API_KEY:
        logger.info("[VEHICLE] LLM provider (primary): OpenAI (model=%s)", OPENAI_MODEL)
        result = _query_openai(full_frame, bbox)
    elif GEMINI_API_KEY:
        logger.info("[VEHICLE] LLM provider (primary): Gemini (model=%s)", GEMINI_MODEL)
        return _query_gemini(full_frame, bbox)

    # If OpenAI couldn't identify the model, try Gemini as a second opinion
    if result["car_model"] == "unknown" and GEMINI_API_KEY:
        logger.info("[VEHICLE] OpenAI returned model=unknown — trying Gemini for car model")
        gemini_result = _query_gemini(full_frame, bbox)
        if gemini_result["car_model"] != "unknown":
            result["car_model"] = gemini_result["car_model"]
            logger.info("[VEHICLE] Gemini identified model: %s", result["car_model"])
        if result["car_number"] == "unreadable" and gemini_result["car_number"] != "unreadable":
            result["car_number"] = gemini_result["car_number"]
            logger.info("[VEHICLE] Gemini identified plate: %s", result["car_number"])

    return result


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
            confidence = car.get("confidence", 0.0)

            # Skip low-confidence detections — they often produce hallucinated plates
            if confidence < _MIN_CONFIDENCE:
                logger.info(
                    "[VEHICLE] Skipping track_id=%s — confidence %.3f < %.2f threshold",
                    track_id, confidence, _MIN_CONFIDENCE,
                )
                continue

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
                        # Don't waste this attempt on a bad crop — roll back
                        slot["extraction_attempts"] -= 1
                        slot["extraction_backoff_until"] = slot["frame_counter"] + 5
                    elif not _is_crop_quality_ok(crop):
                        logger.info("[VEHICLE] Skipping LLM — crop quality below threshold")
                        # Don't waste this attempt on a low-quality crop — roll back
                        slot["extraction_attempts"] -= 1
                        slot["extraction_backoff_until"] = slot["frame_counter"] + 5
                    else:
                        logger.info("[VEHICLE] Crop OK: shape=%s | sending full frame with bbox to LLM", crop.shape)
                        extracted  = _query_llm(image, bbox)
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
