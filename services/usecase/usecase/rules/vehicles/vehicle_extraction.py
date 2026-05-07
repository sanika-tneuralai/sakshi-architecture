"""
Vehicle Detail Extraction Rule (frame-level)
============================================
One LLM vision call per snapshot. The LLM analyses the entire annotated frame
(every slot polygon drawn in yellow) and returns *all* visible vehicles with
position, plate, model, parking_quality. We then assign each LLM-returned
vehicle to a slot by matching its `position` (left/center/right/...) to the
slot polygon's centroid class.

Provider selection (env-driven, OpenAI primary, Gemini fallback):
- OPENAI_API_KEY → OpenAI (gpt-4o by default; override OPENAI_MODEL)
- GEMINI_API_KEY → Gemini (gemini-2.5-flash; override GEMINI_MODEL)
  Used as a second opinion when OpenAI returns "unknown"/"unreadable".

Per-slot retry budget is preserved via the shared slot state
(``slot:{camera_id}:{slot_id}``):
- ``slot.extracted``           : True once we have a usable result
- ``slot.extraction_attempts`` : 0..3
- ``slot.extraction_backoff_until`` : frame_counter gate
- ``slot.car_number``, ``slot.car_model`` : last good values

Differences vs the previous per-slot implementation:
- Old code: 1 LLM call per occupied slot per attempt (N calls per frame).
- New code: 1 LLM call per frame, results distributed to all slots at once.
  Strictly cheaper on multi-slot cameras, never more expensive on single-slot.
- Frame is annotated with every slot polygon (yellow). LLM identifies vehicles
  by position; mapping to slot_id happens here, not in the LLM.
- Logo-occluded slots (YOLO missed but G-marker says "occupied") are still
  emitted as vehicle_details rows so the orchestration session row stays open.
"""
import base64
import json
import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Tuple

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

# Per-slot retry backoff (frames) after a failed attempt.
_BACKOFF = [10, 20, 9999]
_MODEL_RETRY_DELAY = int(os.getenv("VEHICLE_MODEL_RETRY_DELAY_FRAMES", "45"))
_MIN_CONFIDENCE    = float(os.getenv("VEHICLE_MIN_CONFIDENCE", "0.35"))

DETECTION_W = int(os.getenv("DETECTION_WIDTH",  "1920"))
DETECTION_H = int(os.getenv("DETECTION_HEIGHT", "1080"))

_PLATE_NEGATIVE = {"unreadable", "vehicle_number_not_visible", "", "none", "null"}
_MODEL_NEGATIVE = {"unknown", "not_clear", "", "none", "null"}


# ---------------------------------------------------------------------------
# S3 download
# ---------------------------------------------------------------------------

def _download_image(snapshot_url: str) -> Optional[np.ndarray]:
    """Download a frame from a private S3 URL using boto3 (authenticated)."""
    try:
        url_path = snapshot_url.split(".amazonaws.com/", 1)
        if len(url_path) != 2:
            raise ValueError(f"Unrecognised S3 URL format: {snapshot_url}")
        key    = url_path[1]
        bucket = snapshot_url.split("//")[1].split(".s3.")[0]

        logger.info("[VEHICLE] Downloading S3 image: bucket=%s key=%s", bucket, key)
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("[VEHICLE] cv2.imdecode returned None — image data may be corrupt")
        else:
            logger.info("[VEHICLE] Image decoded: shape=%s (%d bytes)", img.shape, len(data))
        return img
    except Exception as exc:
        logger.error("[VEHICLE] Failed to download snapshot from %s: %s", snapshot_url, exc)
        return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_centroid_x(polygon: List[List[int]]) -> float:
    if not polygon:
        return 0.0
    return float(sum(p[0] for p in polygon)) / len(polygon)


def _bbox_center_x(bbox: Dict[str, float]) -> float:
    return (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0


def _classify_position(x_center: float, frame_w: float) -> str:
    """left if in left third, right if in right third, else center."""
    if frame_w <= 0:
        return "center"
    if x_center < frame_w / 3.0:
        return "left"
    if x_center > 2.0 * frame_w / 3.0:
        return "right"
    return "center"


def _annotate_all_slots(image: np.ndarray, rois: Dict[str, List[List[int]]]) -> np.ndarray:
    """Draw every slot polygon in yellow on a copy of the frame."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    sx = w / DETECTION_W
    sy = h / DETECTION_H

    for slot_id, polygon in rois.items():
        if not polygon:
            continue
        pts = np.array(
            [[int(p[0] * sx), int(p[1] * sy)] for p in polygon],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=4)
    return annotated


def _b64_jpeg(image: np.ndarray, quality: int = 95) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Strict frame-level prompt (user-supplied schema + parking_quality extension)
# ---------------------------------------------------------------------------

_FRAME_PROMPT = """You are a STRICT vehicle recognition and parking-slot analysis assistant.

You are given ONE CCTV parking/charging frame. Each parking slot is outlined as a YELLOW polygon on the frame.

Your task:
1. Detect whether any vehicle is present in the frame.
2. Identify the vehicle position:
   - "left"
   - "right"
   - "center"
   - "left_and_right"
   - "multiple"
3. For EACH visible vehicle:
   - Identify the vehicle make/model ONLY if visually supported.
   - Read the vehicle number plate ONLY if every character is clearly visible.
   - Judge the parking quality of that vehicle against its YELLOW slot outline.
4. NEVER guess or hallucinate any vehicle number or model.

==================================================
STRICT OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Schema:

{
  "vehicle_present": true,
  "vehicle_positions": ["left", "right"],
  "vehicles": [
    {
      "position": "left",
      "car_model": "Tata Tiago EV",
      "car_number": "KL44R6290",
      "number_plate_visible": true,
      "parking_quality": "proper"
    },
    {
      "position": "right",
      "car_model": "BYD Atto 3",
      "car_number": "unreadable",
      "number_plate_visible": false,
      "parking_quality": "across_line"
    }
  ]
}

If NO vehicle is present:

{
  "vehicle_present": false,
  "vehicle_positions": [],
  "vehicles": []
}

==================================================
STEP 1 — VEHICLE PRESENCE CHECK
==================================================
- First determine whether ANY vehicle exists in the frame.
- Do NOT assume a vehicle exists.
- Ignore: shadows, reflections, posters, printed vehicle images, banners,
  partial tiny vehicle fragments, mirrors/reflections.
- If no real vehicle is visible: vehicle_present = false, vehicles = [].

==================================================
STEP 2 — VEHICLE POSITION DETECTION
==================================================
- left           : vehicle occupies mostly the left half of the frame.
- right          : vehicle occupies mostly the right half of the frame.
- center         : vehicle occupies the central region of the frame.
- left_and_right : vehicles exist on both left and right.
- multiple       : more than 2 vehicles.

==================================================
STEP 3 — VEHICLE MODEL IDENTIFICATION
==================================================
Identify make + model ONLY from visible evidence:
  badge, logo, grille, headlights, body shape, charging port design.

STRICT RULES:
- NEVER invent a model.
- NEVER infer from parking location.
- NEVER use probability guessing.
- If uncertain, return "not_clear".

Allowed values:
- exact model name (e.g. "Tata Tiago EV", "BYD Atto 3", "Hyundai Creta")
- "not_clear"
- "unknown"

==================================================
STEP 4 — LICENSE PLATE RECOGNITION
==================================================
Indian plate format expected:
  2 letters + 1-2 digits + 1-3 letters + 4 digits  (e.g. KL01AB1234, TN37DK7758)

STRICT RULES:
- EVERY character MUST be clearly readable.
- If ANY character is blurry, cut off, tilted excessively, overexposed,
  shadowed, blocked, partially visible, low resolution, or ambiguous:
  return "unreadable".
- NEVER guess missing characters.
- NEVER autocomplete plate numbers.
- NEVER infer state codes or hidden digits.
- NEVER combine information across frames.

Invalid outputs (do NOT produce):
- "KL01AB12??"
- "probably KL44R6290"
- "KL44R629O"
- partial numbers

Only valid outputs:
- the full exact plate
- "unreadable"

==================================================
STEP 5 — SIDE-SPECIFIC NUMBER VISIBILITY
==================================================
Use "vehicle_number_not_visible" when:
- vehicle is clearly present
- but plate region is absent due to angle/cropping.

Use "unreadable" when:
- plate exists in the frame but cannot be read confidently.

==================================================
STEP 6 — PARKING QUALITY
==================================================
For each vehicle, judge against the YELLOW slot outline(s):

- "proper"        : the car is fully inside its yellow outline.
- "double_slot"   : the car spans this slot AND visibly overlaps an adjacent slot.
- "across_line"   : the car straddles the yellow boundary on one side.
- "outside_slot"  : the car is mostly or entirely outside any yellow outline.
- "unknown"       : slot boundary not clearly visible / occluded / cannot tell.

Report what you actually see; do NOT default to "proper" when uncertain.

==================================================
ADDITIONAL GUARDRAILS
==================================================
- NEVER hallucinate.
- NEVER estimate hidden characters.
- NEVER output extra explanation.
- NEVER include markdown.
- NEVER include confidence scores.
- NEVER include comments.
- NEVER include text outside JSON.

If uncertain, use:
  "not_clear", "unknown", "unreadable", "vehicle_number_not_visible".

Be conservative and strict.

Return ONLY valid JSON. No explanations. No markdown. No additional text.
"""


# ---------------------------------------------------------------------------
# LLM frame-level calls
# ---------------------------------------------------------------------------

def _empty_llm_response() -> Dict[str, Any]:
    return {"vehicle_present": False, "vehicle_positions": [], "vehicles": []}


def _validate_plate(value: Any) -> str:
    """Normalise plate field. Returns the plate, "unreadable", or "vehicle_number_not_visible"."""
    s = str(value or "").strip()
    if not s:
        return "unreadable"
    low = s.lower()
    if low in {"unreadable", "vehicle_number_not_visible"}:
        return low
    cleaned = s.replace(" ", "").replace("-", "")
    if len(cleaned) < 8:
        logger.warning("[VEHICLE] Plate '%s' rejected — too short (%d chars)", s, len(cleaned))
        return "unreadable"
    return cleaned.upper()


def _validate_model(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    if s.lower() in {"unknown", "not_clear"}:
        return s.lower()
    return s


def _validate_quality(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"proper", "double_slot", "across_line", "outside_slot", "unknown"}:
        return s
    return "unknown"


def _normalise_llm_vehicles(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce raw LLM JSON into a canonical shape, filtering bad entries."""
    if not isinstance(parsed, dict):
        return _empty_llm_response()

    vehicle_present = bool(parsed.get("vehicle_present", False))
    vehicle_positions = parsed.get("vehicle_positions") or []
    if not isinstance(vehicle_positions, list):
        vehicle_positions = []

    raw_vehicles = parsed.get("vehicles") or []
    if not isinstance(raw_vehicles, list):
        raw_vehicles = []

    vehicles: List[Dict[str, Any]] = []
    for v in raw_vehicles:
        if not isinstance(v, dict):
            continue
        position = str(v.get("position", "")).strip().lower()
        if position not in {"left", "right", "center", "left_and_right", "multiple"}:
            position = "center"
        vehicles.append({
            "position":              position,
            "car_model":             _validate_model(v.get("car_model")),
            "car_number":            _validate_plate(v.get("car_number")),
            "number_plate_visible":  bool(v.get("number_plate_visible", False)),
            "parking_quality":       _validate_quality(v.get("parking_quality")),
        })

    if not vehicles:
        vehicle_present = False

    return {
        "vehicle_present":   vehicle_present,
        "vehicle_positions": [str(p).strip().lower() for p in vehicle_positions],
        "vehicles":          vehicles,
    }


def _query_openai_frame(image: np.ndarray, rois: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        logger.warning("[VEHICLE] OPENAI_API_KEY not set — skipping OpenAI frame call")
        return _empty_llm_response()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        annotated = _annotate_all_slots(image, rois)
        frame_b64 = _b64_jpeg(annotated, quality=95)

        logger.info("[VEHICLE] Sending frame-level request to OpenAI (model=%s)", OPENAI_MODEL)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _FRAME_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"}},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("[VEHICLE] OpenAI raw response: %s", text)
        return _normalise_llm_vehicles(json.loads(text))
    except Exception as exc:
        logger.warning("[VEHICLE] OpenAI frame extraction failed: %s", exc, exc_info=True)
        return _empty_llm_response()


def _query_gemini_frame(image: np.ndarray, rois: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        logger.warning("[VEHICLE] GEMINI_API_KEY not set — skipping Gemini frame call")
        return _empty_llm_response()

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        annotated = _annotate_all_slots(image, rois)
        frame_b64 = _b64_jpeg(annotated, quality=95)

        response_schema = {
            "type": "object",
            "properties": {
                "vehicle_present":   {"type": "boolean"},
                "vehicle_positions": {"type": "array", "items": {"type": "string"}},
                "vehicles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position":             {"type": "string"},
                            "car_model":            {"type": "string"},
                            "car_number":           {"type": "string"},
                            "number_plate_visible": {"type": "boolean"},
                            "parking_quality":      {"type": "string"},
                        },
                        "required": ["position", "car_model", "car_number",
                                     "number_plate_visible", "parking_quality"],
                    },
                },
            },
            "required": ["vehicle_present", "vehicle_positions", "vehicles"],
        }

        logger.info("[VEHICLE] Sending frame-level request to Gemini (model=%s)", GEMINI_MODEL)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=_FRAME_PROMPT),
                types.Part.from_bytes(data=base64.b64decode(frame_b64), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0,
            ),
        )
        text = response.text.strip()
        logger.info("[VEHICLE] Gemini raw response: %s", text)
        return _normalise_llm_vehicles(json.loads(text))
    except Exception as exc:
        logger.warning("[VEHICLE] Gemini frame extraction failed: %s", exc, exc_info=True)
        return _empty_llm_response()


def _merge_llm(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in plate/model/parking_quality on the primary's vehicles[] from secondary
    when primary returned negative ("unreadable" / "unknown") at the same position.
    """
    if not secondary or not secondary.get("vehicles"):
        return primary
    if not primary.get("vehicle_present") and secondary.get("vehicle_present"):
        return secondary

    sec_by_pos: Dict[str, Dict[str, Any]] = {}
    for v in secondary["vehicles"]:
        sec_by_pos.setdefault(v["position"], v)

    for v in primary.get("vehicles", []):
        sec = sec_by_pos.get(v["position"])
        if not sec:
            continue
        if v["car_number"] in _PLATE_NEGATIVE and sec["car_number"] not in _PLATE_NEGATIVE:
            v["car_number"] = sec["car_number"]
            v["number_plate_visible"] = sec.get("number_plate_visible", True)
            logger.info("[VEHICLE] Gemini filled plate at position=%s: %s", v["position"], v["car_number"])
        if v["car_model"] in _MODEL_NEGATIVE and sec["car_model"] not in _MODEL_NEGATIVE:
            v["car_model"] = sec["car_model"]
            logger.info("[VEHICLE] Gemini filled model at position=%s: %s", v["position"], v["car_model"])
        if v["parking_quality"] == "unknown" and sec["parking_quality"] != "unknown":
            v["parking_quality"] = sec["parking_quality"]
    return primary


def _query_llm_frame(image: np.ndarray, rois: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    """OpenAI primary, Gemini second-opinion to fill negative fields."""
    if OPENAI_API_KEY:
        primary = _query_openai_frame(image, rois)
        needs_help = (
            not primary.get("vehicle_present")
            or any(v["car_model"] in _MODEL_NEGATIVE or v["car_number"] in _PLATE_NEGATIVE
                   for v in primary.get("vehicles", []))
        )
        if needs_help and GEMINI_API_KEY:
            logger.info("[VEHICLE] Asking Gemini for second opinion")
            secondary = _query_gemini_frame(image, rois)
            primary = _merge_llm(primary, secondary)
        return primary
    if GEMINI_API_KEY:
        return _query_gemini_frame(image, rois)
    logger.warning("[VEHICLE] No LLM API keys configured")
    return _empty_llm_response()


# ---------------------------------------------------------------------------
# Slot ↔ LLM-vehicle assignment
# ---------------------------------------------------------------------------

def _assign_vehicles_to_targets(
    vehicles: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    frame_w: float,
) -> Dict[str, Dict[str, Any]]:
    """
    targets: list of {"key": str, "x_center": float, "expected_class": str|None}
    Returns dict {key: vehicle_dict}.

    Strategy:
      1. Compute each target's position class (left/center/right) from x_center.
      2. Phase 1 — exact class match (one vehicle per target).
      3. Phase 2 — vehicles with position 'left_and_right' or 'multiple' fill
         any remaining target.
      4. Phase 3 — leftover vehicles in left→center→right order map to leftover
         targets sorted by x_center (deterministic fallback when LLM's position
         label disagrees with our class gridding).
    """
    if not vehicles or not targets:
        return {}

    for t in targets:
        if not t.get("expected_class"):
            t["expected_class"] = _classify_position(t["x_center"], frame_w)

    assignments: Dict[str, Dict[str, Any]] = {}
    used: set = set()

    # Phase 1
    for t in targets:
        if t["key"] in assignments:
            continue
        for i, v in enumerate(vehicles):
            if i in used:
                continue
            if v["position"] == t["expected_class"]:
                assignments[t["key"]] = v
                used.add(i)
                break

    # Phase 2
    for t in targets:
        if t["key"] in assignments:
            continue
        for i, v in enumerate(vehicles):
            if i in used:
                continue
            if v["position"] in ("left_and_right", "multiple"):
                assignments[t["key"]] = v
                used.add(i)
                break

    # Phase 3
    remaining_targets = sorted(
        (t for t in targets if t["key"] not in assignments),
        key=lambda t: t["x_center"],
    )
    pos_order = {"left": 0, "center": 1, "right": 2,
                 "left_and_right": 3, "multiple": 4}
    remaining_vehicles = [v for i, v in enumerate(vehicles) if i not in used]
    remaining_vehicles.sort(key=lambda v: pos_order.get(v["position"], 5))
    for t, v in zip(remaining_targets, remaining_vehicles):
        assignments[t["key"]] = v

    return assignments


# ---------------------------------------------------------------------------
# Slot-state update helper
# ---------------------------------------------------------------------------

def _apply_extraction_result(
    slot: Dict[str, Any],
    llm_vehicle: Optional[Dict[str, Any]],
) -> None:
    """Mutates slot in place. Mirrors the success/partial/fail rules from the
    previous per-slot implementation, but keyed off a frame-level result."""
    if llm_vehicle is None:
        # The LLM didn't see this slot's vehicle (or no vehicles at all).
        slot["extraction_attempts"] = int(slot.get("extraction_attempts", 0)) + 1
        if slot["extraction_attempts"] >= 3:
            slot["extracted"] = True
            logger.warning("[VEHICLE] 3 attempts exhausted (no LLM match) — marking extracted, fields null")
        else:
            slot["extraction_backoff_until"] = (
                int(slot.get("frame_counter", 0)) + _BACKOFF[slot["extraction_attempts"] - 1]
            )
        return

    car_number = llm_vehicle["car_number"]
    car_model  = llm_vehicle["car_model"]
    plate_ok = car_number not in _PLATE_NEGATIVE
    model_ok = car_model  not in _MODEL_NEGATIVE

    slot["parking_quality"] = llm_vehicle.get("parking_quality", "unknown")

    if plate_ok and model_ok:
        slot["extracted"]  = True
        slot["car_number"] = car_number
        slot["car_model"]  = car_model
        logger.info("[VEHICLE] Extraction success (full): plate=%s model=%s quality=%s",
                    car_number, car_model, slot["parking_quality"])
        return

    if plate_ok and not model_ok:
        slot["car_number"] = car_number
        if not slot.get("model_retry_done", False):
            slot["model_retry_done"] = True
            slot["extraction_backoff_until"] = max(
                int(slot.get("extraction_backoff_until", 0)),
                int(slot.get("frame_counter", 0)) + _MODEL_RETRY_DELAY,
            )
            logger.info(
                "[VEHICLE] Plate-only — scheduling one model retry at frame %d",
                slot["extraction_backoff_until"],
            )
        else:
            slot["extracted"] = True
            logger.info("[VEHICLE] Plate-only — finalised (model retry already used)")
        return

    if model_ok and not plate_ok:
        # Model only — keep the model, count the attempt against retries.
        slot["car_model"] = car_model
        slot["extraction_attempts"] = int(slot.get("extraction_attempts", 0)) + 1
        if slot["extraction_attempts"] >= 3:
            slot["extracted"] = True
            logger.info("[VEHICLE] Model-only finalised (3 attempts) model=%s", car_model)
        else:
            slot["extraction_backoff_until"] = (
                int(slot.get("frame_counter", 0)) + _BACKOFF[slot["extraction_attempts"] - 1]
            )
        return

    # Neither plate nor model.
    slot["extraction_attempts"] = int(slot.get("extraction_attempts", 0)) + 1
    if slot["extraction_attempts"] >= 3:
        slot["extracted"] = True
        logger.warning("[VEHICLE] 3 attempts exhausted — marking extracted, fields null")
    else:
        slot["extraction_backoff_until"] = (
            int(slot.get("frame_counter", 0)) + _BACKOFF[slot["extraction_attempts"] - 1]
        )


def _wants_extraction(slot: Dict[str, Any], force_occupied: bool = False) -> bool:
    return (
        (bool(slot.get("occupied")) or force_occupied)
        and not slot.get("extracted", False)
        and int(slot.get("extraction_attempts", 0)) < 3
        and int(slot.get("frame_counter", 0)) >= int(slot.get("extraction_backoff_until", 0))
    )


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

class VehicleExtractionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "vehicle_extraction"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id     = detection_output.get("camera_id", "unknown")
        rois          = detection_output.get("rois") or {}
        snapshot_url  = detection_output.get("snapshot_url")
        logo_occluded = detection_output.get("logo_occluded") or {}

        tracked_cars = detection_output.get("tracked_cars") or [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") == "car"
        ]

        # Filter low-confidence YOLO detections.
        tracked_cars = [
            c for c in tracked_cars
            if float(c.get("confidence", 0.0)) >= _MIN_CONFIDENCE
            and all(k in (c.get("bbox") or {}) for k in ("x1", "y1", "x2", "y2"))
        ]

        if not tracked_cars and not any(logo_occluded.values()):
            logger.info("[VEHICLE] camera=%s no cars and no logo-occluded slots", camera_id)
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        # ── Build target list ────────────────────────────────────────────
        # Each "target" is something we may want plate/model for. Three kinds:
        #   - in-slot YOLO car      key=slot_id
        #   - unauthorized YOLO car key=track:<track_id>
        #   - logo-occluded slot    key=slot_id (no YOLO bbox)
        targets: List[Dict[str, Any]] = []
        target_meta: Dict[str, Dict[str, Any]] = {}  # key -> {kind, track_id, slot_id, bbox}

        slots_with_yolo: set = set()

        for car in tracked_cars:
            bbox = car["bbox"]
            track_id = car.get("track_id")
            matched = which_rois(bbox, rois) if rois else []

            if matched:
                slot_id = matched[0]
                slots_with_yolo.add(slot_id)
                key = slot_id
                kind = "yolo_in_slot"
            else:
                if not track_id:
                    continue
                slot_id = f"track:{track_id}"
                key = slot_id
                kind = "yolo_unauthorized"

            if key in target_meta:
                continue   # multi-frame dedupe within a single evaluate()
            target_meta[key] = {
                "kind": kind, "track_id": track_id, "slot_id": slot_id,
                "bbox": bbox, "confidence": float(car.get("confidence", 0.0)),
            }
            targets.append({"key": key, "x_center": _bbox_center_x(bbox)})

        # CV-only logo-occluded slots (G-marker says "occluded" but YOLO has no car).
        for roi_name, occluded in logo_occluded.items():
            if not occluded or roi_name not in rois or roi_name in slots_with_yolo:
                continue
            key = roi_name
            if key in target_meta:
                continue
            synthetic_track = f"cv:{camera_id}:{roi_name}"
            target_meta[key] = {
                "kind": "cv_only_logo", "track_id": synthetic_track,
                "slot_id": roi_name, "bbox": None, "confidence": 1.0,
            }
            targets.append({
                "key": key,
                "x_center": _polygon_centroid_x(rois[roi_name]),
            })

        if not targets:
            return {"triggered": False, "matched_objects": [], "vehicle_details": []}

        # ── Decide whether to call the LLM ───────────────────────────────
        targets_needing_llm: List[Dict[str, Any]] = []
        slot_states: Dict[str, Dict[str, Any]] = {}
        for t in targets:
            key = t["key"]
            slot = get_slot_state(camera_id, key)
            slot_states[key] = slot
            kind = target_meta[key]["kind"]
            force = kind in ("yolo_unauthorized", "cv_only_logo")
            if _wants_extraction(slot, force_occupied=force):
                targets_needing_llm.append(t)

        llm_resp: Dict[str, Any] = _empty_llm_response()
        image: Optional[np.ndarray] = None

        if targets_needing_llm and snapshot_url:
            image = _download_image(snapshot_url)
            if image is not None:
                llm_resp = _query_llm_frame(image, rois)
                logger.info(
                    "[VEHICLE] Frame LLM result: present=%s vehicles=%d",
                    llm_resp.get("vehicle_present"), len(llm_resp.get("vehicles") or []),
                )

        # ── Distribute LLM vehicles back to targets ──────────────────────
        frame_w = float(image.shape[1]) if image is not None else float(DETECTION_W)
        assignments = _assign_vehicles_to_targets(
            llm_resp.get("vehicles", []),
            [{"key": t["key"], "x_center": t["x_center"], "expected_class": None}
             for t in targets_needing_llm],
            frame_w,
        )

        # ── Update slot state for targets that participated in the call ──
        for t in targets_needing_llm:
            key = t["key"]
            slot = slot_states[key]
            llm_vehicle = assignments.get(key)
            _apply_extraction_result(slot, llm_vehicle)
            set_slot_state(camera_id, key, slot)

        # ── Build vehicle_details output (contract-preserving) ───────────
        vehicle_details: List[Dict[str, Any]] = []
        for t in targets:
            key = t["key"]
            meta = target_meta[key]
            slot = slot_states[key]
            kind = meta["kind"]

            slot_id_out = None if kind == "yolo_unauthorized" else meta["slot_id"]
            vehicle_details.append({
                "track_id":   meta["track_id"],
                "slot_id":    slot_id_out,
                "car_number": slot.get("car_number"),
                "car_model":  slot.get("car_model"),
                "confidence": meta["confidence"],
            })

        logger.info(
            "[VEHICLE] result: triggered=True | targets=%d | llm_called=%s | "
            "rows=%s",
            len(targets), bool(targets_needing_llm and image is not None),
            [(v["slot_id"], v["car_number"], v["car_model"]) for v in vehicle_details],
        )

        return {
            "triggered":       True,
            "matched_objects": tracked_cars,
            "vehicle_details": vehicle_details,
        }
