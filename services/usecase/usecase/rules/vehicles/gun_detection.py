"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Detects when a charging gun is plugged into a car and when it is unplugged.

Backends (env GUN_DETECTION_BACKEND):
  llm    — call an LLM Vision API on the snapshot. Default. Used while the
           in-house gun YOLO model is unreliable. The rule polls per slot:
              Plug-in:  1 min cadence, 2-of-N confirmation, max 20 attempts
              Plug-out: 5 min cadence, one-shot, runs until parking_outtime
  yolo   — original YOLO-based gun detector with inferred plug-in fallback.
           Kept for the future where a retrained gun model becomes reliable.

Design notes that apply to both backends:
- NO own CentroidTracker. Tracked cars are read from
  detection_output["tracked_cars"], which parking_detection populates via the
  engine's slim-payload backfill before this rule runs.
- Slot state is read/written via get_slot_state / set_slot_state using key
  ``slot:{camera_id}:{slot_id}``.  The slot_id (ROI name) is the only key.
- Events emitted (gun_plugin / gun_plugout / gun_unauthorized) and their
  metadata schema are identical across backends — the persistence layer
  doesn't know or care which backend produced them.

LLM backend specifics:
- One LLM call per evaluation per camera, listing every occupied slot.
  Whole-frame call returns a per-slot answer dict, so the LLM can use
  context (cable routing, gun visible vs. holstered) when deciding.
- Plug-in cadence is wall-clock seconds, NOT frame-counter. Robust to any
  change in orchestration poll_interval.
- 2-of-N for plug-in suppresses single-frame flicker. Plug-out is one-shot
  because the orchestration pacing already gives ~5 min between polls and
  parking_outtime is the safety net if we miss it.
- "Gun never plugged in" is a valid terminal state: after
  GUN_LLM_PLUGIN_MAX_POLLS unsuccessful attempts, plug_time stays NULL and
  no plug_out polling starts. Persistence will write an "incomplete"
  session row with NULL plug_time / plug_out_time, which is correct.

YOLO backend (legacy) — see plug-in / plug-out documentation in code below.
"""
import base64
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Dict, List, Optional

import boto3
import cv2
import numpy as np

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_slot_state, set_slot_state

logger = logging.getLogger(__name__)

GUN_CLASS          = "gun"
CAR_CLASS          = "car"
GUN_PLUGIN_FRAMES  = 3    # consecutive frames gun must be present to confirm plugin

# ── Backend selection ───────────────────────────────────────────────────────
# Default to LLM until we have a reliable gun YOLO model. Flip to "yolo" by
# setting GUN_DETECTION_BACKEND=yolo (no code change required).
GUN_DETECTION_BACKEND = os.getenv("GUN_DETECTION_BACKEND", "llm").lower()

# ── LLM backend tunables ────────────────────────────────────────────────────
# Wall-clock based: robust to orchestration poll_interval changes (1.0 today,
# could change tomorrow). All three are envoverridable.
GUN_LLM_PLUGIN_INTERVAL_S   = int(os.getenv("GUN_LLM_PLUGIN_INTERVAL_S",  "60"))    # 1 min
GUN_LLM_PLUGOUT_INTERVAL_S  = int(os.getenv("GUN_LLM_PLUGOUT_INTERVAL_S", "300"))   # 5 min
GUN_LLM_PLUGIN_MAX_POLLS    = int(os.getenv("GUN_LLM_PLUGIN_MAX_POLLS",   "20"))    # ~20 min cap
GUN_LLM_PLUGIN_CONSECUTIVE  = int(os.getenv("GUN_LLM_PLUGIN_CONSECUTIVE", "2"))     # 2-of-N
# Plug-out is symmetric with plug-in: a single not_plugged_in verdict isn't
# enough to close a session, because the LLM can flicker on occluded views.
# Need this many consecutive not_plugged_in verdicts before firing gun_plugout.
GUN_LLM_PLUGOUT_CONSECUTIVE = int(os.getenv("GUN_LLM_PLUGOUT_CONSECUTIVE", "2"))     # 2-of-N

# Time-based inferred plug-in for the LLM backend. The visual model has
# imperfect recall — some camera angles make the cable hard to see and Claude
# returns "not_plugged_in" even when the OCPP energy meter shows the car is
# charging. After this many seconds since parking_intime with no LLM-confirmed
# plug, fire a synthetic gun_plugin so the dashboard isn't stuck on
# "Plug In: Not yet" indefinitely. The fired event is marked source=inferred
# so downstream knows it's a fallback, not a visual confirmation.
GUN_LLM_INFERRED_PLUGIN_SECONDS = int(os.getenv("GUN_LLM_INFERRED_PLUGIN_SECONDS", "180"))  # 3 min

# Provider selection mirrors vehicle_extraction. Set GUN_LLM_PROVIDER to
# "claude" (default), "openai", or "gemini". All three call sites are kept so
# switching back is an env-var change, not a code change.
GUN_LLM_PROVIDER = os.getenv("GUN_LLM_PROVIDER", "openai").strip().lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DETECTION_W = int(os.getenv("DETECTION_WIDTH",  "1920"))
DETECTION_H = int(os.getenv("DETECTION_HEIGHT", "1080"))

# ── YOLO backend tunables (legacy, unchanged) ───────────────────────────────
# Inferred plug-in: after this many seconds since parking_intime with no real
# gun detection, fire a synthetic gun_plugin so downstream energy lookups have
# a plug_time anchor.
INFERRED_PLUGIN_SECONDS = 120
# After firing an inferred plug, expect a real gun frame within this window.
# If never seen, roll back so the row doesn't end up with a phantom plug_time.
INFERRED_VERIFY_TIMEOUT_SECONDS = 300

# Plug-out debounce. PLUGGED → MAYBE_OUT after sustained absence; only commit
# gun_plugout once the gun stays absent past both the grace and confirmation
# windows without returning.
GUN_MAYBE_OUT_SECONDS    = 60
GUN_RETURN_GRACE_SECONDS = 120
GUN_CONFIRM_OUT_SECONDS  = 120


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _event_ts(detection_output: Dict[str, Any]) -> str:
    """
    Pick the timestamp that should anchor events emitted from this evaluation.

    Prefer `detection_output["timestamp"]` (the camera service stamps it from
    the frame's capture time). Falls back to wall-clock when the field is
    absent or unparseable. Returns an ISO 8601 string — drop-in for `_now()`
    on event payloads.
    """
    ts = detection_output.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return _now()


def _parse_iso(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _seconds_since(iso_str: str | None, now: datetime) -> float:
    """Wall-clock seconds since iso_str. 0 if iso_str is None/invalid."""
    dt = _parse_iso(iso_str)
    if dt is None:
        return 0.0
    return max(0.0, (now - dt).total_seconds())


def _gun_name_for_roi(roi_name: str) -> str:
    parts = roi_name.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else roi_name
    return f"Gun {suffix}"


# ---------------------------------------------------------------------------
# LLM backend helpers
# ---------------------------------------------------------------------------

def _download_snapshot(snapshot_url: str) -> Optional[np.ndarray]:
    """Download a frame from a private S3 URL using boto3 (authenticated).

    Mirrors vehicle_extraction's downloader so the two rules behave the same
    on auth/retry. Returns None on any failure (caller handles fallback).
    """
    try:
        url_path = snapshot_url.split(".amazonaws.com/", 1)
        if len(url_path) != 2:
            raise ValueError(f"Unrecognised S3 URL format: {snapshot_url}")
        key    = url_path[1]
        bucket = snapshot_url.split("//")[1].split(".s3.")[0]

        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = s3.get_object(Bucket=bucket, Key=key)
        arr = np.frombuffer(response["Body"].read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.error("[GUN-LLM] Snapshot download failed: %s", exc)
        return None


def _annotate_slots(image: np.ndarray, slot_polygons: Dict[str, List[List[int]]]) -> np.ndarray:
    """Draw each polled slot polygon on a copy of the frame, with its slot_id
    label printed at the polygon's top-left vertex. The LLM uses the labels
    to address slots in its JSON response."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    sx = w / DETECTION_W
    sy = h / DETECTION_H
    for slot_id, polygon in slot_polygons.items():
        pts = np.array(
            [[int(p[0] * sx), int(p[1] * sy)] for p in polygon],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=4)
        # Label: top-left corner of the bounding rect of the polygon
        xs = [p[0][0] for p in pts]
        ys = [p[0][1] for p in pts]
        label_pos = (min(xs), max(0, min(ys) - 10))
        cv2.putText(
            annotated, slot_id, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3,
        )
    return annotated


def _frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_gun_prompt(slot_ids: List[str]) -> str:
    return (
        "You are a strict EV charging-gun observer. Each highlighted YELLOW polygon "
        "is a parking slot with a charging gun station next to it.\n"
        "For EVERY slot listed below, decide whether the charging gun is currently "
        "PLUGGED INTO the car parked in that slot.\n\n"
        f"Slots to evaluate: {', '.join(slot_ids)}\n\n"

        "Return ONE JSON object whose top-level keys are EXACTLY the slot ids above. "
        "For each slot, return one of these string values:\n"
        '  "plugged_in"      — the gun cable visibly enters the car\'s charging port.\n'
        '  "not_plugged_in"  — the gun is in its holster on the station, OR the cable '
        'hangs loose, OR no gun is visible near the car.\n'
        '  "unclear"         — you cannot tell (occlusion, glare, gun out of frame).\n\n'

        "Guardrails:\n"
        '- "plugged_in" requires VISIBLE evidence of the cable entering the car. '
        'A person standing near the gun is NOT plug-in evidence.\n'
        '- A coiled / holstered cable is "not_plugged_in" even if a car is in the slot.\n'
        '- When in genuine doubt, return "unclear" rather than guessing.\n'
        '- Output MUST be valid JSON, no extra text, no comments.\n\n'

        "Example output:\n"
        '{ "ROI_1": "plugged_in", "ROI_2": "not_plugged_in" }\n'
    )


def _query_gun_llm_claude(annotated_frame: np.ndarray, slot_ids: List[str]) -> Dict[str, str]:
    """One Claude vision call. Returns {slot_id: status} for every slot.

    JSON shape is constrained via output_config — the schema is built from
    slot_ids so Claude returns one key per slot.
    """
    if not ANTHROPIC_API_KEY:
        return {sid: "unclear" for sid in slot_ids}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = _build_gun_prompt(slot_ids)
        b64 = _frame_to_b64(annotated_frame)

        logger.info("[GUN-LLM] Claude request slots=%s", slot_ids)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                ],
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {sid: {"type": "string"} for sid in slot_ids},
                        "required": list(slot_ids),
                        "additionalProperties": False,
                    },
                },
            },
        )
        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()
        logger.info("[GUN-LLM] Claude response: %s", text)
        if not text:
            return {sid: "unclear" for sid in slot_ids}
        parsed = json.loads(text)
        return {sid: str(parsed.get(sid, "unclear")).strip().lower() for sid in slot_ids}
    except Exception as exc:
        logger.warning("[GUN-LLM] Claude call failed: %s", exc, exc_info=True)
        return {sid: "unclear" for sid in slot_ids}


def _query_gun_llm_openai(annotated_frame: np.ndarray, slot_ids: List[str]) -> Dict[str, str]:
    """One OpenAI vision call. Returns {slot_id: status} for every slot.

    Falls back to {slot: "unclear"} for everyone on any error so the rule
    treats the call as a no-op rather than mis-firing events.
    """
    if not OPENAI_API_KEY:
        return {sid: "unclear" for sid in slot_ids}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = _build_gun_prompt(slot_ids)
        b64 = _frame_to_b64(annotated_frame)

        logger.info("[GUN-LLM] OpenAI request slots=%s", slot_ids)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("[GUN-LLM] OpenAI response: %s", text)
        parsed = json.loads(text)
        return {sid: str(parsed.get(sid, "unclear")).strip().lower() for sid in slot_ids}
    except Exception as exc:
        logger.warning("[GUN-LLM] OpenAI call failed: %s", exc, exc_info=True)
        return {sid: "unclear" for sid in slot_ids}


def _query_gun_llm_gemini(annotated_frame: np.ndarray, slot_ids: List[str]) -> Dict[str, str]:
    """Gemini fallback. Same contract as the OpenAI variant."""
    if not GEMINI_API_KEY:
        return {sid: "unclear" for sid in slot_ids}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = _build_gun_prompt(slot_ids)
        b64 = _frame_to_b64(annotated_frame)

        logger.info("[GUN-LLM] Gemini request slots=%s", slot_ids)
        response_schema = {
            "type": "object",
            "properties": {sid: {"type": "string"} for sid in slot_ids},
            "required": slot_ids,
        }
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0,
            ),
        )
        text = response.text.strip()
        logger.info("[GUN-LLM] Gemini response: %s", text)
        parsed = json.loads(text)
        return {sid: str(parsed.get(sid, "unclear")).strip().lower() for sid in slot_ids}
    except Exception as exc:
        logger.warning("[GUN-LLM] Gemini call failed: %s", exc, exc_info=True)
        return {sid: "unclear" for sid in slot_ids}


def _query_gun_llm(annotated_frame: np.ndarray, slot_ids: List[str]) -> Dict[str, str]:
    """Dispatch to the provider configured via GUN_LLM_PROVIDER.

    - ``claude`` (default): single Claude call.
    - ``openai``: OpenAI primary; Gemini fallback only if OpenAI returns
      all-unclear (preserves the previous behaviour for callers switching back).
    - ``gemini``: Gemini only.
    """
    if GUN_LLM_PROVIDER == "claude":
        if ANTHROPIC_API_KEY:
            return _query_gun_llm_claude(annotated_frame, slot_ids)
        logger.warning("[GUN-LLM] GUN_LLM_PROVIDER=claude but ANTHROPIC_API_KEY not set")
        return {sid: "unclear" for sid in slot_ids}

    if GUN_LLM_PROVIDER == "openai":
        if OPENAI_API_KEY:
            result = _query_gun_llm_openai(annotated_frame, slot_ids)
            if any(v in ("plugged_in", "not_plugged_in") for v in result.values()):
                return result
        if GEMINI_API_KEY:
            return _query_gun_llm_gemini(annotated_frame, slot_ids)
        return {sid: "unclear" for sid in slot_ids}

    if GUN_LLM_PROVIDER == "gemini":
        if GEMINI_API_KEY:
            return _query_gun_llm_gemini(annotated_frame, slot_ids)
        logger.warning("[GUN-LLM] GUN_LLM_PROVIDER=gemini but GEMINI_API_KEY not set")
        return {sid: "unclear" for sid in slot_ids}

    logger.warning("[GUN-LLM] Unknown GUN_LLM_PROVIDER=%r — returning all-unclear",
                   GUN_LLM_PROVIDER)
    return {sid: "unclear" for sid in slot_ids}


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        # Backend dispatcher. Default is "llm" until the in-house gun YOLO is
        # reliable; flipping the env var to "yolo" reverts to the legacy path
        # without code changes. Both paths emit identical events.
        if GUN_DETECTION_BACKEND == "llm":
            return self._evaluate_llm(detection_output)
        return self._evaluate_yolo(detection_output)

    # ------------------------------------------------------------------
    # LLM backend
    # ------------------------------------------------------------------
    def _evaluate_llm(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id    = detection_output.get("camera_id", "unknown")
        task_id      = detection_output.get("_task_id")
        rois         = detection_output.get("rois") or {}
        snapshot_url = detection_output.get("snapshot_url")
        event_ts     = _event_ts(detection_output)
        now_dt       = _now_dt()

        events: List[dict] = []

        # ── Decide which slots need a poll this evaluation ────────────────
        # Plug-in candidates: occupied slots that haven't logged plug-in yet
        # and whose attempt budget isn't exhausted, AND whose last LLM check
        # was more than GUN_LLM_PLUGIN_INTERVAL_S ago.
        # Plug-out candidates: slots with plugin_logged but not plugout_logged,
        # whose last check was more than GUN_LLM_PLUGOUT_INTERVAL_S ago.
        plugin_candidates: List[str]  = []
        plugout_candidates: List[str] = []

        for roi_name in rois:
            slot = get_slot_state(camera_id, roi_name)
            if not slot["occupied"]:
                continue
            since_check = _seconds_since(slot.get("last_gun_check_at"), now_dt)

            if not slot["plugin_logged"]:
                if slot["gun_check_attempts"] >= GUN_LLM_PLUGIN_MAX_POLLS:
                    # Budget exhausted — give up on this slot for this lifecycle.
                    continue
                if (
                    slot.get("last_gun_check_at") is None
                    or since_check >= GUN_LLM_PLUGIN_INTERVAL_S
                ):
                    plugin_candidates.append(roi_name)
            elif not slot["plugout_logged"]:
                if (
                    slot.get("last_gun_check_at") is None
                    or since_check >= GUN_LLM_PLUGOUT_INTERVAL_S
                ):
                    plugout_candidates.append(roi_name)

        # Time-based inference: occupied slots that haven't logged plug-in
        # AND have been parked long enough that we should fire a synthetic
        # plug_time even if the LLM never confirmed visually. Built here so
        # we don't return early below when there's nothing to poll but
        # inference is still due.
        inference_candidates: List[str] = []
        for roi_name in rois:
            slot = get_slot_state(camera_id, roi_name)
            if not slot.get("occupied") or slot.get("plugin_logged"):
                continue
            since_intime = _seconds_since(slot.get("in_time"), now_dt)
            if since_intime >= GUN_LLM_INFERRED_PLUGIN_SECONDS:
                inference_candidates.append(roi_name)

        slots_to_poll = plugin_candidates + plugout_candidates
        if not slots_to_poll and not inference_candidates:
            return {"triggered": False, "matched_objects": [], "events": []}

        # ── One LLM call covering every polled slot ───────────────────────
        # The call is conditional — if slots_to_poll is empty (nothing due
        # for a poll, only inference) or the snapshot is missing, we skip the
        # LLM and still fall through to the inference pass below.
        verdicts: Dict[str, str] = {}
        if slots_to_poll:
            if not snapshot_url:
                logger.info(
                    "[GUN-LLM] camera=%s no snapshot_url — skipping LLM call (slots=%s)",
                    camera_id, slots_to_poll,
                )
            else:
                image = _download_snapshot(snapshot_url)
                if image is None:
                    logger.warning(
                        "[GUN-LLM] camera=%s snapshot download failed — skipping",
                        camera_id,
                    )
                else:
                    slot_polygons = {sid: rois[sid] for sid in slots_to_poll}
                    annotated = _annotate_slots(image, slot_polygons)
                    verdicts = _query_gun_llm(annotated, slots_to_poll)
                    logger.info(
                        "[GUN-LLM] camera=%s verdicts=%s plugin_targets=%s plugout_targets=%s",
                        camera_id, verdicts, plugin_candidates, plugout_candidates,
                    )

        # ── Apply verdicts to slot state and emit events ──────────────────
        for roi_name in slots_to_poll:
            slot     = get_slot_state(camera_id, roi_name)
            track_id = slot.get("track_id") or "unknown"
            verdict  = verdicts.get(roi_name, "unclear")
            slot["last_gun_check_at"] = now_dt.isoformat()

            if not slot["plugin_logged"]:
                # Plug-in hunt: count attempts whether or not the verdict
                # is conclusive — that's how we cap at MAX_POLLS.
                slot["gun_check_attempts"] += 1

                if verdict == "plugged_in":
                    slot["gun_consecutive_pluggedin_count"] += 1
                else:
                    slot["gun_consecutive_pluggedin_count"] = 0

                if slot["gun_consecutive_pluggedin_count"] >= GUN_LLM_PLUGIN_CONSECUTIVE:
                    plug_time = event_ts
                    gun_name  = _gun_name_for_roi(roi_name)
                    slot["plugin_logged"]         = True
                    slot["plug_time"]             = plug_time
                    slot["gun_name"]              = gun_name
                    slot["plug_time_inferred"]    = False
                    slot["gun_seen_after_plugin"] = True
                    slot["gun_consecutive_pluggedin_count"] = 0
                    # Reset cadence anchor so plug-out polling starts at the
                    # full interval after plug-in confirmation.
                    slot["last_gun_check_at"] = now_dt.isoformat()

                    evt = build_event(
                        event_type="gun_plugin",
                        camera_id=camera_id,
                        timestamp=plug_time,
                        track_id=track_id,
                        metadata={
                            "gun_name": gun_name,
                            "roi":      roi_name,
                            "slot_id":  roi_name,
                            "source":   "llm",
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info(
                        "[GUN-LLM] Plugin confirmed (2-of-%d): camera=%s roi=%s",
                        GUN_LLM_PLUGIN_CONSECUTIVE, camera_id, roi_name,
                    )
                elif slot["gun_check_attempts"] >= GUN_LLM_PLUGIN_MAX_POLLS:
                    # Budget exhausted with no confirmation — terminal.
                    # plug_time stays NULL; persistence will record an
                    # "incomplete" session row when the car eventually leaves.
                    logger.warning(
                        "[GUN-LLM] Plugin search exhausted (%d polls): camera=%s roi=%s",
                        GUN_LLM_PLUGIN_MAX_POLLS, camera_id, roi_name,
                    )

            elif not slot["plugout_logged"]:
                # Plug-out watch: 2-of-N to suppress single-frame flicker (the
                # LLM can mis-read the cable on occluded views; one false
                # not_plugged_in used to close the session prematurely).
                if verdict == "not_plugged_in":
                    slot["gun_consecutive_notpluggedin_count"] += 1
                else:
                    slot["gun_consecutive_notpluggedin_count"] = 0

                if slot["gun_consecutive_notpluggedin_count"] >= GUN_LLM_PLUGOUT_CONSECUTIVE:
                    plugout_time = event_ts
                    slot["plug_out_time"]  = plugout_time
                    slot["plugout_logged"] = True
                    slot["gun_consecutive_notpluggedin_count"] = 0

                    evt = build_event(
                        event_type="gun_plugout",
                        camera_id=camera_id,
                        timestamp=plugout_time,
                        track_id=track_id,
                        metadata={
                            "gun_name":     slot.get("gun_name") or _gun_name_for_roi(roi_name),
                            "roi":          roi_name,
                            "slot_id":      roi_name,
                            "plugin_time":  slot.get("plug_time"),
                            "plugout_time": plugout_time,
                            "source":       "llm",
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info(
                        "[GUN-LLM] Plugout confirmed (2-of-%d): camera=%s roi=%s",
                        GUN_LLM_PLUGOUT_CONSECUTIVE, camera_id, roi_name,
                    )

            set_slot_state(camera_id, roi_name, slot)

        # ── Inferred plug-in fallback ────────────────────────────────────
        # The LLM has imperfect recall on some camera angles — the cable can
        # be hard to see even when the OCPP energy meter shows charging is
        # happening. Fire a synthetic gun_plugin for any occupied slot that
        # has been parked past the inference threshold without a confirmed
        # plug-in. The event is marked source=inferred so downstream knows
        # it's a fallback rather than a visual confirmation, and plug_time
        # is anchored at parking_intime + threshold (a reasonable proxy for
        # when charging actually started).
        for roi_name in inference_candidates:
            slot = get_slot_state(camera_id, roi_name)
            if slot.get("plugin_logged"):
                # The LLM call above may have just confirmed plug-in for
                # this slot. Honor that real verdict over inference.
                continue

            in_time_dt = _parse_iso(slot.get("in_time"))
            if in_time_dt is None:
                continue
            inferred_plug_dt = in_time_dt + timedelta(seconds=GUN_LLM_INFERRED_PLUGIN_SECONDS)
            plug_time = inferred_plug_dt.isoformat()
            track_id  = slot.get("track_id") or "unknown"
            gun_name  = _gun_name_for_roi(roi_name)

            slot["plugin_logged"]      = True
            slot["plug_time"]          = plug_time
            slot["plug_time_inferred"] = True
            slot["gun_name"]           = gun_name
            slot["last_gun_check_at"]  = now_dt.isoformat()
            slot["gun_consecutive_pluggedin_count"] = 0

            evt = build_event(
                event_type="gun_plugin",
                camera_id=camera_id,
                timestamp=plug_time,
                track_id=track_id,
                metadata={
                    "gun_name": gun_name,
                    "roi":      roi_name,
                    "slot_id":  roi_name,
                    "source":   "inferred_llm",
                    "reason":   (
                        f"no LLM-confirmed plug-in within "
                        f"{GUN_LLM_INFERRED_PLUGIN_SECONDS}s of parking_intime"
                    ),
                },
            )
            events.append(evt)
            publish_sync("gun_events", evt, task_id=task_id)
            logger.warning(
                "[GUN-LLM] Plug-in inferred (no visual confirmation in %ds): "
                "camera=%s roi=%s plug_time=%s",
                GUN_LLM_INFERRED_PLUGIN_SECONDS, camera_id, roi_name, plug_time,
            )
            set_slot_state(camera_id, roi_name, slot)

        triggered = bool(events)
        return {"triggered": triggered, "matched_objects": [], "events": events}

    # ------------------------------------------------------------------
    # YOLO backend (legacy)
    # ------------------------------------------------------------------
    def _evaluate_yolo(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        task_id   = detection_output.get("_task_id")
        rois      = detection_output.get("rois", {})
        all_dets  = detection_output.get("detections", [])
        # Anchor every emitted event to the frame's capture time. Wall-clock
        # debounces (now_dt below) keep using _now_dt().
        event_ts  = _event_ts(detection_output)

        # Tracked cars are injected by the engine after parking_detection runs.
        # Falls back to raw car detections if not yet available (e.g. first frame).
        tracked_cars = detection_output.get("tracked_cars") or [
            d for d in all_dets if d.get("class_name") == CAR_CLASS
        ]

        print(f"[GUN] camera={camera_id} | rois={list(rois.keys())} | tracked_cars={len(tracked_cars)}")

        # ── Map cars → ROIs ───────────────────────────────────────────────
        # One car per ROI (first car wins if multiple overlap the same ROI).
        roi_to_track: Dict[str, str] = {}
        if rois:
            for car in tracked_cars:
                for roi_name in which_rois(car["bbox"], rois):
                    if roi_name not in roi_to_track:
                        roi_to_track[roi_name] = car.get("track_id", "unknown")

        # ── Map guns → ROIs ───────────────────────────────────────────────
        # Keep the highest-confidence gun per ROI.
        gun_dets = [d for d in all_dets if d.get("class_name") == GUN_CLASS]
        roi_to_gun: Dict[str, dict] = {}
        for gun in gun_dets:
            for roi_name in (which_rois(gun.get("bbox", {}), rois) if rois else []):
                if roi_name not in roi_to_gun or gun.get("confidence", 0) > roi_to_gun[roi_name].get("confidence", 0):
                    roi_to_gun[roi_name] = gun

        print(f"[GUN] roi_to_track={roi_to_track} | roi_to_gun={list(roi_to_gun.keys())}")

        events: List[dict] = []

        # ── Process each ROI independently ───────────────────────────────
        now_dt = _now_dt()

        for roi_name in rois:
            slot     = get_slot_state(camera_id, roi_name)
            gun_det  = roi_to_gun.get(roi_name)
            gun_name = _gun_name_for_roi(roi_name)
            track_id = roi_to_track.get(roi_name, slot.get("track_id") or "unknown")

            # Only process gun logic when the slot is confirmed occupied.
            # If parking_detection hasn't confirmed a car here yet, skip.
            if not slot["occupied"]:
                continue

            if gun_det:
                # ── Gun visible this frame ──────────────────────────────
                slot["gun_present_frames"] += 1
                slot["gun_absent_frames"]   = 0  # legacy counter, kept for back-compat
                slot["gun_name"]            = slot["gun_name"] or gun_name

                # Real plug-in path: gun confirmed for GUN_PLUGIN_FRAMES.
                # If the slot was on an inferred plug, this real detection
                # confirms the inference — we keep the inferred timestamp
                # (it's earlier and more representative of when the user
                # actually plugged in) but flip the verification flag.
                if not slot["plugin_logged"] and slot["gun_present_frames"] >= GUN_PLUGIN_FRAMES:
                    plug_time              = event_ts
                    slot["plugin_logged"]  = True
                    slot["plug_time"]      = plug_time
                    slot["gun_name"]       = gun_name
                    slot["gun_present_frames"]    = 0
                    slot["plug_time_inferred"]    = False
                    slot["gun_seen_after_plugin"] = True

                    evt = build_event(
                        event_type="gun_plugin",
                        camera_id=camera_id,
                        timestamp=plug_time,
                        track_id=track_id,
                        metadata={
                            "gun_name": gun_name,
                            "roi":      roi_name,
                            "slot_id":  roi_name,
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info("[GUN] Plugin: camera=%s roi=%s gun=%s", camera_id, roi_name, gun_name)

                # Already plugged in (real or inferred). A real gun frame
                # confirms the plug is genuine and clears any in-progress
                # plug-out debounce — the gun is back, so MAYBE_OUT was a
                # false alarm (occlusion).
                elif slot["plugin_logged"] and not slot["plugout_logged"]:
                    slot["gun_seen_after_plugin"] = True

                    if slot.get("gun_maybe_out_since"):
                        logger.info(
                            "[GUN] MAYBE_OUT cleared (gun returned): camera=%s roi=%s",
                            camera_id, roi_name,
                        )
                        slot["gun_maybe_out_since"] = None

            else:
                # ── Gun NOT visible this frame ──────────────────────────
                slot["gun_present_frames"] = 0

                # Inferred plug-in path: no real gun seen, but car has been
                # parked long enough that we synthesize a plug.
                if (
                    not slot["plugin_logged"]
                    and slot.get("in_time")
                ):
                    parked_secs = _seconds_since(slot["in_time"], now_dt)
                    if parked_secs >= INFERRED_PLUGIN_SECONDS:
                        plug_time = event_ts
                        slot["plugin_logged"]  = True
                        slot["plug_time"]      = plug_time
                        slot["gun_name"]       = slot["gun_name"] or gun_name
                        slot["plug_time_inferred"]    = True
                        slot["gun_seen_after_plugin"] = False

                        evt = build_event(
                            event_type="gun_plugin",
                            camera_id=camera_id,
                            timestamp=plug_time,
                            track_id=track_id,
                            metadata={
                                "gun_name": slot["gun_name"] or gun_name,
                                "roi":      roi_name,
                                "slot_id":  roi_name,
                                "inferred": True,
                                "reason":   "no gun detection within INFERRED_PLUGIN_SECONDS of parking_intime",
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt, task_id=task_id)
                        logger.info(
                            "[GUN] Plugin (inferred, %.0fs after intime): camera=%s roi=%s",
                            parked_secs, camera_id, roi_name,
                        )

                # Plug-out debounce — only runs after a confirmed plug.
                # plugin_logged is NEVER cleared by a missed frame here; it
                # only flips off via the rollback below or via reset_slot_state.
                elif slot["plugin_logged"] and not slot["plugout_logged"]:
                    plug_dt = _parse_iso(slot.get("plug_time"))
                    secs_since_plug = (
                        (now_dt - plug_dt).total_seconds() if plug_dt else 0.0
                    )

                    # Inference rollback: if the plug was inferred and we have
                    # never actually seen the gun since, give up after the
                    # verification window. The car parked but never plugged
                    # in (or the gun is permanently in a blind spot — either
                    # way we can't confirm, so don't synthesize a plugout
                    # later either).
                    if (
                        slot.get("plug_time_inferred")
                        and not slot.get("gun_seen_after_plugin")
                        and secs_since_plug >= INFERRED_VERIFY_TIMEOUT_SECONDS
                    ):
                        logger.warning(
                            "[GUN] Inferred plug rolled back (gun never observed in %.0fs): "
                            "camera=%s roi=%s",
                            secs_since_plug, camera_id, roi_name,
                        )
                        slot["plugin_logged"]         = False
                        slot["plug_time"]             = None
                        slot["plug_time_inferred"]    = False
                        slot["gun_seen_after_plugin"] = False
                        slot["gun_maybe_out_since"]   = None
                        # No event fired; the orchestration row never got a
                        # plug_time for this slot, so there is nothing to undo
                        # downstream. (We did fire gun_plugin earlier — but
                        # the upsert is first-write-wins, so a future real
                        # plug for this same slot+session won't overwrite.
                        # Acceptable: this slot's row simply keeps the
                        # inferred plug_time, which is conservative.)
                        set_slot_state(camera_id, roi_name, slot)
                        continue

                    # Plug-out state machine.
                    #
                    # PLUGGED  → after GUN_MAYBE_OUT_SECONDS of absence,
                    #            enter MAYBE_OUT (no event yet).
                    # MAYBE_OUT → if gun reappears, exit entirely (handled
                    #             in the gun_det branch above; that's the
                    #             "miss → return" half of the user's pattern).
                    # MAYBE_OUT → after GUN_RETURN_GRACE_SECONDS +
                    #             GUN_CONFIRM_OUT_SECONDS of continued absence
                    #             without a return, fire gun_plugout (the
                    #             "miss → return → miss again" pattern only
                    #             reaches this point if the second miss is
                    #             sustained, since any return resets state).
                    maybe_since = slot.get("gun_maybe_out_since")
                    if not maybe_since:
                        # PLUGGED → MAYBE_OUT after sustained absence.
                        # last_gun_seen_at is the anchor; on the first plug
                        # cycle there might not be one yet, so fall back to
                        # plug_time which marks the start of "should be visible".
                        last_seen = (
                            slot.get("last_gun_seen_at")
                            or slot.get("plug_time")
                        )
                        absent_secs = _seconds_since(last_seen, now_dt)
                        if absent_secs >= GUN_MAYBE_OUT_SECONDS:
                            slot["gun_maybe_out_since"] = now_dt.isoformat()
                            logger.info(
                                "[GUN] Entered MAYBE_OUT (absent %.1fs): camera=%s roi=%s",
                                absent_secs, camera_id, roi_name,
                            )
                    else:
                        # Already in MAYBE_OUT and gun is still absent (we're
                        # in the not-gun_det branch). Time accumulates; only
                        # fire once both windows have elapsed.
                        in_maybe_secs = _seconds_since(maybe_since, now_dt)
                        ready_to_fire = in_maybe_secs >= (
                            GUN_RETURN_GRACE_SECONDS + GUN_CONFIRM_OUT_SECONDS
                        )

                        if ready_to_fire:
                            plugout_time = event_ts
                            slot["plug_out_time"]  = plugout_time
                            slot["plugout_logged"] = True
                            # Re-arm for a possible second plug cycle in this slot
                            slot["plugin_logged"]         = False
                            slot["plugout_logged"]        = False
                            slot["gun_maybe_out_since"]   = None
                            slot["gun_absent_frames"]     = 0
                            slot["gun_present_frames"]    = 0
                            slot["plug_time_inferred"]    = False
                            slot["gun_seen_after_plugin"] = False

                            evt = build_event(
                                event_type="gun_plugout",
                                camera_id=camera_id,
                                timestamp=plugout_time,
                                track_id=track_id,
                                metadata={
                                    "gun_name":     slot["gun_name"] or gun_name,
                                    "roi":          roi_name,
                                    "slot_id":      roi_name,
                                    "plugin_time":  slot["plug_time"],
                                    "plugout_time": plugout_time,
                                },
                            )
                            events.append(evt)
                            publish_sync("gun_events", evt, task_id=task_id)
                            logger.info(
                                "[GUN] Plugout (debounced %.0fs): camera=%s roi=%s",
                                in_maybe_secs, camera_id, roi_name,
                            )

            # Always update last_gun_seen_at while gun is visible — this is
            # the anchor the plug-out debounce uses to measure absence.
            if gun_det:
                slot["last_gun_seen_at"] = now_dt.isoformat()

            set_slot_state(camera_id, roi_name, slot)

        # ── Unauthorized attribution ──────────────────────────────────────
        # If a gun sits in an ROI whose slot is NOT occupied, and exactly one
        # tracked car is currently outside every defined ROI, attribute the
        # gun to that unauthorized car. We emit a lightweight
        # "gun_unauthorized" event (gun_number only — no plug_time, no
        # plug_out_time, no slot_id) so the orchestration layer can attach
        # gun_number to the unauthorized charging session.
        #
        # Conservative matching: skip when there are 0 or >1 unauthorized
        # cars — guessing which car the gun belongs to would mis-attribute.
        if rois:
            unauthorized_cars = [
                c for c in tracked_cars
                if not which_rois(c.get("bbox", {}), rois)
                and c.get("track_id")
            ]
            unoccupied_gun_rois = [
                roi_name for roi_name, gun in roi_to_gun.items()
                if not get_slot_state(camera_id, roi_name)["occupied"]
            ]
            if len(unauthorized_cars) == 1 and unoccupied_gun_rois:
                ua_track_id = unauthorized_cars[0]["track_id"]
                for roi_name in unoccupied_gun_rois:
                    gun_name = _gun_name_for_roi(roi_name)
                    evt = build_event(
                        event_type="gun_unauthorized",
                        camera_id=camera_id,
                        timestamp=event_ts,
                        track_id=ua_track_id,
                        metadata={
                            "gun_name": gun_name,
                            "roi":      roi_name,
                            "source":   "unauthorized_parking",
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info(
                        "[GUN] Unauthorized gun attribution: camera=%s gun=%s track=%s",
                        camera_id, gun_name, ua_track_id,
                    )

        triggered = bool(events)
        print(f"[GUN] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": list(gun_dets), "events": events}
