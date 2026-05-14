"""
Parking Compliance Rule
========================
Checks for three compliance violations per car:

  A. Unauthorized parking — car parked outside any designated charging slot.
  B. Wrong parking        — car straddles two slots (double-slot occupancy).
  C. Non-EV parking       — non-EV vehicle occupying an EV charging slot.

Source-of-truth hierarchy:

  1. The frame-level LLM verdict (parking_quality + is_ev) wins when present.
     vehicle_extraction's cached_llm_frame_call gives one normalised response
     per snapshot, shared with every rule via an in-process cache — so
     consulting it here is free after the first call.
  2. Geometric ROI overlap is the fallback when the LLM produced no verdict
     for this track (no snapshot URL, no API key, or LLM saw nothing in the
     vehicle's position).

For unauthorized parking, this rule also fires parking session events so the
full session lifecycle (in_time → out_time) is captured even when the car
never enters a legitimate ROI:

  - parking_intime  fired once when an unauthorized car is confirmed present
                    for UNAUTH_DWELL_SECONDS of wall-clock time
  - parking_outtime fired once when the same car has been absent for
                    EXIT_FRAMES consecutive frames

This ensures the dashboard session table is populated for all vehicles,
not only those that park inside a configured ROI.

State is persisted in Redis key ``compliance:<camera_id>``.

ROI polygons are NOT hardcoded here.
The orchestrator injects them via detection_output["rois"]:
    {
        "ROI_1": [[x, y], ...],
        "ROI_2": [[x, y], ...]
    }
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

from shared.common.roi import which_rois, which_rois_bbox_overlap
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
from usecase.rules.vehicles.vehicle_extraction import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DETECTION_W,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    _annotate_all_slots,
    _assign_vehicles_to_targets,
    _b64_jpeg,
    _bbox_center_x,
    _download_image,
    cached_llm_frame_call,
)
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3   # consecutive frames car must be outside ROI to confirm unauthorized entry (wrong_parking gate)
EXIT_FRAMES  = 3   # consecutive frames car must be absent to confirm exit
MIN_CAR_CONFIDENCE = 0.5  # drop low-confidence detections before evaluating compliance
# Wrong-parking threshold: a neighbour ROI is only counted as "occupied by this car"
# when ≥50 % of the bbox sample grid falls inside it. Tuned for Indian parking — a
# car parked askew but mostly inside its own slot will not alert; a car whose body
# is genuinely half in the next slot will.
WRONG_PARKING_OVERLAP = 0.50
# Unauthorized parking is dwell-gated: a car must sit outside every ROI for at
# least UNAUTH_DWELL_SECONDS of wall-clock time before the violation fires.
# Frame-count debouncing was too aggressive — a car entering or leaving a slot
# spends a few seconds with its centroid in the driveway, which used to fire a
# false unauthorized_parking. Dwell time eliminates the in-transit case.
UNAUTH_DWELL_SECONDS = 60

# Containment threshold for the "duplicate-detection" filter. When YOLO
# multi-detects a single physical car (a fragment like a bumper or hood gets
# its own bbox + track_id), the fragment's bbox is mostly contained within the
# main car's bbox even if its centroid happens to lie outside every ROI. We
# measure intersection_area / min(bbox_area_a, bbox_area_b) and skip the
# unauthorized-parking branch when this is above the threshold — the "outside"
# track is almost certainly a duplicate of an in-slot car. 0.5 = the smaller
# bbox is at least half-inside the larger one.
DUPLICATE_DETECTION_CONTAINMENT = float(os.getenv("UNAUTH_DUPLICATE_CONTAINMENT", "0.5"))

# Merged-bbox guard. When two physical cars park close together, YOLO can emit
# a single bbox covering both — the centroid lands inside whichever ROI
# happens to win, and the rule then labels the merged blob as "proper". We
# detect this by comparing the track's bbox width to the narrowest slot's
# width: a properly-parked car sits inside one slot (ratio < ~1.0), a
# merged-pair bbox spans ≥ this ratio of the slot width. Default 1.15 was
# tuned against a real merged-pair frame at ratio 1.28 while clean single-car
# bboxes in the same camera came in at 0.77–1.04. When tripped we force the
# verdict to double_slot so wrong_parking fires through the existing
# debounce path.
MERGED_BBOX_SLOT_RATIO = float(os.getenv("MERGED_BBOX_SLOT_RATIO", "1.15"))

# Use the dedicated compliance LLM call as an arbiter when the geometric
# path or the merge-guard flags a suspect verdict. On by default; set to 0
# to disable and fall back to geometry only.
COMPLIANCE_LLM_ENABLED = os.getenv("COMPLIANCE_LLM_ENABLED", "1") == "1"

# Which provider to use for compliance arbitration. "openai" (default) or
# "claude". Mirrors VEHICLE_LLM_PROVIDER / GUN_LLM_PROVIDER.
COMPLIANCE_LLM_PROVIDER = os.getenv("COMPLIANCE_LLM_PROVIDER", "openai").strip().lower()

# Compliance prompt. The frame has yellow polygon outlines overlaid by
# _annotate_all_slots; the LLM does not see slot ID labels. Vehicles are
# referenced by visual position (left / center / right) and the code maps
# position -> ROI ID before emitting violations.
_COMPLIANCE_PROMPT = """You are a STRICT parking-compliance auditor for an EV charging station.

You are given ONE CCTV frame. Parking slots are outlined as YELLOW polygons. There are NO slot ID labels in the image — refer to each slot only by its visual position (left, center, right).

Your job is to judge — for EACH visible vehicle — whether it is parked correctly inside a single slot, occupying two slots, or outside every slot.

==================================================
VERDICTS
==================================================
- "proper" — the vehicle's body is FULLY inside ONE yellow polygon. The car may be slightly askew, but no part of its body crosses a yellow line.
- "occupies_two_slots" — ONE vehicle's body overlaps TWO yellow polygons. ALL of these cases qualify:
    * a forward-parked car straddling the divider between two slots,
    * a car parked horizontally (sideways / perpendicular) covering two slots end-to-end,
    * a car parked diagonally across two slots.
  Orientation does not matter. If any part of the body lies in one slot and any other part lies in another slot, the verdict is "occupies_two_slots".
- "outside_all_slots" — the vehicle's body is entirely OUTSIDE every yellow polygon (parked in the driveway, blocking access, etc.).
- "partially_outside" — the vehicle is partly inside ONE slot and partly outside every slot (e.g. tail sticking into driveway).
- "insufficient_evidence" — the vehicle is heavily occluded by people, other cars, or the frame edge such that you cannot judge its position with confidence. NEVER GUESS.

==================================================
RULES
==================================================
1. List EVERY visible vehicle, including those that look properly parked. The downstream system needs occupancy counts to detect when YOLO has merged two cars into one detection.
2. If you see TWO distinct vehicles whose bodies both overlap the SAME yellow polygon, BOTH must be reported, and `slot_occupancy` for that position must be 2. Two cars in one slot is a compliance failure even when each looks "proper" individually.
3. Count carefully: do not collapse two adjacent vehicles into one "occupies_two_slots" verdict. If you can see two distinct vehicles (two roofs, two number plates, two pairs of wheels), report two vehicles each with their own verdict.

==================================================
OUTPUT FORMAT
==================================================
Return ONLY valid JSON matching this schema:

{
  "vehicles": [
    {
      "position": "left" | "center" | "right" | "left_and_right" | "multiple",
      "verdict": "proper" | "occupies_two_slots" | "outside_all_slots" | "partially_outside" | "insufficient_evidence",
      "confidence": 0.0-1.0
    }
  ],
  "slot_occupancy": {
    "left":  0,
    "right": 0
  }
}

`slot_occupancy` is keyed by visual position (left / right / center) and gives the integer count of distinct vehicles whose body overlaps that slot's yellow polygon. Only include keys for slots that are visible in the frame.

If no vehicle is visible:
{ "vehicles": [], "slot_occupancy": {} }
"""

_COMPLIANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position":   {"type": "string"},
                    "verdict":    {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["position", "verdict", "confidence"],
                "additionalProperties": False,
            },
        },
        "slot_occupancy": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    },
    "required": ["vehicles", "slot_occupancy"],
    "additionalProperties": False,
}

_COMPLIANCE_CACHE: Dict[str, Dict[str, Any]] = {}
_COMPLIANCE_CACHE_ORDER: List[str] = []
_COMPLIANCE_CACHE_MAX = 16

# Short-label descriptions surfaced on the dashboard and forwarded to the
# client's fine system. Two or three words; slot IDs live in metadata.
_DESC_UNAUTHORIZED       = "Unauthorised parking"
_DESC_DOUBLE_SLOT        = "Double slot parking"
_DESC_NON_EV             = "Non-EV vehicle"
_DESC_UNAUTHORIZED_NONEV = "Unauthorised non-EV"


def _empty_compliance_response() -> Dict[str, Any]:
    return {"vehicles": [], "slot_occupancy": {}}


def _query_claude_compliance(
    image, rois: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """Send the annotated frame to Claude with the compliance prompt. Returns
    the empty response on any failure so callers degrade gracefully."""
    if not ANTHROPIC_API_KEY:
        return _empty_compliance_response()
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        annotated = _annotate_all_slots(image, rois)
        frame_b64 = _b64_jpeg(annotated, quality=95)
        logger.info("[COMPLIANCE-LLM] Sending frame to Claude (model=%s)", ANTHROPIC_MODEL)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _COMPLIANCE_PROMPT},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_b64,
                        },
                    },
                ],
            }],
            output_config={"format": {"type": "json_schema", "schema": _COMPLIANCE_SCHEMA}},
        )
        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()
        logger.info("[COMPLIANCE-LLM] Claude raw response: %s", text)
        if not text:
            return _empty_compliance_response()
        parsed = json.loads(text)
        if not isinstance(parsed.get("vehicles"), list):
            return _empty_compliance_response()
        return parsed
    except Exception as exc:
        logger.warning("[COMPLIANCE-LLM] Claude call failed: %s", exc)
        return _empty_compliance_response()


def _query_openai_compliance(
    image, rois: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """Send the annotated frame to OpenAI with the compliance prompt. Returns
    the empty response on any failure so callers degrade gracefully."""
    if not OPENAI_API_KEY:
        return _empty_compliance_response()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        annotated = _annotate_all_slots(image, rois)
        frame_b64 = _b64_jpeg(annotated, quality=95)
        logger.info("[COMPLIANCE-LLM] Sending frame to OpenAI (model=%s)", OPENAI_MODEL)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _COMPLIANCE_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"}},
                ],
            }],
        )
        text = (response.choices[0].message.content or "").strip()
        logger.info("[COMPLIANCE-LLM] OpenAI raw response: %s", text)
        if not text:
            return _empty_compliance_response()
        parsed = json.loads(text)
        if not isinstance(parsed.get("vehicles"), list):
            return _empty_compliance_response()
        return parsed
    except Exception as exc:
        logger.warning("[COMPLIANCE-LLM] OpenAI call failed: %s", exc)
        return _empty_compliance_response()


def cached_compliance_llm_call(
    snapshot_url: str,
    rois: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """One compliance LLM call per snapshot URL, shared across all cars in
    the same evaluate cycle. Provider is selected by COMPLIANCE_LLM_PROVIDER."""
    if not snapshot_url or not COMPLIANCE_LLM_ENABLED:
        return _empty_compliance_response()
    cached = _COMPLIANCE_CACHE.get(snapshot_url)
    if cached is not None:
        return cached
    image = _download_image(snapshot_url)
    if image is None:
        return _empty_compliance_response()
    if COMPLIANCE_LLM_PROVIDER == "claude":
        resp = _query_claude_compliance(image, rois)
    else:
        resp = _query_openai_compliance(image, rois)
    _COMPLIANCE_CACHE[snapshot_url] = resp
    _COMPLIANCE_CACHE_ORDER.append(snapshot_url)
    while len(_COMPLIANCE_CACHE_ORDER) > _COMPLIANCE_CACHE_MAX:
        old = _COMPLIANCE_CACHE_ORDER.pop(0)
        _COMPLIANCE_CACHE.pop(old, None)
    return resp


def _polygon_x_extent(polygon: List) -> float:
    """Return max_x - min_x of a polygon's vertices, or 0 if malformed."""
    try:
        xs = [float(p[0]) for p in polygon if len(p) >= 2]
    except (TypeError, ValueError):
        return 0.0
    return (max(xs) - min(xs)) if xs else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_ts(detection_output: Dict[str, Any]) -> str:
    """
    Pick the timestamp that should anchor events emitted from this evaluation.
    Prefers the camera-stamped frame timestamp; falls back to wall-clock.
    """
    ts = detection_output.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return _now()


def _bbox_containment(a: dict, b: dict) -> float:
    """Intersection area divided by the smaller bbox's area.

    Returns a value in [0, 1]. 1.0 means the smaller bbox is entirely inside
    the larger one. Used to detect YOLO multi-detections of the same physical
    car: a fragment bbox (bumper, hood) is mostly contained within the main
    car's bbox even when its centroid happens to fall outside every ROI.
    """
    if not a or not b:
        return 0.0
    keys = ("x1", "y1", "x2", "y2")
    if not all(k in a for k in keys) or not all(k in b for k in keys):
        return 0.0
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    b_area = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    smaller = min(a_area, b_area)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp; tolerant of None / malformed input."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ParkingComplianceRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_compliance"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois")
        event_ts = _event_ts(detection_output)
        if not rois:
            logger.error("[COMPLIANCE] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "violations": [], "events": []}

        # Read the canonical tracked-car list populated by parking_detection.
        # Falling back to raw detections only when the tracker output is missing
        # (e.g. parking_detection disabled) keeps the rule usable in isolation,
        # but the tracked path is what filters out single-frame false positives
        # on painted ground markings — a real car persists across frames and
        # gets a stable track_id; a flicker does not.
        tracked_cars = detection_output.get("tracked_cars")
        if tracked_cars is None:
            tracked_cars = [
                d for d in detection_output.get("detections", [])
                if d.get("class_name") == "car"
            ]
        cars = [
            c for c in tracked_cars
            if (c.get("confidence") or 0.0) >= MIN_CAR_CONFIDENCE
        ]
        print(
            f"[COMPLIANCE] camera={camera_id} | rois={list(rois.keys())}"
            f" | tracked_cars={len(tracked_cars)} | cars_after_conf={len(cars)}"
        )

        # --- Load Redis state for session tracking of unauthorized cars ---
        redis_key = f"compliance:{camera_id}"
        state = get_state(redis_key)
        # state structure per track_id:
        # {
        #   "<track_id>": {
        #     "outside_since": str,  ISO timestamp of first frame seen outside any ROI
        #     "wrong_buf":     int,  consecutive frames straddling >1 ROI
        #     "exit_buf":      int,  consecutive frames absent
        #     "occupied":      bool, intime has been fired
        #     "violated":      bool, unauthorized_parking violation already fired
        #     "wrong_fired":   bool, wrong_parking violation already fired
        #     "non_ev_fired":  bool, non_ev_parking violation already fired
        #     "intime":        str,  ISO timestamp of intime event
        #   }
        # }
        event_dt = _parse_iso(event_ts)

        # ── LLM verdicts (one frame call, cached and shared with vehicle_extraction)
        # When ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY is unset, or
        # the snapshot URL is missing, llm_vehicles ends up empty and assignments
        # is {} — every car falls through to the geometric fallback below.
        snapshot_url = detection_output.get("snapshot_url")
        llm_resp = cached_llm_frame_call(snapshot_url, rois) if snapshot_url else None
        llm_vehicles = (llm_resp or {}).get("vehicles", [])
        llm_targets = [
            {"key": c.get("track_id"),
             "x_center": _bbox_center_x(c.get("bbox") or {})}
            for c in cars if c.get("track_id")
        ]
        llm_assignments = (
            _assign_vehicles_to_targets(llm_vehicles, llm_targets, float(DETECTION_W))
            if (llm_vehicles and llm_targets) else {}
        )

        violations: List[dict] = []
        events: List[dict] = []
        flagged: List[dict] = []

        # Track which track_ids are active this frame (outside all ROIs)
        active_unauthorized: set = set()

        # Narrowest slot width in pixels — used to detect YOLO bboxes that
        # have merged two adjacent cars into one detection. Falls back to 0
        # (disabling the merged-bbox check) when ROI polygons are malformed.
        slot_widths = [_polygon_x_extent(poly) for poly in rois.values()]
        slot_widths = [w for w in slot_widths if w > 0]
        min_slot_w = min(slot_widths) if slot_widths else 0.0

        # Pre-pass: collect bboxes of any car that overlaps a slot this frame.
        # YOLO can multi-detect a single physical car (one bbox in the slot,
        # a second fragment bbox outside the slot). The unauthorized-parking
        # branch below uses these to skip phantom "outside" tracks that are
        # spatially the same vehicle.
        in_slot_bboxes: List[dict] = []
        for c in cars:
            bb = c.get("bbox") or {}
            if not all(k in bb for k in ("x1", "y1", "x2", "y2")):
                continue
            if which_rois(bb, rois) or which_rois_bbox_overlap(
                bb, rois, overlap_threshold=WRONG_PARKING_OVERLAP,
            ):
                in_slot_bboxes.append(bb)

        if not cars:
            print(f"[COMPLIANCE] no cars — skipping violation check")
        else:
            for car in cars:
                track_id = car.get("track_id", "unknown")
                # Centroid-based check: which ROI the car's centre is in.
                matched_rois = which_rois(car["bbox"], rois)
                # Overlap-based check: a car whose centroid sits in ROI_1 but
                # whose body is genuinely half in ROI_2 still qualifies as
                # wrong-parking under the geometric fallback. 50 % threshold —
                # a car parked askew but mostly in its own slot won't alert.
                overlap_rois = which_rois_bbox_overlap(
                    car["bbox"], rois, overlap_threshold=WRONG_PARKING_OVERLAP,
                )
                all_matched_rois = list(dict.fromkeys(matched_rois + overlap_rois))

                llm_vehicle = llm_assignments.get(track_id)
                llm_quality = (llm_vehicle or {}).get("parking_quality")
                llm_is_ev  = (llm_vehicle or {}).get("is_ev")

                # ── Decide authoritative location verdict ────────────────
                # LLM wins when it has a non-"unknown" parking_quality for
                # this track. Otherwise fall back to geometric overlap.
                if llm_quality in ("proper", "double_slot", "across_line", "outside_slot"):
                    verdict = llm_quality
                    verdict_source = "llm"
                else:
                    if len(all_matched_rois) == 0:
                        verdict = "outside_slot"
                    elif len(all_matched_rois) > 1:
                        verdict = "double_slot"
                    else:
                        verdict = "proper"
                    verdict_source = "geom"

                # ── Merged-bbox override ─────────────────────────────────
                # If the track's bbox is wider than MERGED_BBOX_SLOT_RATIO ×
                # the narrowest slot, YOLO has almost certainly merged two
                # adjacent cars into one detection. Both LLM and geometry
                # then mislabel the merged blob as "proper" in whichever slot
                # its centroid happens to fall in. Force double_slot so the
                # wrong_parking debounce path fires.
                bb = car.get("bbox") or {}
                bbox_w = (
                    float(bb["x2"]) - float(bb["x1"])
                    if all(k in bb for k in ("x1", "x2")) else 0.0
                )
                slot_ratio = (bbox_w / min_slot_w) if min_slot_w > 0 else 0.0
                if min_slot_w > 0 and slot_ratio >= MERGED_BBOX_SLOT_RATIO:
                    verdict = "double_slot"
                    verdict_source = "merged_bbox"

                # ── Claude compliance arbiter ─────────────────────────────
                # When the verdict is non-proper (or the merge-guard tripped),
                # consult the dedicated Claude compliance call. Its verdict
                # wins because it can reason about both painted slot lines AND
                # vehicle orientation in ways the geometric path cannot.
                # `insufficient_evidence` falls back to the geometric verdict.
                arbiter_verdict = None
                arbiter_confidence = None
                if (
                    COMPLIANCE_LLM_ENABLED
                    and snapshot_url
                    and verdict != "proper"
                ):
                    comp_resp = cached_compliance_llm_call(snapshot_url, rois)
                    comp_vehicles = comp_resp.get("vehicles", [])
                    comp_assignments = (
                        _assign_vehicles_to_targets(
                            comp_vehicles, llm_targets, float(DETECTION_W),
                        ) if (comp_vehicles and llm_targets) else {}
                    )
                    comp_vehicle = comp_assignments.get(track_id)
                    raw = (comp_vehicle or {}).get("verdict")
                    arbiter_confidence = (comp_vehicle or {}).get("confidence")
                    mapped = {
                        "proper": "proper",
                        "occupies_two_slots": "double_slot",
                        "outside_all_slots": "outside_slot",
                        "partially_outside": "outside_slot",
                    }.get(raw)
                    if mapped is not None:
                        verdict = mapped
                        verdict_source = "compliance_llm"
                        arbiter_verdict = raw

                print(
                    f"[COMPLIANCE] car track={track_id} | matched_rois={matched_rois}"
                    f" | overlap_rois={overlap_rois} | bbox_w={bbox_w:.0f}"
                    f" | slot_ratio={slot_ratio:.2f} | verdict={verdict}"
                    f" ({verdict_source}) | arbiter={arbiter_verdict}"
                    f" conf={arbiter_confidence} | is_ev={llm_is_ev}"
                )

                slot = state.setdefault(track_id, {
                    "outside_since": None, "wrong_buf": 0, "exit_buf": 0,
                    "occupied": False, "violated": False, "wrong_fired": False,
                    "non_ev_fired": False, "intime": None,
                })

                if verdict == "outside_slot":
                    # Duplicate-detection guard only runs in the geometric
                    # fallback. When the LLM has identified a vehicle at this
                    # position, the track is real by construction — no need
                    # to filter YOLO fragments.
                    if verdict_source == "geom":
                        bb = car.get("bbox") or {}
                        best_containment = max(
                            (_bbox_containment(bb, slot_bb) for slot_bb in in_slot_bboxes),
                            default=0.0,
                        )
                        if best_containment >= DUPLICATE_DETECTION_CONTAINMENT:
                            logger.info(
                                "[COMPLIANCE] Skipping unauthorized for track=%s — "
                                "duplicate of in-slot car (containment=%.2f)",
                                track_id, best_containment,
                            )
                            slot["outside_since"] = None
                            slot["wrong_buf"] = 0
                            slot["exit_buf"] = 0
                            continue

                    # ── Unauthorized parking (dwell-gated) ────────────────────
                    # The violation only fires after the same tracked car has
                    # been outside every ROI for UNAUTH_DWELL_SECONDS of
                    # wall-clock time. A car *entering* or *leaving* a slot
                    # spends a few seconds in transit with its centroid in the
                    # driveway — dwell-gating filters those out. A genuinely
                    # abandoned car sits there well past the threshold.
                    active_unauthorized.add(track_id)
                    slot["exit_buf"] = 0
                    slot["wrong_buf"] = 0
                    if not slot.get("outside_since"):
                        slot["outside_since"] = event_ts

                    outside_since_dt = _parse_iso(slot.get("outside_since"))
                    elapsed = (
                        (event_dt - outside_since_dt).total_seconds()
                        if event_dt and outside_since_dt else 0.0
                    )

                    if elapsed >= UNAUTH_DWELL_SECONDS and not slot["violated"]:
                        slot["violated"] = True
                        evt = build_event(
                            event_type="unauthorized_parking",
                            camera_id=camera_id,
                            timestamp=event_ts,
                            track_id=track_id,
                            metadata={
                                "bbox": car.get("bbox"),
                                "confidence": car.get("confidence"),
                                "reason": "car outside all ROIs",
                                "description": _DESC_UNAUTHORIZED,
                                "dwell_seconds": round(elapsed, 1),
                                "verdict_source": verdict_source,
                                "arbiter_verdict": arbiter_verdict,
                                "arbiter_confidence": arbiter_confidence,
                            },
                        )
                        violations.append(evt)
                        flagged.append(car)
                        publish_sync("violation_events", evt)
                        logger.warning(
                            "[COMPLIANCE] Unauthorized parking: camera=%s track=%s dwell=%.1fs source=%s",
                            camera_id, track_id, elapsed, verdict_source,
                        )

                        # Same threshold also gates the parking_intime so
                        # session lifecycle stays aligned with the violation.
                        if not slot["occupied"]:
                            slot["occupied"] = True
                            slot["intime"] = event_ts
                            intime_evt = build_event(
                                event_type="parking_intime",
                                camera_id=camera_id,
                                timestamp=slot["intime"],
                                track_id=track_id,
                                metadata={"source": "unauthorized_parking"},
                            )
                            events.append(intime_evt)
                            publish_sync("parking_events", intime_evt)
                            logger.info(
                                "[COMPLIANCE] Unauthorized car intime: camera=%s track=%s",
                                camera_id, track_id,
                            )

                elif verdict == "double_slot":
                    # ── Wrong / Double-slot parking (debounced) ───────────────
                    # When the LLM is the source we still apply the ENTRY_FRAMES
                    # debounce. The LLM is per-frame and can flicker on edge
                    # geometry; three consecutive agreeing frames keeps single
                    # bad detections from firing a false wrong_parking.
                    slot["outside_since"] = None
                    slot["wrong_buf"] += 1

                    if slot["wrong_buf"] >= ENTRY_FRAMES and not slot["wrong_fired"]:
                        slot["wrong_fired"] = True
                        evt = build_event(
                            event_type="wrong_parking",
                            camera_id=camera_id,
                            timestamp=event_ts,
                            track_id=track_id,
                            metadata={
                                "bbox": car.get("bbox"),
                                "confidence": car.get("confidence"),
                                "overlapping_rois": all_matched_rois,
                                "reason": "car occupies multiple ROI slots (double parking)",
                                "description": _DESC_DOUBLE_SLOT,
                                "verdict_source": verdict_source,
                                "arbiter_verdict": arbiter_verdict,
                                "arbiter_confidence": arbiter_confidence,
                            },
                        )
                        violations.append(evt)
                        flagged.append(car)
                        publish_sync("violation_events", evt)
                        logger.warning(
                            "[COMPLIANCE] Wrong parking: camera=%s track=%s rois=%s source=%s",
                            camera_id, track_id, all_matched_rois, verdict_source,
                        )

                else:
                    # "proper" or "across_line" — not a location violation.
                    # Reset the violation buffers so a future drift outside the
                    # ROI starts the debounce fresh.
                    slot["outside_since"] = None
                    slot["wrong_buf"] = 0

                # ── Non-EV parking (LLM-only) ─────────────────────────────────
                # A non-EV vehicle occupying any configured EV slot is a
                # violation. We require:
                #   - LLM is_ev verdict == "non_ev" (we never default to non_ev
                #     when uncertain; "unknown" never fires)
                #   - the vehicle overlaps any ROI (centroid in OR bbox-overlap
                #     above WRONG_PARKING_OVERLAP) — i.e. it is occupying a
                #     slot reserved for EVs. A non-EV parked outside every ROI
                #     is just regular unauthorized parking.
                # Fires once per track via slot["non_ev_fired"].
                if (
                    llm_is_ev == "non_ev"
                    and all_matched_rois
                    and not slot.get("non_ev_fired")
                ):
                    slot["non_ev_fired"] = True
                    evt = build_event(
                        event_type="non_ev_parking",
                        camera_id=camera_id,
                        timestamp=event_ts,
                        track_id=track_id,
                        metadata={
                            "bbox": car.get("bbox"),
                            "confidence": car.get("confidence"),
                            "occupied_rois": all_matched_rois,
                            "car_model": (llm_vehicle or {}).get("car_model"),
                            "car_number": (llm_vehicle or {}).get("car_number"),
                            "reason": "non-EV vehicle occupying EV charging slot",
                            "description": _DESC_NON_EV,
                            "occupied_rois_summary": ", ".join(all_matched_rois),
                        },
                    )
                    violations.append(evt)
                    if car not in flagged:
                        flagged.append(car)
                    publish_sync("violation_events", evt)
                    logger.warning(
                        "[COMPLIANCE] Non-EV parking: camera=%s track=%s rois=%s",
                        camera_id, track_id, all_matched_rois,
                    )

        # ── Check exit for unauthorized cars no longer visible ────────────────
        for track_id, slot in list(state.items()):
            if track_id in active_unauthorized:
                continue
            if slot.get("occupied"):
                slot["exit_buf"] = slot.get("exit_buf", 0) + 1
                if slot["exit_buf"] >= EXIT_FRAMES:
                    outtime = event_ts
                    outtime_evt = build_event(
                        event_type="parking_outtime",
                        camera_id=camera_id,
                        timestamp=outtime,
                        track_id=track_id,
                        metadata={
                            "intime": slot.get("intime"),
                            "outtime": outtime,
                            "source": "unauthorized_parking",
                        },
                    )
                    events.append(outtime_evt)
                    publish_sync("parking_events", outtime_evt)
                    logger.info(
                        "[COMPLIANCE] Unauthorized car outtime: camera=%s track=%s",
                        camera_id, track_id,
                    )
                    # Reset slot so a new car in the same track_id starts fresh
                    del state[track_id]
            else:
                # Track was being debounced for unauthorized/wrong parking but
                # never crossed ENTRY_FRAMES (i.e. flicker / brief misclass).
                # Drop it after EXIT_FRAMES of absence so Redis doesn't grow.
                slot["exit_buf"] = slot.get("exit_buf", 0) + 1
                if slot["exit_buf"] >= EXIT_FRAMES:
                    del state[track_id]

        # --- Save state back to Redis ---
        set_state(redis_key, state)

        triggered = len(violations) > 0 or len(events) > 0
        print(f"[COMPLIANCE] result: triggered={triggered} | violations={[v['event_type'] for v in violations]} | events={[e['event_type'] for e in events]}")
        return {
            "triggered": triggered,
            "matched_objects": flagged,
            "violations": violations,
            "events": events,
        }
