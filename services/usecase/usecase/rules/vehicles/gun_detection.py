"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Tracks gun usage events. Prevents duplicate logs per gun position.

Detection classes expected from YOLO:
    "gun_plugged_in"   — gun connected to charging station / holster
    "gun_plugged_out"  — gun removed

State is keyed by a stable spatial hash (camera + grid-snapped bbox position)
because guns are relatively stationary objects.

Per-camera state is persisted in Redis so that Celery workers (separate
processes) share state across frames. The Redis key per camera is:
    ``gun:<camera_id>``
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

GUN_CLASSES = {"gun_plugged_in", "gun_plugged_out"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gun_key(camera_id: str, det: dict) -> str:
    """
    Stable identity key for a gun detection.
    Snaps bbox to a 50-pixel grid so minor jitter doesn't create duplicate keys.
    """
    bbox = det.get("bbox", {})
    raw = f"{camera_id}_{int(bbox.get('x1', 0) // 50)}_{int(bbox.get('y1', 0) // 50)}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _resolve_gun_name(bbox: dict, rois: Dict) -> str:
    """
    Map gun bbox to a named gun based on which ROI it falls in.
    ROI_1 → Gun 1, ROI_2 → Gun 2, ... ROI_N → Gun N.
    Falls back to 'Unknown Gun' if not inside any ROI.
    """
    if not rois:
        return "Unknown Gun"
    matched = which_rois(bbox, rois)
    if not matched:
        return "Unknown Gun"
    # Use the first matched ROI — extract trailing number e.g. "ROI_1" → "Gun 1"
    roi_name = matched[0]
    parts = roi_name.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else roi_name
    return f"Gun {suffix}"


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois", {})

        gun_dets = [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") in GUN_CLASSES
        ]
        if not gun_dets:
            return {"triggered": False, "matched_objects": [], "events": []}

        # --- Load state from Redis ---
        # gun_state: { gun_key: {plugin_time, plugout_time, logged_plugin,
        #                         logged_plugout, gun_name} }
        redis_key = f"gun:{camera_id}"
        gun_state: Dict[str, dict] = get_state(redis_key)

        events: List[dict] = []

        # --- Run existing logic (unchanged) ---
        for det in gun_dets:
            class_name = det["class_name"]
            key = _gun_key(camera_id, det)
            gun_name = _resolve_gun_name(det.get("bbox", {}), rois)

            if key not in gun_state:
                gun_state[key] = {
                    "plugin_time": None,
                    "plugout_time": None,
                    "logged_plugin": False,
                    "logged_plugout": False,
                    "gun_name": gun_name,
                }
            state = gun_state[key]

            if class_name == "gun_plugged_in" and not state["logged_plugin"]:
                state["plugin_time"] = _now()
                state["logged_plugin"] = True
                evt = build_event(
                    event_type="gun_plugin",
                    camera_id=camera_id,
                    timestamp=state["plugin_time"],
                    track_id=key,
                    metadata={
                        "gun_name": gun_name,
                        "bbox": det.get("bbox"),
                        "confidence": det.get("confidence"),
                    },
                )
                events.append(evt)
                publish_sync("gun_events", evt)
                logger.info("[GUN] Plugin logged: camera=%s key=%s gun=%s", camera_id, key, gun_name)

            elif (
                class_name == "gun_plugged_out"
                and state["logged_plugin"]
                and not state["logged_plugout"]
            ):
                state["plugout_time"] = _now()
                state["logged_plugout"] = True
                evt = build_event(
                    event_type="gun_plugout",
                    camera_id=camera_id,
                    timestamp=state["plugout_time"],
                    track_id=key,
                    metadata={
                        "gun_name": gun_name,
                        "bbox": det.get("bbox"),
                        "confidence": det.get("confidence"),
                        "plugin_time": state["plugin_time"],
                        "plugout_time": state["plugout_time"],
                    },
                )
                events.append(evt)
                publish_sync("gun_events", evt)
                logger.info("[GUN] Plugout logged: camera=%s key=%s gun=%s", camera_id, key, gun_name)

        # --- Save updated state back to Redis ---
        set_state(redis_key, gun_state)

        return {"triggered": True, "matched_objects": gun_dets, "events": events}
