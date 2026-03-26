"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Tracks gun usage events. Prevents duplicate logs per gun position.

Detection classes expected from YOLO:
    "gun_plugged_in"   — gun connected to charging station / holster
    "gun_plugged_out"  — gun removed

State is keyed by a stable spatial hash (camera + grid-snapped bbox position)
because guns are relatively stationary objects.
"""
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule

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


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    # class-level state: key → {plugin_time, plugout_time, logged_plugin, logged_plugout}
    _gun_state: Dict[str, dict] = {}
    _lock = threading.Lock()

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")

        gun_dets = [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") in GUN_CLASSES
        ]
        if not gun_dets:
            return {"triggered": False, "matched_objects": [], "events": []}

        events: List[dict] = []

        with self._lock:
            for det in gun_dets:
                class_name = det["class_name"]
                key = _gun_key(camera_id, det)
                state = self._gun_state.setdefault(key, {
                    "plugin_time": None,
                    "plugout_time": None,
                    "logged_plugin": False,
                    "logged_plugout": False,
                })

                if class_name == "gun_plugged_in" and not state["logged_plugin"]:
                    state["plugin_time"] = _now()
                    state["logged_plugin"] = True
                    evt = build_event(
                        event_type="gun_plugin",
                        camera_id=camera_id,
                        timestamp=state["plugin_time"],
                        track_id=key,
                        metadata={
                            "bbox": det.get("bbox"),
                            "confidence": det.get("confidence"),
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt)
                    logger.info("[GUN] Plugin logged: camera=%s key=%s", camera_id, key)

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
                            "bbox": det.get("bbox"),
                            "confidence": det.get("confidence"),
                            "plugin_time": state["plugin_time"],
                            "plugout_time": state["plugout_time"],
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt)
                    logger.info("[GUN] Plugout logged: camera=%s key=%s", camera_id, key)

        return {"triggered": True, "matched_objects": gun_dets, "events": events}
