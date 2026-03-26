"""
Safety Monitoring Rule
=======================
Detects fire and smoke — triggers an alert event immediately on detection.
No debounce: safety hazards are always urgent.
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from usecase.domain.saftey.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

SAFETY_CLASSES = {"smoke", "fire"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SafetyMonitoringRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "safety_monitoring"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")

        hazards = [
            d for d in detection_output.get("detections", [])
            if d.get("class_name") in SAFETY_CLASSES
        ]
        if not hazards:
            return {"triggered": False, "matched_objects": []}

        events: List[dict] = []
        for det in hazards:
            evt = build_event(
                event_type="safety_alert",
                camera_id=camera_id,
                timestamp=_now(),
                track_id=det.get("track_id", "unknown"),
                metadata={
                    "hazard_type": det["class_name"],
                    "confidence": det.get("confidence"),
                    "bbox": det.get("bbox"),
                },
            )
            events.append(evt)
            publish_sync("safety_events", evt)
            logger.warning(
                "[SAFETY] ALERT: camera=%s hazard=%s confidence=%.2f",
                camera_id, det["class_name"], det.get("confidence", 0.0),
            )

        return {"triggered": True, "matched_objects": hazards, "events": events}
