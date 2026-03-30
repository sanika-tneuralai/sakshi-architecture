"""
Parking Compliance Rule
========================
Checks for two compliance violations per car:

  A. Unauthorized parking — car centroid is outside ALL defined ROIs
  B. Wrong parking        — car centroid is inside MORE THAN ONE ROI simultaneously

ROI polygons are NOT hardcoded here.
The orchestrator injects them via detection_output["rois"]:
    {
        "ROI_1": [[x, y], ...],
        "ROI_2": [[x, y], ...]
    }
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ParkingComplianceRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_compliance"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois")
        if not rois:
            logger.error("[COMPLIANCE] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "violations": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]
        print(f"[COMPLIANCE] camera={camera_id} | rois={list(rois.keys())} | cars_detected={len(cars)}")
        if not cars:
            print(f"[COMPLIANCE] no cars — skipping")
            return {"triggered": False, "matched_objects": [], "violations": []}

        violations: List[dict] = []
        flagged: List[dict] = []

        for car in cars:
            track_id = car.get("track_id", "unknown")
            matched_rois = which_rois(car["bbox"], rois)
            print(f"[COMPLIANCE] car track={track_id} | matched_rois={matched_rois}")

            if len(matched_rois) == 0:
                evt = build_event(
                    event_type="unauthorized_parking",
                    camera_id=camera_id,
                    timestamp=_now(),
                    track_id=track_id,
                    metadata={
                        "bbox": car.get("bbox"),
                        "confidence": car.get("confidence"),
                        "reason": "car outside all ROIs",
                    },
                )
                violations.append(evt)
                flagged.append(car)
                publish_sync("violation_events", evt)
                logger.warning(
                    "[COMPLIANCE] Unauthorized parking: camera=%s track=%s",
                    camera_id, track_id,
                )

            elif len(matched_rois) > 1:
                evt = build_event(
                    event_type="wrong_parking",
                    camera_id=camera_id,
                    timestamp=_now(),
                    track_id=track_id,
                    metadata={
                        "bbox": car.get("bbox"),
                        "confidence": car.get("confidence"),
                        "overlapping_rois": matched_rois,
                        "reason": "car centroid inside multiple ROIs simultaneously",
                    },
                )
                violations.append(evt)
                flagged.append(car)
                publish_sync("violation_events", evt)
                logger.warning(
                    "[COMPLIANCE] Wrong parking: camera=%s track=%s rois=%s",
                    camera_id, track_id, matched_rois,
                )

        print(f"[COMPLIANCE] result: triggered={len(violations) > 0} | violations={[v['event_type'] for v in violations]}")
        return {
            "triggered": len(violations) > 0,
            "matched_objects": flagged,
            "violations": violations,
        }
