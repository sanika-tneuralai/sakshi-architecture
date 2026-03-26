"""
Parking Detection Rule
======================
Detects car entry and exit events per parking ROI.

Features:
- Centroid tracking (stable track_ids across frames)
- Entry / exit debounce (N consecutive frames required to confirm)
- Publishes parking_intime / parking_outtime events
- Publishes violation when multiple cars occupy one ROI

ROI polygons are NOT hardcoded here.
The orchestrator injects them via detection_output["rois"]:
    {
        "ROI_1": [[x, y], ...],
        "ROI_2": [[x, y], ...]
    }
"""
import logging
import threading
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import is_bbox_in_roi, which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.domain.vehicles.tracking import CentroidTracker
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3  # consecutive frames car must be present to confirm entry
EXIT_FRAMES = 3   # consecutive frames car must be absent to confirm exit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ParkingDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_detection"

    # Class-level state — persists across evaluate() calls (one process lifetime)
    # Keyed by camera_id
    _trackers: Dict[str, CentroidTracker] = {}
    _roi_state: Dict[str, Dict[str, dict]] = {}
    _entry_buf: Dict[str, Dict[str, Dict[str, int]]] = {}
    _exit_buf: Dict[str, Dict[str, int]] = {}
    _lock = threading.Lock()

    @classmethod
    def _init_camera(cls, camera_id: str, roi_names: List[str]) -> None:
        if camera_id not in cls._trackers:
            cls._trackers[camera_id] = CentroidTracker(max_disappeared=15, max_distance=100)
        if camera_id not in cls._roi_state:
            cls._roi_state[camera_id] = {}
        for name in roi_names:
            cls._roi_state[camera_id].setdefault(name, {
                "occupied": False,
                "track_id": None,
                "intime": None,
            })
        cls._entry_buf.setdefault(camera_id, {name: {} for name in roi_names})
        cls._exit_buf.setdefault(camera_id, {name: 0 for name in roi_names})
        for name in roi_names:
            cls._entry_buf[camera_id].setdefault(name, {})
            cls._exit_buf[camera_id].setdefault(name, 0)

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois")
        if not rois:
            logger.error("[PARKING] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "events": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]

        with self._lock:
            self._init_camera(camera_id, list(rois.keys()))
            tracker = self._trackers[camera_id]

        tracked_cars = tracker.update(cars)

        # Map each ROI → list of track_ids currently inside it
        roi_occupants: Dict[str, List[str]] = {name: [] for name in rois}
        for car in tracked_cars:
            for roi_name in which_rois(car["bbox"], rois):
                roi_occupants[roi_name].append(car["track_id"])

        events: List[dict] = []
        triggered = False

        with self._lock:
            roi_state = self._roi_state[camera_id]
            entry_buf = self._entry_buf[camera_id]
            exit_buf = self._exit_buf[camera_id]

            for roi_name, occupant_ids in roi_occupants.items():
                state = roi_state[roi_name]

                if occupant_ids:
                    triggered = True

                    # Multiple cars in same ROI → violation
                    if len(occupant_ids) > 1:
                        evt = build_event(
                            event_type="multiple_cars_in_roi",
                            camera_id=camera_id,
                            timestamp=_now(),
                            track_id=",".join(occupant_ids),
                            metadata={"roi": roi_name, "count": len(occupant_ids)},
                        )
                        events.append(evt)
                        publish_sync("violation_events", evt)

                    primary_id = occupant_ids[0]
                    exit_buf[roi_name] = 0
                    entry_buf[roi_name][primary_id] = entry_buf[roi_name].get(primary_id, 0) + 1

                    if (
                        not state["occupied"]
                        and entry_buf[roi_name][primary_id] >= ENTRY_FRAMES
                    ):
                        state["occupied"] = True
                        state["track_id"] = primary_id
                        state["intime"] = _now()
                        entry_buf[roi_name].clear()

                        evt = build_event(
                            event_type="parking_intime",
                            camera_id=camera_id,
                            timestamp=state["intime"],
                            track_id=primary_id,
                            metadata={"roi": roi_name},
                        )
                        events.append(evt)
                        publish_sync("parking_events", evt)
                        logger.info(
                            "[PARKING] Intime confirmed: camera=%s roi=%s track=%s",
                            camera_id, roi_name, primary_id,
                        )
                else:
                    # No car in ROI this frame
                    entry_buf[roi_name].clear()

                    if state["occupied"]:
                        exit_buf[roi_name] += 1
                        if exit_buf[roi_name] >= EXIT_FRAMES:
                            outtime = _now()
                            evt = build_event(
                                event_type="parking_outtime",
                                camera_id=camera_id,
                                timestamp=outtime,
                                track_id=state["track_id"] or "unknown",
                                metadata={
                                    "roi": roi_name,
                                    "intime": state["intime"],
                                    "outtime": outtime,
                                },
                            )
                            events.append(evt)
                            publish_sync("parking_events", evt)
                            logger.info(
                                "[PARKING] Outtime confirmed: camera=%s roi=%s track=%s",
                                camera_id, roi_name, state["track_id"],
                            )
                            state["occupied"] = False
                            state["track_id"] = None
                            state["intime"] = None
                            exit_buf[roi_name] = 0

        return {"triggered": triggered, "matched_objects": tracked_cars, "events": events}
