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

State is persisted in Redis so that Celery workers (separate processes)
share state across frames. The Redis key per camera is:
    ``parking:<camera_id>``
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import is_bbox_in_roi, which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.domain.vehicles.tracking import CentroidTracker
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3  # consecutive frames car must be present to confirm entry
EXIT_FRAMES = 3   # consecutive frames car must be absent to confirm exit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_state(state: dict, roi_names: List[str]) -> dict:
    """
    Ensure all required keys exist in the loaded state dict.
    Adds missing ROI slots without overwriting existing data.

    roi_state[roi_name] = {track_id: {"intime": str, "confirmed": bool}}
    entry_buf[roi_name] = {track_id: frame_count}
    exit_buf[roi_name]  = {track_id: frame_count}
    """
    state.setdefault("tracker", {})
    state.setdefault("roi_state", {})
    state.setdefault("entry_buf", {})
    state.setdefault("exit_buf", {})

    for name in roi_names:
        state["roi_state"].setdefault(name, {})
        state["entry_buf"].setdefault(name, {})
        state["exit_buf"].setdefault(name, {})

    return state


class ParkingDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois")
        if not rois:
            logger.error("[PARKING] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "events": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]
        print(f"[PARKING] camera={camera_id} | rois={list(rois.keys())} | cars_detected={len(cars)}")

        # --- Load state from Redis ---
        redis_key = f"parking:{camera_id}"
        state = get_state(redis_key)
        state = _init_state(state, list(rois.keys()))

        # --- Reconstruct CentroidTracker from saved state ---
        tracker = CentroidTracker(max_disappeared=15, max_distance=100)
        if state["tracker"]:
            tracker.from_dict(state["tracker"])

        # --- Run existing tracking logic (unchanged) ---
        tracked_cars = tracker.update(cars)

        # Map each ROI → list of track_ids currently inside it
        roi_occupants: Dict[str, List[str]] = {name: [] for name in rois}
        for car in tracked_cars:
            for roi_name in which_rois(car["bbox"], rois):
                roi_occupants[roi_name].append(car["track_id"])

        events: List[dict] = []
        triggered = False

        roi_state = state["roi_state"]
        entry_buf = state["entry_buf"]
        exit_buf = state["exit_buf"]

        print(f"[PARKING] roi_occupants={roi_occupants}")
        for roi_name, occupant_ids in roi_occupants.items():
            car_states = roi_state[roi_name]  # {track_id: {"intime": str, "confirmed": bool}}

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

                # Process every car in this ROI independently
                for tid in occupant_ids:
                    exit_buf[roi_name].pop(tid, None)  # reset exit counter since car is present
                    entry_buf[roi_name][tid] = entry_buf[roi_name].get(tid, 0) + 1

                    if tid not in car_states and entry_buf[roi_name][tid] >= ENTRY_FRAMES:
                        intime = _now()
                        car_states[tid] = {"intime": intime, "confirmed": True}
                        entry_buf[roi_name].pop(tid, None)

                        evt = build_event(
                            event_type="parking_intime",
                            camera_id=camera_id,
                            timestamp=intime,
                            track_id=tid,
                            metadata={"roi": roi_name},
                        )
                        events.append(evt)
                        publish_sync("parking_events", evt)
                        logger.info(
                            "[PARKING] Intime confirmed: camera=%s roi=%s track=%s",
                            camera_id, roi_name, tid,
                        )

                # Handle cars that were confirmed but are no longer in this ROI
                departed_ids = [tid for tid in car_states if tid not in occupant_ids]
            else:
                # No car in ROI this frame — all confirmed cars are potentially departing
                entry_buf[roi_name].clear()
                departed_ids = list(car_states.keys())

            for tid in departed_ids:
                exit_buf[roi_name][tid] = exit_buf[roi_name].get(tid, 0) + 1
                if exit_buf[roi_name][tid] >= EXIT_FRAMES:
                    outtime = _now()
                    intime = car_states[tid].get("intime")
                    evt = build_event(
                        event_type="parking_outtime",
                        camera_id=camera_id,
                        timestamp=outtime,
                        track_id=tid,
                        metadata={
                            "roi": roi_name,
                            "intime": intime,
                            "outtime": outtime,
                        },
                    )
                    events.append(evt)
                    publish_sync("parking_events", evt)
                    logger.info(
                        "[PARKING] Outtime confirmed: camera=%s roi=%s track=%s",
                        camera_id, roi_name, tid,
                    )
                    del car_states[tid]
                    exit_buf[roi_name].pop(tid, None)

        # --- Save updated state back to Redis ---
        state["tracker"] = tracker.to_dict()
        state["roi_state"] = roi_state
        state["entry_buf"] = entry_buf
        state["exit_buf"] = exit_buf
        set_state(redis_key, state)

        print(f"[PARKING] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": tracked_cars, "events": events}
