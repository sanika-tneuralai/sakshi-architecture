"""
Parking Compliance Rule
========================
Checks for two compliance violations per car:

  A. Unauthorized parking — car centroid is outside ALL defined ROIs
  B. Wrong parking        — car centroid is inside MORE THAN ONE ROI simultaneously

For unauthorized parking, this rule also fires parking session events so the
full session lifecycle (in_time → out_time) is captured even when the car
never enters a legitimate ROI:

  - parking_intime  fired once when an unauthorized car is confirmed present
                    for ENTRY_FRAMES consecutive frames
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
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3   # consecutive frames car must be outside ROI to confirm unauthorized entry
EXIT_FRAMES  = 3   # consecutive frames car must be absent to confirm exit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ParkingComplianceRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_compliance"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois = detection_output.get("rois")
        if not rois:
            logger.error("[COMPLIANCE] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "violations": [], "events": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]
        print(f"[COMPLIANCE] camera={camera_id} | rois={list(rois.keys())} | cars_detected={len(cars)}")

        # --- Load Redis state for session tracking of unauthorized cars ---
        redis_key = f"compliance:{camera_id}"
        state = get_state(redis_key)
        # state structure per track_id:
        # {
        #   "<track_id>": {
        #     "entry_buf": int,   consecutive frames seen outside ROI
        #     "exit_buf":  int,   consecutive frames absent
        #     "occupied":  bool,  intime has been fired
        #     "intime":    str,   ISO timestamp of intime event
        #   }
        # }

        violations: List[dict] = []
        events: List[dict] = []
        flagged: List[dict] = []

        # Track which track_ids are active this frame (outside all ROIs)
        active_unauthorized: set = set()

        if not cars:
            print(f"[COMPLIANCE] no cars — skipping violation check")
        else:
            for car in cars:
                track_id = car.get("track_id", "unknown")
                matched_rois = which_rois(car["bbox"], rois)
                print(f"[COMPLIANCE] car track={track_id} | matched_rois={matched_rois}")

                if len(matched_rois) == 0:
                    # ── Unauthorized parking ──────────────────────────────────
                    active_unauthorized.add(track_id)

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

                    # Debounced intime for session tracking
                    slot = state.setdefault(track_id, {"entry_buf": 0, "exit_buf": 0, "occupied": False, "intime": None})
                    slot["exit_buf"] = 0
                    slot["entry_buf"] += 1

                    if not slot["occupied"] and slot["entry_buf"] >= ENTRY_FRAMES:
                        slot["occupied"] = True
                        slot["intime"] = _now()
                        slot["entry_buf"] = 0
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

                elif len(matched_rois) > 1:
                    # ── Wrong parking ─────────────────────────────────────────
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

        # ── Check exit for unauthorized cars no longer visible ────────────────
        for track_id, slot in list(state.items()):
            if track_id not in active_unauthorized and slot.get("occupied"):
                slot["exit_buf"] = slot.get("exit_buf", 0) + 1
                if slot["exit_buf"] >= EXIT_FRAMES:
                    outtime = _now()
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
