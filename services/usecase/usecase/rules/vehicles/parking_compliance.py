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

from shared.common.roi import which_rois, which_rois_bbox_overlap
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
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
        #     "intime":        str,  ISO timestamp of intime event
        #   }
        # }
        event_dt = _parse_iso(event_ts)

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
                # Centroid-based check: which ROI the car's centre is in
                matched_rois = which_rois(car["bbox"], rois)
                # Overlap-based check: ALWAYS run, so a car whose centroid sits
                # in ROI_1 but whose body is genuinely half in ROI_2 still
                # qualifies as wrong-parking. Threshold is high (50 %) — a car
                # parked askew but mostly in its own slot won't alert; only a
                # car genuinely straddling the boundary will.
                overlap_rois = which_rois_bbox_overlap(
                    car["bbox"], rois, overlap_threshold=WRONG_PARKING_OVERLAP,
                )
                print(
                    f"[COMPLIANCE] car track={track_id} | matched_rois={matched_rois}"
                    f" | overlap_rois={overlap_rois}"
                )

                all_matched_rois = list(dict.fromkeys(matched_rois + overlap_rois))

                slot = state.setdefault(track_id, {
                    "outside_since": None, "wrong_buf": 0, "exit_buf": 0,
                    "occupied": False, "violated": False, "wrong_fired": False,
                    "intime": None,
                })

                if len(all_matched_rois) == 0:
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
                                "description": "Car parked outside any designated charging slot.",
                                "dwell_seconds": round(elapsed, 1),
                            },
                        )
                        violations.append(evt)
                        flagged.append(car)
                        publish_sync("violation_events", evt)
                        logger.warning(
                            "[COMPLIANCE] Unauthorized parking: camera=%s track=%s dwell=%.1fs",
                            camera_id, track_id, elapsed,
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

                elif len(all_matched_rois) > 1:
                    # ── Wrong / Double-slot parking (debounced) ───────────────
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
                                "description": (
                                    f"Car straddles multiple charging slots ({', '.join(all_matched_rois)})."
                                ),
                            },
                        )
                        violations.append(evt)
                        flagged.append(car)
                        publish_sync("violation_events", evt)
                        logger.warning(
                            "[COMPLIANCE] Wrong parking: camera=%s track=%s rois=%s",
                            camera_id, track_id, all_matched_rois,
                        )

                else:
                    # Car is cleanly inside exactly one ROI — not a violation.
                    # Reset the violation buffers so a future drift outside
                    # the ROI starts the debounce fresh.
                    slot["outside_since"] = None
                    slot["wrong_buf"] = 0

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
