"""
Parking Detection Rule
======================
Detects car entry and exit events per parking ROI.

Design (locked):
- One canonical CentroidTracker per camera, owned by this rule.
  gun_detection and vehicle_extraction read tracked cars from the slim
  payload — they do NOT instantiate their own trackers.
- Slot state (occupied, in_time, car_absent_since, gun flags) lives in
  Redis under ``slot:{camera_id}:{slot_id}`` and is shared with the other
  vehicle rules via get_slot_state / set_slot_state.
- parking_intime fires only when slot.occupied == False (prevents duplicate
  sessions on tracker reset / service restart).
- parking_outtime fires only when wall-clock seconds since car_absent_since
  >= EXIT_SECONDS AND the gun is NOT currently plugged in
  (plugin_logged=True, plugout_logged=False). Wall-clock means the rule
  produces correct outtime regardless of detection cadence — a slow
  pipeline still fires outtime within ~EXIT_SECONDS of the car leaving.
- On parking_outtime: slot state is fully reset via reset_slot_state().

ROI polygons are injected via detection_output["rois"]:
    {"ROI_1": [[x,y], ...], "ROI_2": [[x,y], ...]}

Tracker state is persisted in Redis under ``tracker:{camera_id}``.
Entry debounce buffers are persisted under ``parking:{camera_id}``.
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.domain.vehicles.tracking import CentroidTracker
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state, get_slot_state, set_slot_state, reset_slot_state

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3    # consecutive frames car must be present to confirm entry

# Wall-clock thresholds for exit. Frame-count gating produced bad outtimes
# whenever the detection pipeline ran slower than 1 fps (e.g. RTSP stalls or
# orchestrator backpressure pushing inter-frame gaps to minutes), because
# car_absent_frames advanced at most once per call.
EXIT_SECONDS = 60   # car must be absent this many wall-clock seconds to confirm exit

# Hard ceiling on how long the "gun still plugged in" exit-guard can block
# outtime. Protects against a deadlock where a stale/hallucinated gun detection
# keeps gun_active=True forever while the car is long gone. After this many
# car-absent seconds we force-fire both parking_outtime and gun_plugout.
FORCE_EXIT_SECONDS = 180  # 3× EXIT_SECONDS


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> str:
    return _now_dt().isoformat()


def _absent_seconds(absent_since_iso: str | None, now: datetime) -> float:
    """Wall-clock seconds since the slot first became absent. 0 if never set."""
    if not absent_since_iso:
        return 0.0
    try:
        since = datetime.fromisoformat(absent_since_iso)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (now - since).total_seconds())


def _init_debounce(state: dict, roi_names: List[str]) -> dict:
    """Ensure entry debounce buckets exist for every ROI without overwriting."""
    state.setdefault("entry_buf", {})
    for name in roi_names:
        state["entry_buf"].setdefault(name, {})
    return state


class ParkingDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "parking_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        task_id   = detection_output.get("_task_id")
        rois      = detection_output.get("rois")
        if not rois:
            logger.error("[PARKING] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "events": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]
        print(f"[PARKING] camera={camera_id} | rois={list(rois.keys())} | cars_detected={len(cars)}")

        # ── Canonical tracker (shared across all vehicle usecases) ────────
        tracker_key = f"tracker:{camera_id}"
        tracker_state = get_state(tracker_key)
        tracker = CentroidTracker(max_disappeared=20, max_distance=100)
        if tracker_state:
            tracker.from_dict(tracker_state)
        tracked_cars = tracker.update(cars)

        # ── Entry debounce state ──────────────────────────────────────────
        debounce_key = f"parking:{camera_id}"
        debounce     = get_state(debounce_key)
        debounce     = _init_debounce(debounce, list(rois.keys()))
        entry_buf    = debounce["entry_buf"]

        # ── Map each ROI → occupant track_ids this frame ──────────────────
        roi_occupants: Dict[str, List[str]] = {name: [] for name in rois}
        for car in tracked_cars:
            for roi_name in which_rois(car["bbox"], rois):
                roi_occupants[roi_name].append(car["track_id"])

        # Build a quick lookup: track_id → car dict (for track_id enrichment)
        car_by_tid = {c["track_id"]: c for c in tracked_cars}

        events: List[dict]  = []
        triggered           = False

        print(f"[PARKING] roi_occupants={roi_occupants}")

        for roi_name, occupant_ids in roi_occupants.items():

            # ── Load per-slot state ───────────────────────────────────────
            slot = get_slot_state(camera_id, roi_name)

            # Advance the frame counter every iteration — monotonic, survives restarts
            slot["frame_counter"] += 1

            if occupant_ids:
                triggered = True
                # Car is present this frame — clear absent timestamp
                slot["car_absent_since"] = None

                # Multiple cars in the same ROI — violation event (unchanged)
                if len(occupant_ids) > 1:
                    evt = build_event(
                        event_type="multiple_cars_in_roi",
                        camera_id=camera_id,
                        timestamp=_now(),
                        track_id=",".join(occupant_ids),
                        metadata={"roi": roi_name, "slot_id": roi_name, "count": len(occupant_ids)},
                    )
                    events.append(evt)
                    publish_sync("violation_events", evt, task_id=task_id)

                # Entry debounce — fire parking_intime only when slot was idle
                for tid in occupant_ids:
                    entry_buf[roi_name][tid] = entry_buf[roi_name].get(tid, 0) + 1

                    if not slot["occupied"] and entry_buf[roi_name][tid] >= ENTRY_FRAMES:
                        intime = _now()
                        # Mark slot occupied before publishing so re-entrant calls can't
                        # double-fire even within the same frame batch
                        slot["occupied"]         = True
                        slot["in_time"]          = intime
                        slot["track_id"]         = tid
                        slot["car_absent_since"] = None
                        entry_buf[roi_name].pop(tid, None)

                        evt = build_event(
                            event_type="parking_intime",
                            camera_id=camera_id,
                            timestamp=intime,
                            track_id=tid,
                            metadata={"roi": roi_name, "slot_id": roi_name},
                        )
                        events.append(evt)
                        publish_sync("parking_events", evt, task_id=task_id)
                        logger.info(
                            "[PARKING] Intime confirmed: camera=%s roi=%s track=%s",
                            camera_id, roi_name, tid,
                        )
                    elif slot["occupied"]:
                        prev_tid = slot.get("track_id")
                        if prev_tid and tid != prev_tid and entry_buf[roi_name][tid] >= ENTRY_FRAMES:
                            # A different track_id has been consistently present for
                            # ENTRY_FRAMES — the previous car left without a clean exit
                            # (e.g. compliance violation, tracker re-ID after service
                            # restart). Fire outtime for the old car, then intime for
                            # the new one so the session boundary is correct in the DB.
                            swap_ts = _now()

                            outtime_evt = build_event(
                                event_type="parking_outtime",
                                camera_id=camera_id,
                                timestamp=swap_ts,
                                track_id=prev_tid,
                                metadata={
                                    "roi":     roi_name,
                                    "slot_id": roi_name,
                                    "intime":  slot["in_time"],
                                    "outtime": swap_ts,
                                    "reason":  "car swap — new car confirmed in slot",
                                },
                            )
                            events.append(outtime_evt)
                            publish_sync("parking_events", outtime_evt, task_id=task_id)
                            logger.warning(
                                "[PARKING] Car swap detected: camera=%s roi=%s old=%s new=%s",
                                camera_id, roi_name, prev_tid, tid,
                            )

                            # Reset slot state then record the new car's intime
                            reset_slot_state(camera_id, roi_name)
                            slot = get_slot_state(camera_id, roi_name)

                            slot["occupied"]         = True
                            slot["in_time"]          = swap_ts
                            slot["track_id"]         = tid
                            slot["car_absent_since"] = None
                            entry_buf[roi_name].pop(tid, None)

                            intime_evt = build_event(
                                event_type="parking_intime",
                                camera_id=camera_id,
                                timestamp=swap_ts,
                                track_id=tid,
                                metadata={"roi": roi_name, "slot_id": roi_name},
                            )
                            events.append(intime_evt)
                            publish_sync("parking_events", intime_evt, task_id=task_id)
                            logger.info(
                                "[PARKING] Intime for swapped car: camera=%s roi=%s track=%s",
                                camera_id, roi_name, tid,
                            )
                        else:
                            # Same car still in slot — refresh track_id (informational)
                            slot["track_id"] = tid
                            entry_buf[roi_name].pop(tid, None)

            else:
                # No car in this ROI this frame
                entry_buf[roi_name].clear()

                if slot["occupied"]:
                    now_dt = _now_dt()
                    if not slot.get("car_absent_since"):
                        slot["car_absent_since"] = now_dt.isoformat()
                    absent_secs = _absent_seconds(slot.get("car_absent_since"), now_dt)

                    # ── Outtime guard: blocked while gun is plugged in ────
                    gun_active = slot["plugin_logged"] and not slot["plugout_logged"]

                    # Force-close safety net: if the car has been gone far
                    # longer than a normal exit window, the gun_plugout must
                    # have been missed by detection. Synthesize it here and
                    # let outtime fire so the slot is not wedged forever.
                    force_close = gun_active and absent_secs >= FORCE_EXIT_SECONDS
                    if force_close:
                        plugout_ts  = _now()
                        plugout_evt = build_event(
                            event_type="gun_plugout",
                            camera_id=camera_id,
                            timestamp=plugout_ts,
                            track_id=slot["track_id"] or "",
                            metadata={
                                "gun_name":     slot.get("gun_name"),
                                "roi":          roi_name,
                                "slot_id":      roi_name,
                                "plugin_time":  slot.get("plug_time"),
                                "plugout_time": plugout_ts,
                                "reason":       "forced — car absent > FORCE_EXIT_SECONDS",
                            },
                        )
                        events.append(plugout_evt)
                        publish_sync("gun_events", plugout_evt, task_id=task_id)
                        logger.warning(
                            "[PARKING] Forced gun_plugout — car absent %.1fs: camera=%s roi=%s",
                            absent_secs, camera_id, roi_name,
                        )
                        # Treat the gun as no longer active for the outtime check below
                        gun_active = False

                    if absent_secs >= EXIT_SECONDS and not gun_active:
                        outtime = _now()
                        evt = build_event(
                            event_type="parking_outtime",
                            camera_id=camera_id,
                            timestamp=outtime,
                            track_id=slot["track_id"] or "",
                            metadata={
                                "roi":     roi_name,
                                "slot_id": roi_name,
                                "intime":  slot["in_time"],
                                "outtime": outtime,
                            },
                        )
                        events.append(evt)
                        publish_sync("parking_events", evt)
                        logger.info(
                            "[PARKING] Outtime confirmed: camera=%s roi=%s track=%s",
                            camera_id, roi_name, slot["track_id"],
                        )
                        # Reset slot — preserves frame_counter
                        reset_slot_state(camera_id, roi_name)
                        # Skip set_slot_state below; reset already persisted
                        continue

            # Persist updated slot state for this ROI
            set_slot_state(camera_id, roi_name, slot)

        # ── Persist tracker + debounce state ─────────────────────────────
        set_state(tracker_key, tracker.to_dict())
        debounce["entry_buf"] = entry_buf
        set_state(debounce_key, debounce)

        print(f"[PARKING] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": tracked_cars, "events": events}
