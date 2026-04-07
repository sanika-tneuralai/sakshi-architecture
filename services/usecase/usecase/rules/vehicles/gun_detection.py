"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Only handles the generic "gun" YOLO class (no gun_plugged_in / gun_plugged_out).

Plugin logic (debounced):
    Gun in same ROI as a confirmed car for GUN_PLUGIN_FRAMES consecutive frames
    → gun_plugin event (once per car per ROI).

Plugout logic (debounced):
    Gun absent for GUN_PLUGOUT_FRAMES consecutive frames while the same car is
    still in its ROI + plugin already logged → gun_plugout event.
    GUN_PLUGOUT_FRAMES = 7 to tolerate poor detections on small guns.

Re-arm after plugout: slot resets so a second plugin cycle fires correctly.

triggered=True only when a real event fires — not during debounce phase.

Car identity falls back to car_number / car_model from vehicle_extraction
so the same physical car is recognised even after a tracker reset.

State persisted in Redis key ``gun:<camera_id>``.
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.domain.vehicles.tracking import CentroidTracker
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_state, set_state

logger = logging.getLogger(__name__)

GUN_CLASS          = "gun"
CAR_CLASS          = "car"
GUN_PLUGIN_FRAMES  = 3
GUN_PLUGOUT_FRAMES = 7   # raised from 4 to handle poor detections on small guns


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slot_key(roi_name: str, car_identity: str) -> str:
    return f"{roi_name}::{car_identity}"


def _gun_name_for_roi(roi_name: str) -> str:
    parts = roi_name.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else roi_name
    return f"Gun {suffix}"


def _init_slot() -> dict:
    return {
        "gun_present_buf":  0,
        "gun_absent_buf":   0,
        "logged_plugin":    False,
        "logged_plugout":   False,
        "plugin_time":      None,
        "plugout_time":     None,
        "gun_name":         None,
        "gun_bbox":         None,
        "gun_confidence":   None,
    }


def _car_identity(car: dict, vehicle_cache: dict) -> str:
    """
    Stable identity for a car: car_number > car_model > track_id.
    Uses vehicle_extraction results cached in Redis state.
    """
    track_id   = car.get("track_id", "unknown")
    cached     = vehicle_cache.get(track_id, {})
    car_number = cached.get("car_number")
    car_model  = cached.get("car_model")

    if car_number and car_number not in ("unreadable", "unknown"):
        return f"plate:{car_number}"
    if car_model and car_model not in ("unknown",):
        return f"model:{car_model}"
    return track_id


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        task_id   = detection_output.get("_task_id")  # injected by Celery task for idempotency
        rois      = detection_output.get("rois", {})
        all_dets  = detection_output.get("detections", [])

        redis_key = f"gun:{camera_id}"
        state     = get_state(redis_key)

        print(f"[GUN] camera={camera_id} | rois={list(rois.keys())} | total_dets={len(all_dets)}")

        # ── Track cars ───────────────────────────────────────────────────
        car_dets = [d for d in all_dets if d.get("class_name") == CAR_CLASS]
        tracker  = CentroidTracker(max_disappeared=15, max_distance=100)
        if state.get("tracker"):
            tracker.from_dict(state["tracker"])
        tracked_cars = tracker.update(car_dets)

        vehicle_cache: dict = state.get("vehicle_cache", {})
        for vd in detection_output.get("vehicle_details", []):
            tid = vd.get("track_id")
            if tid:
                vehicle_cache[tid] = {
                    "car_number": vd.get("car_number"),
                    "car_model":  vd.get("car_model"),
                }

        roi_to_car: Dict[str, str] = {}
        roi_to_track: Dict[str, str] = {}
        if rois:
            for car in tracked_cars:
                for roi_name in which_rois(car["bbox"], rois):
                    if roi_name not in roi_to_car:
                        roi_to_car[roi_name]   = _car_identity(car, vehicle_cache)
                        roi_to_track[roi_name] = car.get("track_id", "unknown")

        print(f"[GUN] cars_tracked={len(tracked_cars)} | roi_to_car={roi_to_car}")

        # ── Map guns to ROIs ─────────────────────────────────────────────
        gun_dets = [d for d in all_dets if d.get("class_name") == GUN_CLASS]

        roi_to_gun: Dict[str, dict] = {}
        for gun in gun_dets:
            for roi_name in (which_rois(gun.get("bbox", {}), rois) if rois else []):
                if roi_name not in roi_to_gun or gun.get("confidence", 0) > roi_to_gun[roi_name].get("confidence", 0):
                    roi_to_gun[roi_name] = gun

        print(f"[GUN] guns_detected={len(gun_dets)} | roi_to_gun={list(roi_to_gun.keys())}")

        # ── Maintain per-(roi, car) state slots ──────────────────────────
        slots: Dict[str, dict] = state.get("slots", {})
        active_keys = {_slot_key(roi, ident) for roi, ident in roi_to_car.items()}

        for k in list(slots.keys()):
            if k not in active_keys:
                del slots[k]

        for roi_name, identity in roi_to_car.items():
            k = _slot_key(roi_name, identity)
            if k not in slots:
                slots[k] = _init_slot()

        # ── Debounce and fire events ─────────────────────────────────────
        events: List[dict] = []

        for roi_name, identity in roi_to_car.items():
            k        = _slot_key(roi_name, identity)
            cs       = slots[k]
            track_id = roi_to_track.get(roi_name, identity)
            gun_det: Optional[dict] = roi_to_gun.get(roi_name)
            gun_name = _gun_name_for_roi(roi_name)

            if gun_det:
                cs["gun_present_buf"] += 1
                cs["gun_absent_buf"]   = 0
                cs["gun_name"]         = gun_name
                cs["gun_bbox"]         = gun_det.get("bbox")
                cs["gun_confidence"]   = gun_det.get("confidence")

                if not cs["logged_plugin"] and cs["gun_present_buf"] >= GUN_PLUGIN_FRAMES:
                    cs["plugin_time"]   = _now()
                    cs["logged_plugin"] = True
                    evt = build_event(
                        event_type="gun_plugin",
                        camera_id=camera_id,
                        timestamp=cs["plugin_time"],
                        track_id=track_id,
                        metadata={
                            "gun_name":     gun_name,
                            "roi":          roi_name,
                            "slot_id":      roi_name,
                            "car_track":    track_id,
                            "car_identity": identity,
                            "bbox":         cs["gun_bbox"],
                            "confidence":   cs["gun_confidence"],
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info("[GUN] Plugin: camera=%s roi=%s car=%s gun=%s", camera_id, roi_name, identity, gun_name)

            else:
                cs["gun_present_buf"] = 0

                if cs["logged_plugin"] and not cs["logged_plugout"]:
                    cs["gun_absent_buf"] += 1
                    if cs["gun_absent_buf"] >= GUN_PLUGOUT_FRAMES:
                        cs["plugout_time"]   = _now()
                        cs["logged_plugout"] = True
                        evt = build_event(
                            event_type="gun_plugout",
                            camera_id=camera_id,
                            timestamp=cs["plugout_time"],
                            track_id=track_id,
                            metadata={
                                "gun_name":     cs["gun_name"] or gun_name,
                                "roi":          roi_name,
                                "slot_id":      roi_name,
                                "car_track":    track_id,
                                "car_identity": identity,
                                "plugin_time":  cs["plugin_time"],
                                "plugout_time": cs["plugout_time"],
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt, task_id=task_id)
                        logger.info("[GUN] Plugout: camera=%s roi=%s car=%s gun=%s", camera_id, roi_name, identity, gun_name)
                        slots[k] = _init_slot()  # re-arm for next cycle

        # ── Persist state ─────────────────────────────────────────────────
        state["tracker"]       = tracker.to_dict()
        state["slots"]         = slots
        state["vehicle_cache"] = vehicle_cache
        set_state(redis_key, state)

        triggered = bool(events)
        print(f"[GUN] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": list(gun_dets), "events": events}
