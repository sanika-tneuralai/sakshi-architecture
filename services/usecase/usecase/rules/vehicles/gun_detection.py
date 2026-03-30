"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Tracks gun plug-in / plug-out events, one per car per ROI.

Multi-car / multi-gun support:
    Each ROI can hold one car and one gun simultaneously.
    Gun ↔ car association is done by ROI membership:
        - Gun in ROI_1 belongs to the car currently in ROI_1.
        - Gun in ROI_2 belongs to the car currently in ROI_2.
    State is keyed by (roi_name, car_track_id) so two cars in two ROIs
    are tracked independently.

Detection classes handled:
    "gun"              — generic gun class from YOLO model
    "gun_plugged_in"   — explicit plug-in class (bypasses debounce)
    "gun_plugged_out"  — explicit plug-out class (bypasses debounce)

Plugin logic (debounced for generic "gun" class):
    Gun detected in the same ROI as a confirmed car for GUN_PLUGIN_FRAMES
    consecutive frames → gun_plugin event (once per car per ROI).

Plugout logic (debounced):
    Gun absent for GUN_PLUGOUT_FRAMES consecutive frames while the same
    car is still in its ROI + plugin was already logged → gun_plugout event.

Guards:
    - No car in a ROI → no events fire for that ROI.
    - Car leaves ROI → its state slot is discarded; next car starts fresh.
    - Fresh car arrives, no gun ever seen → gun_present_buf stays 0,
      plugin never fires, so plugout can never fire either.

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

GUN_CLASSES        = {"gun_plugged_in", "gun_plugged_out", "gun"}
CAR_CLASS          = "car"
GUN_PLUGIN_FRAMES  = 3   # consecutive frames gun must be present  → confirm plugin
GUN_PLUGOUT_FRAMES = 4   # consecutive frames gun must be absent   → confirm plugout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slot_key(roi_name: str, track_id: str) -> str:
    """Unique state key for one (ROI, car) pair."""
    return f"{roi_name}::{track_id}"


def _gun_name_for_roi(roi_name: str) -> str:
    """Derive a human-readable gun name from the ROI name. ROI_1 → Gun 1."""
    parts = roi_name.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else roi_name
    return f"Gun {suffix}"


def _init_slot() -> dict:
    """Fresh state for a (ROI, car) pair."""
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


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        rois      = detection_output.get("rois", {})
        all_dets  = detection_output.get("detections", [])

        # ── Step 1: load Redis state ──────────────────────────────────────
        redis_key = f"gun:{camera_id}"
        state     = get_state(redis_key)

        print(f"[GUN] camera={camera_id} | rois={list(rois.keys())} | total_dets={len(all_dets)}")
        # ── Step 2: track cars across frames ─────────────────────────────
        car_dets = [d for d in all_dets if d.get("class_name") == CAR_CLASS]
        tracker  = CentroidTracker(max_disappeared=15, max_distance=100)
        if state.get("tracker"):
            tracker.from_dict(state["tracker"])
        tracked_cars = tracker.update(car_dets)

        # Map ROI → car track_id for every car currently inside a ROI
        # (parking_detection already enforces one car per ROI; we take the
        #  first occupant per ROI to stay consistent with that rule)
        roi_to_car: Dict[str, str] = {}   # roi_name → track_id
        car_to_rois: Dict[str, List[str]] = {}  # track_id → [roi_names]
        if rois:
            for car in tracked_cars:
                for roi_name in which_rois(car["bbox"], rois):
                    if roi_name not in roi_to_car:
                        roi_to_car[roi_name] = car["track_id"]
                    car_to_rois.setdefault(car["track_id"], []).append(roi_name)

        print(f"[GUN] cars_tracked={len(tracked_cars)} | roi_to_car={roi_to_car}")
        # ── Step 3: map guns to ROIs ──────────────────────────────────────
        gun_dets = [d for d in all_dets if d.get("class_name") in GUN_CLASSES]

        # roi_name → best gun detection for that ROI this frame
        roi_to_gun: Dict[str, dict] = {}
        for gun in gun_dets:
            for roi_name in (which_rois(gun.get("bbox", {}), rois) if rois else []):
                # Keep highest-confidence gun per ROI
                if roi_name not in roi_to_gun or gun.get("confidence", 0) > roi_to_gun[roi_name].get("confidence", 0):
                    roi_to_gun[roi_name] = gun

        print(f"[GUN] guns_detected={len(gun_dets)} | roi_to_gun={list(roi_to_gun.keys())}")
        # ── Step 4: maintain per-(roi, car) state slots ───────────────────
        slots: Dict[str, dict] = state.get("slots", {})

        # Active slot keys this frame
        active_keys = {
            _slot_key(roi_name, track_id)
            for roi_name, track_id in roi_to_car.items()
        }

        # Remove stale slots (car left the ROI)
        for k in list(slots.keys()):
            if k not in active_keys:
                del slots[k]

        # Ensure fresh slot for every active (roi, car) pair
        for roi_name, track_id in roi_to_car.items():
            k = _slot_key(roi_name, track_id)
            if k not in slots:
                slots[k] = _init_slot()

        # ── Step 5: update debounce buffers and fire events ───────────────
        events:  List[dict] = []
        matched: List[dict] = list(gun_dets)

        for roi_name, track_id in roi_to_car.items():
            k        = _slot_key(roi_name, track_id)
            cs       = slots[k]
            gun_det: Optional[dict] = roi_to_gun.get(roi_name)
            gun_name = _gun_name_for_roi(roi_name)

            if gun_det:
                # ── Gun present in this ROI this frame ────────────────────
                cs["gun_present_buf"] += 1
                cs["gun_absent_buf"]   = 0
                cs["gun_name"]         = gun_name
                cs["gun_bbox"]         = gun_det.get("bbox")
                cs["gun_confidence"]   = gun_det.get("confidence")
                class_name             = gun_det["class_name"]

                if not cs["logged_plugin"]:
                    explicit_plugin = class_name == "gun_plugged_in"
                    debounced_plugin = (
                        class_name == "gun"
                        and cs["gun_present_buf"] >= GUN_PLUGIN_FRAMES
                    )
                    if explicit_plugin or debounced_plugin:
                        cs["plugin_time"]   = _now()
                        cs["logged_plugin"] = True
                        evt = build_event(
                            event_type="gun_plugin",
                            camera_id=camera_id,
                            timestamp=cs["plugin_time"],
                            track_id=track_id,
                            metadata={
                                "gun_name":   gun_name,
                                "roi":        roi_name,
                                "car_track":  track_id,
                                "bbox":       cs["gun_bbox"],
                                "confidence": cs["gun_confidence"],
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt)
                        logger.info(
                            "[GUN] Plugin logged: camera=%s roi=%s car=%s gun=%s",
                            camera_id, roi_name, track_id, gun_name,
                        )

                # Explicit plugout class (gun still detected but flagged as removed)
                if (
                    class_name == "gun_plugged_out"
                    and cs["logged_plugin"]
                    and not cs["logged_plugout"]
                ):
                    cs["plugout_time"]   = _now()
                    cs["logged_plugout"] = True
                    evt = build_event(
                        event_type="gun_plugout",
                        camera_id=camera_id,
                        timestamp=cs["plugout_time"],
                        track_id=track_id,
                        metadata={
                            "gun_name":    gun_name,
                            "roi":         roi_name,
                            "car_track":   track_id,
                            "bbox":        cs["gun_bbox"],
                            "confidence":  cs["gun_confidence"],
                            "plugin_time": cs["plugin_time"],
                            "plugout_time": cs["plugout_time"],
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt)
                    logger.info(
                        "[GUN] Plugout (explicit) logged: camera=%s roi=%s car=%s gun=%s",
                        camera_id, roi_name, track_id, gun_name,
                    )

            else:
                # ── Gun absent from this ROI this frame ───────────────────
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
                                "gun_name":    cs["gun_name"] or gun_name,
                                "roi":         roi_name,
                                "car_track":   track_id,
                                "plugin_time": cs["plugin_time"],
                                "plugout_time": cs["plugout_time"],
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt)
                        logger.info(
                            "[GUN] Plugout (absence) logged: camera=%s roi=%s car=%s gun=%s",
                            camera_id, roi_name, track_id, gun_name,
                        )

        # ── Step 6: persist state ─────────────────────────────────────────
        state["tracker"] = tracker.to_dict()
        state["slots"]   = slots
        set_state(redis_key, state)

        triggered = bool(events) or bool(matched)
        print(f"[GUN] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": matched, "events": events}
