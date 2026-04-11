"""
Gun Plug-in / Plug-out Detection Rule
=======================================
Detects when a charging gun is plugged into a car and when it is unplugged.

Design (locked):
- NO own CentroidTracker. Tracked cars are read from
  detection_output["tracked_cars"], which parking_detection populates via the
  engine's slim-payload backfill before this rule runs.
- Slot state is read/written via get_slot_state / set_slot_state using key
  ``slot:{camera_id}:{slot_id}``.  The slot_id (ROI name) is the only key —
  no car identity string in the slot key.
- gun_absent_frames only accumulates while plugin_logged=True.
  plugin_logged is NEVER reset on a missed frame — only on confirmed plugout
  (gun_plugout event fires) or on car exit (reset_slot_state by parking_detection).
- GUN_PLUGOUT_FRAMES = 25: requires 25 consecutive gun-absent frames after
  plugin to fire gun_plugout. Large enough that a human walking past (2-5 s)
  never triggers a false plugout.
- Gun state is preserved across missed frames because slot state in Redis
  persists the plugin_logged flag — absence of a gun detection does NOT clear it.

triggered=True only when a real event fires (gun_plugin or gun_plugout).
"""
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.rules.base import BaseUsecaseRule
from workers.redis_state import get_slot_state, set_slot_state

logger = logging.getLogger(__name__)

GUN_CLASS          = "gun"
CAR_CLASS          = "car"
GUN_PLUGIN_FRAMES  = 3    # consecutive frames gun must be present to confirm plugin
GUN_PLUGOUT_FRAMES = 25   # consecutive frames gun must be absent (post-plugin) to confirm plugout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gun_name_for_roi(roi_name: str) -> str:
    parts = roi_name.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else roi_name
    return f"Gun {suffix}"


class GunDetectionRule(BaseUsecaseRule):
    USECASE_ID: ClassVar[str] = "gun_detection"

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = detection_output.get("camera_id", "unknown")
        task_id   = detection_output.get("_task_id")
        rois      = detection_output.get("rois", {})
        all_dets  = detection_output.get("detections", [])

        # Tracked cars are injected by the engine after parking_detection runs.
        # Falls back to raw car detections if not yet available (e.g. first frame).
        tracked_cars = detection_output.get("tracked_cars") or [
            d for d in all_dets if d.get("class_name") == CAR_CLASS
        ]

        print(f"[GUN] camera={camera_id} | rois={list(rois.keys())} | tracked_cars={len(tracked_cars)}")

        # ── Map cars → ROIs ───────────────────────────────────────────────
        # One car per ROI (first car wins if multiple overlap the same ROI).
        roi_to_track: Dict[str, str] = {}
        if rois:
            for car in tracked_cars:
                for roi_name in which_rois(car["bbox"], rois):
                    if roi_name not in roi_to_track:
                        roi_to_track[roi_name] = car.get("track_id", "unknown")

        # ── Map guns → ROIs ───────────────────────────────────────────────
        # Keep the highest-confidence gun per ROI.
        gun_dets = [d for d in all_dets if d.get("class_name") == GUN_CLASS]
        roi_to_gun: Dict[str, dict] = {}
        for gun in gun_dets:
            for roi_name in (which_rois(gun.get("bbox", {}), rois) if rois else []):
                if roi_name not in roi_to_gun or gun.get("confidence", 0) > roi_to_gun[roi_name].get("confidence", 0):
                    roi_to_gun[roi_name] = gun

        print(f"[GUN] roi_to_track={roi_to_track} | roi_to_gun={list(roi_to_gun.keys())}")

        events: List[dict] = []

        # ── Process each ROI independently ───────────────────────────────
        for roi_name in rois:
            slot     = get_slot_state(camera_id, roi_name)
            gun_det  = roi_to_gun.get(roi_name)
            gun_name = _gun_name_for_roi(roi_name)
            track_id = roi_to_track.get(roi_name, slot.get("track_id") or "unknown")

            # Only process gun logic when the slot is confirmed occupied.
            # If parking_detection hasn't confirmed a car here yet, skip.
            if not slot["occupied"]:
                # Nothing to do — no car confirmed in this slot
                continue

            if gun_det:
                # Gun is visible this frame
                slot["gun_present_frames"] += 1
                slot["gun_absent_frames"]   = 0  # reset absent counter while gun visible
                slot["gun_name"]            = slot["gun_name"] or gun_name

                if not slot["plugin_logged"] and slot["gun_present_frames"] >= GUN_PLUGIN_FRAMES:
                    plug_time              = _now()
                    slot["plugin_logged"]  = True
                    slot["plug_time"]      = plug_time
                    slot["gun_name"]       = gun_name
                    slot["gun_present_frames"] = 0

                    evt = build_event(
                        event_type="gun_plugin",
                        camera_id=camera_id,
                        timestamp=plug_time,
                        track_id=track_id,
                        metadata={
                            "gun_name": gun_name,
                            "roi":      roi_name,
                            "slot_id":  roi_name,
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info("[GUN] Plugin: camera=%s roi=%s gun=%s", camera_id, roi_name, gun_name)

            else:
                # Gun not visible this frame
                slot["gun_present_frames"] = 0

                # Only accumulate absence counter after a confirmed plugin.
                # plugin_logged is NEVER cleared by a missed frame — only by
                # gun_plugout firing or reset_slot_state on car exit.
                if slot["plugin_logged"] and not slot["plugout_logged"]:
                    slot["gun_absent_frames"] += 1

                    if slot["gun_absent_frames"] >= GUN_PLUGOUT_FRAMES:
                        plugout_time              = _now()
                        slot["plugout_logged"]    = True
                        slot["plug_out_time"]     = plugout_time
                        slot["gun_absent_frames"] = 0
                        # Re-arm for a potential second plug cycle
                        slot["plugin_logged"]     = False
                        slot["plugout_logged"]    = False
                        slot["gun_present_frames"] = 0

                        evt = build_event(
                            event_type="gun_plugout",
                            camera_id=camera_id,
                            timestamp=plugout_time,
                            track_id=track_id,
                            metadata={
                                "gun_name":    slot["gun_name"] or gun_name,
                                "roi":         roi_name,
                                "slot_id":     roi_name,
                                "plugin_time": slot["plug_time"],
                                "plugout_time": plugout_time,
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt, task_id=task_id)
                        logger.info("[GUN] Plugout: camera=%s roi=%s gun=%s", camera_id, roi_name, gun_name)

            set_slot_state(camera_id, roi_name, slot)

        triggered = bool(events)
        print(f"[GUN] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": list(gun_dets), "events": events}
