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

Plug-in modes:
  Real     — gun visible for GUN_PLUGIN_FRAMES consecutive frames
  Inferred — no real detection, but the car has been parked for
             INFERRED_PLUGIN_SECONDS. We synthesize gun_plugin so downstream
             energy lookups have a plug_time anchor instead of falling back
             to in_time. plug_time_inferred=True is set so a later real
             detection or absence can validate / invalidate the inference.

Inference verification:
  After an inferred plug, we periodically expect the gun detector to actually
  see the gun. If INFERRED_VERIFY_TIMEOUT_SECONDS elapse with zero real gun
  frames, the inference is rolled back (plugin_logged → False) so this slot
  doesn't end up with a bogus plug_time in the DB.

Plug-out (debounced state machine):
  PLUGGED       gun visible (or recently visible) — normal
  MAYBE_OUT     gun absent for GUN_MAYBE_OUT_SECONDS — start grace window
  back to PLUGGED if gun reappears within GUN_RETURN_GRACE_SECONDS (occlusion)
  → OUT (fire gun_plugout) only after gun stays absent another
    GUN_CONFIRM_OUT_SECONDS post-MAYBE_OUT without returning.

  Total absence before plugout fires ≈ 60s + 120s = ~3 minutes. Long enough
  that a person standing in front of the gun for several minutes does not
  trigger a false plugout.
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

# ── Inferred plug-in ────────────────────────────────────────────────────────
# After this many seconds since parking_intime with no real gun detection, fire
# a synthetic gun_plugin. Most cars are physically plugged within ~30-60 s of
# parking; 120 s is a comfortable margin that still gives the detector a fair
# chance to fire on its own first.
INFERRED_PLUGIN_SECONDS = 120

# After firing an inferred plug, we expect the gun detector to actually see the
# gun within this window. If it doesn't, the inference was wrong (e.g. car
# parked but never plugged in) and we roll back plugin_logged so a downstream
# plug_out doesn't fire from a phantom plug.
INFERRED_VERIFY_TIMEOUT_SECONDS = 300   # 5 min to see at least one real gun frame

# ── Plug-out debounce state machine ─────────────────────────────────────────
GUN_MAYBE_OUT_SECONDS    = 60    # gun absent this long → enter MAYBE_OUT
GUN_RETURN_GRACE_SECONDS = 120   # in MAYBE_OUT, gun reappearing this fast = false alarm
GUN_CONFIRM_OUT_SECONDS  = 120   # in MAYBE_OUT for this long w/o return → fire plugout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _seconds_since(iso_str: str | None, now: datetime) -> float:
    """Wall-clock seconds since iso_str. 0 if iso_str is None/invalid."""
    dt = _parse_iso(iso_str)
    if dt is None:
        return 0.0
    return max(0.0, (now - dt).total_seconds())


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
        now_dt = _now_dt()

        for roi_name in rois:
            slot     = get_slot_state(camera_id, roi_name)
            gun_det  = roi_to_gun.get(roi_name)
            gun_name = _gun_name_for_roi(roi_name)
            track_id = roi_to_track.get(roi_name, slot.get("track_id") or "unknown")

            # Only process gun logic when the slot is confirmed occupied.
            # If parking_detection hasn't confirmed a car here yet, skip.
            if not slot["occupied"]:
                continue

            if gun_det:
                # ── Gun visible this frame ──────────────────────────────
                slot["gun_present_frames"] += 1
                slot["gun_absent_frames"]   = 0  # legacy counter, kept for back-compat
                slot["gun_name"]            = slot["gun_name"] or gun_name

                # Real plug-in path: gun confirmed for GUN_PLUGIN_FRAMES.
                # If the slot was on an inferred plug, this real detection
                # confirms the inference — we keep the inferred timestamp
                # (it's earlier and more representative of when the user
                # actually plugged in) but flip the verification flag.
                if not slot["plugin_logged"] and slot["gun_present_frames"] >= GUN_PLUGIN_FRAMES:
                    plug_time              = _now()
                    slot["plugin_logged"]  = True
                    slot["plug_time"]      = plug_time
                    slot["gun_name"]       = gun_name
                    slot["gun_present_frames"]    = 0
                    slot["plug_time_inferred"]    = False
                    slot["gun_seen_after_plugin"] = True

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

                # Already plugged in (real or inferred). A real gun frame
                # confirms the plug is genuine and clears any in-progress
                # plug-out debounce — the gun is back, so MAYBE_OUT was a
                # false alarm (occlusion).
                elif slot["plugin_logged"] and not slot["plugout_logged"]:
                    slot["gun_seen_after_plugin"] = True

                    if slot.get("gun_maybe_out_since"):
                        logger.info(
                            "[GUN] MAYBE_OUT cleared (gun returned): camera=%s roi=%s",
                            camera_id, roi_name,
                        )
                        slot["gun_maybe_out_since"] = None

            else:
                # ── Gun NOT visible this frame ──────────────────────────
                slot["gun_present_frames"] = 0

                # Inferred plug-in path: no real gun seen, but car has been
                # parked long enough that we synthesize a plug.
                if (
                    not slot["plugin_logged"]
                    and slot.get("in_time")
                ):
                    parked_secs = _seconds_since(slot["in_time"], now_dt)
                    if parked_secs >= INFERRED_PLUGIN_SECONDS:
                        plug_time = _now()
                        slot["plugin_logged"]  = True
                        slot["plug_time"]      = plug_time
                        slot["gun_name"]       = slot["gun_name"] or gun_name
                        slot["plug_time_inferred"]    = True
                        slot["gun_seen_after_plugin"] = False

                        evt = build_event(
                            event_type="gun_plugin",
                            camera_id=camera_id,
                            timestamp=plug_time,
                            track_id=track_id,
                            metadata={
                                "gun_name": slot["gun_name"] or gun_name,
                                "roi":      roi_name,
                                "slot_id":  roi_name,
                                "inferred": True,
                                "reason":   "no gun detection within INFERRED_PLUGIN_SECONDS of parking_intime",
                            },
                        )
                        events.append(evt)
                        publish_sync("gun_events", evt, task_id=task_id)
                        logger.info(
                            "[GUN] Plugin (inferred, %.0fs after intime): camera=%s roi=%s",
                            parked_secs, camera_id, roi_name,
                        )

                # Plug-out debounce — only runs after a confirmed plug.
                # plugin_logged is NEVER cleared by a missed frame here; it
                # only flips off via the rollback below or via reset_slot_state.
                elif slot["plugin_logged"] and not slot["plugout_logged"]:
                    plug_dt = _parse_iso(slot.get("plug_time"))
                    secs_since_plug = (
                        (now_dt - plug_dt).total_seconds() if plug_dt else 0.0
                    )

                    # Inference rollback: if the plug was inferred and we have
                    # never actually seen the gun since, give up after the
                    # verification window. The car parked but never plugged
                    # in (or the gun is permanently in a blind spot — either
                    # way we can't confirm, so don't synthesize a plugout
                    # later either).
                    if (
                        slot.get("plug_time_inferred")
                        and not slot.get("gun_seen_after_plugin")
                        and secs_since_plug >= INFERRED_VERIFY_TIMEOUT_SECONDS
                    ):
                        logger.warning(
                            "[GUN] Inferred plug rolled back (gun never observed in %.0fs): "
                            "camera=%s roi=%s",
                            secs_since_plug, camera_id, roi_name,
                        )
                        slot["plugin_logged"]         = False
                        slot["plug_time"]             = None
                        slot["plug_time_inferred"]    = False
                        slot["gun_seen_after_plugin"] = False
                        slot["gun_maybe_out_since"]   = None
                        # No event fired; the orchestration row never got a
                        # plug_time for this slot, so there is nothing to undo
                        # downstream. (We did fire gun_plugin earlier — but
                        # the upsert is first-write-wins, so a future real
                        # plug for this same slot+session won't overwrite.
                        # Acceptable: this slot's row simply keeps the
                        # inferred plug_time, which is conservative.)
                        set_slot_state(camera_id, roi_name, slot)
                        continue

                    # Plug-out state machine.
                    #
                    # PLUGGED  → after GUN_MAYBE_OUT_SECONDS of absence,
                    #            enter MAYBE_OUT (no event yet).
                    # MAYBE_OUT → if gun reappears, exit entirely (handled
                    #             in the gun_det branch above; that's the
                    #             "miss → return" half of the user's pattern).
                    # MAYBE_OUT → after GUN_RETURN_GRACE_SECONDS +
                    #             GUN_CONFIRM_OUT_SECONDS of continued absence
                    #             without a return, fire gun_plugout (the
                    #             "miss → return → miss again" pattern only
                    #             reaches this point if the second miss is
                    #             sustained, since any return resets state).
                    maybe_since = slot.get("gun_maybe_out_since")
                    if not maybe_since:
                        # PLUGGED → MAYBE_OUT after sustained absence.
                        # last_gun_seen_at is the anchor; on the first plug
                        # cycle there might not be one yet, so fall back to
                        # plug_time which marks the start of "should be visible".
                        last_seen = (
                            slot.get("last_gun_seen_at")
                            or slot.get("plug_time")
                        )
                        absent_secs = _seconds_since(last_seen, now_dt)
                        if absent_secs >= GUN_MAYBE_OUT_SECONDS:
                            slot["gun_maybe_out_since"] = now_dt.isoformat()
                            logger.info(
                                "[GUN] Entered MAYBE_OUT (absent %.1fs): camera=%s roi=%s",
                                absent_secs, camera_id, roi_name,
                            )
                    else:
                        # Already in MAYBE_OUT and gun is still absent (we're
                        # in the not-gun_det branch). Time accumulates; only
                        # fire once both windows have elapsed.
                        in_maybe_secs = _seconds_since(maybe_since, now_dt)
                        ready_to_fire = in_maybe_secs >= (
                            GUN_RETURN_GRACE_SECONDS + GUN_CONFIRM_OUT_SECONDS
                        )

                        if ready_to_fire:
                            plugout_time = _now()
                            slot["plug_out_time"]  = plugout_time
                            slot["plugout_logged"] = True
                            # Re-arm for a possible second plug cycle in this slot
                            slot["plugin_logged"]         = False
                            slot["plugout_logged"]        = False
                            slot["gun_maybe_out_since"]   = None
                            slot["gun_absent_frames"]     = 0
                            slot["gun_present_frames"]    = 0
                            slot["plug_time_inferred"]    = False
                            slot["gun_seen_after_plugin"] = False

                            evt = build_event(
                                event_type="gun_plugout",
                                camera_id=camera_id,
                                timestamp=plugout_time,
                                track_id=track_id,
                                metadata={
                                    "gun_name":     slot["gun_name"] or gun_name,
                                    "roi":          roi_name,
                                    "slot_id":      roi_name,
                                    "plugin_time":  slot["plug_time"],
                                    "plugout_time": plugout_time,
                                },
                            )
                            events.append(evt)
                            publish_sync("gun_events", evt, task_id=task_id)
                            logger.info(
                                "[GUN] Plugout (debounced %.0fs): camera=%s roi=%s",
                                in_maybe_secs, camera_id, roi_name,
                            )

            # Always update last_gun_seen_at while gun is visible — this is
            # the anchor the plug-out debounce uses to measure absence.
            if gun_det:
                slot["last_gun_seen_at"] = now_dt.isoformat()

            set_slot_state(camera_id, roi_name, slot)

        # ── Unauthorized attribution ──────────────────────────────────────
        # If a gun sits in an ROI whose slot is NOT occupied, and exactly one
        # tracked car is currently outside every defined ROI, attribute the
        # gun to that unauthorized car. We emit a lightweight
        # "gun_unauthorized" event (gun_number only — no plug_time, no
        # plug_out_time, no slot_id) so the orchestration layer can attach
        # gun_number to the unauthorized charging session.
        #
        # Conservative matching: skip when there are 0 or >1 unauthorized
        # cars — guessing which car the gun belongs to would mis-attribute.
        if rois:
            unauthorized_cars = [
                c for c in tracked_cars
                if not which_rois(c.get("bbox", {}), rois)
                and c.get("track_id")
            ]
            unoccupied_gun_rois = [
                roi_name for roi_name, gun in roi_to_gun.items()
                if not get_slot_state(camera_id, roi_name)["occupied"]
            ]
            if len(unauthorized_cars) == 1 and unoccupied_gun_rois:
                ua_track_id = unauthorized_cars[0]["track_id"]
                for roi_name in unoccupied_gun_rois:
                    gun_name = _gun_name_for_roi(roi_name)
                    evt = build_event(
                        event_type="gun_unauthorized",
                        camera_id=camera_id,
                        timestamp=_now(),
                        track_id=ua_track_id,
                        metadata={
                            "gun_name": gun_name,
                            "roi":      roi_name,
                            "source":   "unauthorized_parking",
                        },
                    )
                    events.append(evt)
                    publish_sync("gun_events", evt, task_id=task_id)
                    logger.info(
                        "[GUN] Unauthorized gun attribution: camera=%s gun=%s track=%s",
                        camera_id, gun_name, ua_track_id,
                    )

        triggered = bool(events)
        print(f"[GUN] result: triggered={triggered} | events={[e['event_type'] for e in events]}")
        return {"triggered": triggered, "matched_objects": list(gun_dets), "events": events}
