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
- parking_outtime is debounced via a MAYBE_GONE state machine — see the
  threshold constants below for full timing. Brief tracker drops no longer
  fragment one physical visit into multiple ChargingSession rows. The
  debounce depends only on car presence; gun state does not gate outtime.
- On parking_outtime: slot state is fully reset via reset_slot_state().

YOLO-miss safety net (LLM polling):
- When YOLO sees no car in a slot, every LLM_POLL_INTERVAL_SECONDS we ask
  the strict-prompt LLM whether there's a vehicle in that slot's polygon.
  The verdict is sticky for the poll interval (cached in slot state) so we
  never burn an LLM call every frame.
- A True verdict injects a synthetic ``llm:{camera}:{ROI}`` track into
  roi_occupants so the rest of this rule's logic (entry debounce, swap
  detection, MAYBE_GONE) treats it identically to a YOLO-detected car.
  parking_intime fires after ENTRY_FRAMES of consistent injection.
- Replaces the previous G-marker / logo-occlusion CV fallback, which was
  removed because shadows on the painted G triggered phantom sessions.
- Gated by PARKING_DETECTION_BACKEND (env, default "llm"). Set to "yolo"
  to disable the LLM poll entirely — the surrounding YOLO tracking stays
  unchanged. Code is preserved (not deleted) so we can flip back at any
  time if YOLO regresses on a camera.

ROI polygons are injected via detection_output["rois"]:
    {"ROI_1": [[x,y], ...], "ROI_2": [[x,y], ...]}

Tracker state is persisted in Redis under ``tracker:{camera_id}``.
Entry debounce buffers are persisted under ``parking:{camera_id}``.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List

from shared.common.roi import which_rois
from usecase.domain.vehicles.events import build_event, publish_sync
from usecase.domain.vehicles.tracking import CentroidTracker
from usecase.rules.base import BaseUsecaseRule
from usecase.rules.vehicles.vehicle_extraction import llm_confirms_vehicle_in_slot
from workers.redis_state import get_state, set_state, get_slot_state, set_slot_state, reset_slot_state

logger = logging.getLogger(__name__)

ENTRY_FRAMES = 3    # consecutive frames car must be present to confirm entry

# Backend flag: "llm" (default) keeps the YOLO-miss LLM safety net active.
# "yolo" disables the LLM poll entirely — YOLO becomes the sole occupancy
# source. Switch via env at deploy time (orchestrate sets this); the LLM
# code path is preserved so we can flip back if YOLO regresses.
PARKING_DETECTION_BACKEND = os.getenv("PARKING_DETECTION_BACKEND", "llm").lower()

# How often (seconds) to re-poll the LLM for an empty-per-YOLO slot. Verdict
# is cached in slot.last_llm_poll_at / slot.last_llm_verdict so we never call
# the LLM more than once per slot per interval.
LLM_POLL_INTERVAL_SECONDS = int(os.getenv("LLM_POLL_INTERVAL_SECONDS", "60"))

# Synthetic track_id for slots where the LLM confirms a vehicle but YOLO
# does not. Deterministic per slot so vehicle_extraction can anchor to it
# for the whole session, and so a real YOLO track_id never collides
# (CentroidTracker emits integer-string ids; this prefix is stable).
def _llm_track_id(camera_id: str, roi_name: str) -> str:
    return f"llm:{camera_id}:{roi_name}"

# Exit is a debounced state machine, NOT a single-threshold check. The naive
# "absent N seconds → outtime" approach fragmented one physical visit into
# multiple ChargingSession rows whenever the tracker briefly lost the car
# (a person walking past, a frame drop, partial occlusion). The new flow:
#
#   OCCUPIED   → car visible, normal
#   MAYBE_GONE → after CAR_MAYBE_GONE_SECONDS of absence, mark suspicious
#                (no event yet)
#   ↳ car returns → exit MAYBE_GONE entirely, slot stays OCCUPIED
#                   (this is the "miss → return" half of the user's pattern)
#   ↳ stays absent CAR_RETURN_GRACE_SECONDS + CAR_CONFIRM_GONE_SECONDS
#                   without returning → fire parking_outtime
#
# Total absence before outtime fires ≈ 30 + 60 + 60 = ~2.5 minutes. Short
# enough to track real exits; long enough to absorb tracker hiccups and
# brief occlusion (a person walking past) without fragmenting one visit
# into multiple ChargingSession rows.
CAR_MAYBE_GONE_SECONDS    = 30   # absent this long → enter MAYBE_GONE
CAR_RETURN_GRACE_SECONDS  = 60   # in MAYBE_GONE; a return cancels (cleared in occupied branch)
CAR_CONFIRM_GONE_SECONDS  = 30   # additional absence after grace → fire outtime


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> str:
    return _now_dt().isoformat()


def _event_ts(detection_output: Dict[str, Any]) -> str:
    """
    Pick the timestamp that should anchor events emitted from this evaluation.

    Prefer `detection_output["timestamp"]` (the camera service stamps it from
    the frame's capture time). Fall back to wall-clock when the field is
    absent or unparseable so existing callers/tests keep working.

    Returns an ISO 8601 string — same format as the legacy `_now()` helper —
    so it's a drop-in replacement for `timestamp=...` on event payloads.
    """
    ts = detection_output.get("timestamp")
    if isinstance(ts, str) and ts:
        # Camera service serializes datetime via Pydantic — already ISO. Keep
        # the string as-is to avoid round-trip drift.
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return _now()


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
        # Anchor every event emitted from this evaluation to the frame's capture
        # time. Wall-clock comparisons (debounce timers below) keep using
        # _now_dt() — those measure real elapsed time, independent of camera.
        event_ts = _event_ts(detection_output)
        if not rois:
            logger.error("[PARKING] 'rois' missing from payload for camera '%s'", camera_id)
            return {"triggered": False, "matched_objects": [], "events": []}

        cars = [d for d in detection_output.get("detections", []) if d.get("class_name") == "car"]
        logger.debug(
            "[PARKING] camera=%s | rois=%s | cars_detected=%d",
            camera_id, list(rois.keys()), len(cars),
        )

        # ── Canonical tracker (shared across all vehicle usecases) ────────
        tracker_key = f"tracker:{camera_id}"
        tracker_state = get_state(tracker_key)
        # (B) Stickier tracker: keep a parked car's id across longer detection
        # drops and a jumpier centroid (large fisheye bboxes on a dark/wet car),
        # so it isn't re-id'd into a "new" car every drop. Env-overridable.
        tracker = CentroidTracker(
            max_disappeared=int(os.getenv("TRACKER_MAX_DISAPPEARED", "60")),
            max_distance=int(os.getenv("TRACKER_MAX_DISTANCE", "250")),
        )
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

        # ── LLM safety net for YOLO misses ───────────────────────────────
        # When YOLO sees no car in an ROI, periodically (every
        # LLM_POLL_INTERVAL_SECONDS) ask the strict-prompt LLM if there's
        # actually a vehicle in this slot's polygon. The LLM is robust to
        # low-light / odd-angle conditions where YOLO under-recalls.
        #
        # The verdict is cached on slot state so the LLM call happens at most
        # once per slot per interval — every frame in between just reuses
        # the cached verdict (sticky). A True verdict injects a synthetic
        # llm:{cam}:{ROI} track so the rest of the rule (entry debounce,
        # MAYBE_GONE, swap detection, vehicle_extraction anchor) treats it
        # identically to a YOLO entry.
        #
        # Gated by PARKING_DETECTION_BACKEND: in "yolo" mode this whole block
        # is skipped and synthetic_llm_cars stays empty — the later return
        # path already handles that as a no-op.
        synthetic_llm_cars: List[Dict[str, Any]] = []
        if PARKING_DETECTION_BACKEND == "llm":
            snapshot_url = detection_output.get("snapshot_url")
            now_for_poll = _now_dt()
            # Synthetic "cars" appended here are forwarded to vehicle_extraction
            # via matched_objects so it can run plate/model extraction on
            # YOLO-blind sessions. The synthetic bbox is the slot polygon's
            # bounding rectangle so existing which_rois / cropping code handles
            # them identically to YOLO detections.

            def _emit_synthetic(roi_name: str) -> None:
                roi_occupants[roi_name].append(_llm_track_id(camera_id, roi_name))
                poly = rois.get(roi_name) or []
                if not poly:
                    return
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                synthetic_llm_cars.append({
                    "track_id":   _llm_track_id(camera_id, roi_name),
                    "bbox": {
                        "x1": float(min(xs)), "y1": float(min(ys)),
                        "x2": float(max(xs)), "y2": float(max(ys)),
                    },
                    "class_name": "car",
                    "confidence": 1.0,
                })

            for roi_name in list(roi_occupants.keys()):
                if roi_occupants[roi_name]:
                    continue   # YOLO has a car — skip LLM, free of charge
                slot_for_poll = get_slot_state(camera_id, roi_name)
                last_poll_iso = slot_for_poll.get("last_llm_poll_at")
                elapsed = _absent_seconds(last_poll_iso, now_for_poll) if last_poll_iso else float("inf")

                if elapsed < LLM_POLL_INTERVAL_SECONDS:
                    # Within the sticky window — reuse last verdict, no new call.
                    if slot_for_poll.get("last_llm_verdict") is True:
                        _emit_synthetic(roi_name)
                    continue

                if not snapshot_url:
                    continue  # Can't ask the LLM without a frame.

                verdict = llm_confirms_vehicle_in_slot(snapshot_url, rois, roi_name)
                slot_for_poll["last_llm_poll_at"] = now_for_poll.isoformat()
                slot_for_poll["last_llm_verdict"] = verdict   # True / False / None
                set_slot_state(camera_id, roi_name, slot_for_poll)
                logger.info(
                    "[PARKING] LLM poll: camera=%s roi=%s verdict=%s",
                    camera_id, roi_name, verdict,
                )
                if verdict is True:
                    _emit_synthetic(roi_name)
        else:
            logger.debug(
                "[PARKING] LLM safety net disabled (PARKING_DETECTION_BACKEND=%s) — YOLO-only",
                PARKING_DETECTION_BACKEND,
            )

        events: List[dict]  = []
        triggered           = False

        logger.debug("[PARKING] roi_occupants=%s", roi_occupants)

        for roi_name, occupant_ids in roi_occupants.items():

            # ── Load per-slot state ───────────────────────────────────────
            slot = get_slot_state(camera_id, roi_name)

            # Advance the frame counter every iteration — monotonic, survives restarts
            slot["frame_counter"] += 1

            if occupant_ids:
                triggered = True

                # Car-swap-during-absence guard. The previous design cleared
                # MAYBE_GONE the moment any car returned to the ROI. That
                # masked the case where Car A leaves and Car B arrives within
                # the debounce window — Car A's outtime never fires, Car B
                # gets attributed to Car A's session.
                #
                # Two sub-cases trigger an immediate session boundary:
                #
                #   (a) The returning track_id differs from slot.track_id
                #       (and isn't the YOLO↔CV alias for the same physical
                #       car). The tracker confirms a different identity.
                #
                #   (b) The same track_id returns, BUT slot.car_absent_since
                #       is older than CAR_MAYBE_GONE_SECONDS. The centroid
                #       tracker re-identifies new arrivals at the same
                #       position with the same id when the gap is short
                #       (< max_disappeared frames), so a stale absence
                #       coupled with a "returning" id is a strong signal
                #       that this is actually a different physical car.
                llm_tid_for_swap = _llm_track_id(camera_id, roi_name)
                prev_tid_for_swap = slot.get("track_id")
                fire_immediate_swap = False
                swap_reason = None

                if slot["occupied"] and slot.get("car_maybe_gone_since"):
                    # A returning car during exit-debounce is a genuine new
                    # vehicle only when its track_id differs from the slot's
                    # (and isn't the LLM-poll alias for the same car). A *same*
                    # id returning is the same car — never a swap. (Dropped the
                    # old "same id after long absence" case: it fired on routine
                    # re-ids and fragmented one parked car into many sessions.)
                    different_real_id = (
                        prev_tid_for_swap is not None
                        and all(tid != prev_tid_for_swap for tid in occupant_ids)
                        and prev_tid_for_swap != llm_tid_for_swap
                        and all(tid != llm_tid_for_swap for tid in occupant_ids)
                    )
                    if different_real_id:
                        fire_immediate_swap = True
                        swap_reason = "different track_id during MAYBE_GONE"

                # (A) Identity gate. A different track_id during MAYBE_GONE is
                # usually the SAME parked car re-identified after a detection
                # drop (the tracker minted a new id), not a real new vehicle —
                # the giveaway is the slot's extracted plate is unchanged. When
                # the slot already has a confident car_number, adopt the new
                # track_id into the current session and clear the exit debounce
                # instead of firing a swap (which fragments one visit into many
                # ChargingSession rows). Genuine swaps before a plate is read
                # still fall through to the swap path below.
                if fire_immediate_swap and slot.get("car_number"):
                    adopted_tid = next(
                        (tid for tid in occupant_ids if tid != prev_tid_for_swap),
                        prev_tid_for_swap,
                    )
                    logger.info(
                        "[PARKING] Re-id adopted (same-plate session car=%s), NOT a "
                        "swap: camera=%s roi=%s old_track=%s new_track=%s (%s)",
                        slot.get("car_number"), camera_id, roi_name,
                        prev_tid_for_swap, adopted_tid, swap_reason,
                    )
                    slot["track_id"]             = adopted_tid
                    slot["car_absent_since"]     = None
                    slot["car_maybe_gone_since"] = None
                    fire_immediate_swap = False
                    swap_reason = None

                if fire_immediate_swap:
                    swap_ts = event_ts
                    outtime_evt = build_event(
                        event_type="parking_outtime",
                        camera_id=camera_id,
                        timestamp=swap_ts,
                        track_id=prev_tid_for_swap or "",
                        metadata={
                            "roi":     roi_name,
                            "slot_id": roi_name,
                            "intime":  slot.get("in_time"),
                            "outtime": swap_ts,
                            # Last frame the gun was on this (old) car ≈ real
                            # unplug moment. Lets persistence record the true
                            # plug_out_time instead of inferring = out_time when
                            # the car left before a gun_plugout could commit.
                            "last_gun_seen_at": slot.get("last_gun_seen_at"),
                            "reason":  swap_reason,
                        },
                    )
                    events.append(outtime_evt)
                    publish_sync("parking_events", outtime_evt, task_id=task_id)
                    logger.warning(
                        "[PARKING] Immediate exit during MAYBE_GONE: camera=%s roi=%s "
                        "old=%s new=%s reason=%s",
                        camera_id, roi_name, prev_tid_for_swap, occupant_ids, swap_reason,
                    )
                    # Hard reset — the new car must go through normal entry
                    # debounce (3 frames) before parking_intime fires.
                    reset_slot_state(camera_id, roi_name)
                    slot = get_slot_state(camera_id, roi_name)
                    entry_buf[roi_name].clear()
                    # Don't start counting frames toward the new entry yet;
                    # let the next iteration handle this slot from a clean state.
                    set_slot_state(camera_id, roi_name, slot)
                    continue

                # Car is present this frame — clear absent timestamp AND
                # cancel any in-progress MAYBE_GONE debounce. The car came
                # back, so the previous absence was a tracker hiccup or
                # transient occlusion, not a real exit.
                if slot.get("car_maybe_gone_since"):
                    logger.info(
                        "[PARKING] MAYBE_GONE cleared (car returned): camera=%s roi=%s",
                        camera_id, roi_name,
                    )
                    slot["car_maybe_gone_since"] = None
                slot["car_absent_since"] = None

                # Multiple cars in the same ROI — violation event (unchanged)
                if len(occupant_ids) > 1:
                    evt = build_event(
                        event_type="multiple_cars_in_roi",
                        camera_id=camera_id,
                        timestamp=event_ts,
                        track_id=",".join(occupant_ids),
                        metadata={"roi": roi_name, "slot_id": roi_name, "count": len(occupant_ids)},
                    )
                    events.append(evt)
                    publish_sync("violation_events", evt, task_id=task_id)

                # Entry debounce — fire parking_intime only when slot was idle
                for tid in occupant_ids:
                    entry_buf[roi_name][tid] = entry_buf[roi_name].get(tid, 0) + 1

                    if not slot["occupied"] and entry_buf[roi_name][tid] >= ENTRY_FRAMES:
                        intime = event_ts
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
                        # Don't fire car-swap when the only difference is YOLO
                        # vs LLM-poll id for the same physical car. A YOLO
                        # drop + LLM pickup (or the reverse) on a parked car
                        # is a detection blip, not a new vehicle. Treat any
                        # transition involving the synthetic llm:* id as the
                        # same session.
                        llm_tid = _llm_track_id(camera_id, roi_name)
                        same_physical_car = (
                            tid == llm_tid or prev_tid == llm_tid
                        )
                        if (
                            prev_tid
                            and tid != prev_tid
                            and not same_physical_car
                            and entry_buf[roi_name][tid] >= ENTRY_FRAMES
                        ):
                            # A different track_id has been consistently present for
                            # ENTRY_FRAMES — the previous car left without a clean exit
                            # (e.g. compliance violation, tracker re-ID after service
                            # restart). Fire outtime for the old car, then intime for
                            # the new one so the session boundary is correct in the DB.
                            swap_ts = event_ts

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
                                    "last_gun_seen_at": slot.get("last_gun_seen_at"),
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

                    # parking_outtime depends only on the car-absence debounce —
                    # NOT on gun state. The earlier `gun_active` guard tied
                    # outtime to gun_plugout firing, which in this deployment
                    # is unreliable (the gun detector misses most plugs/unplugs)
                    # and pushed every recorded out_time onto the FORCE_EXIT
                    # safety net at +5min past the real exit. Decoupling means
                    # outtime fires on the natural 150s debounce; if the gun
                    # rule never produced a plug_out_time, the persistence
                    # layer infers plug_out_time = out_time downstream.

                    # ── MAYBE_GONE state machine ──────────────────────────
                    # Stage 1: enter MAYBE_GONE after sustained absence.
                    if (
                        absent_secs >= CAR_MAYBE_GONE_SECONDS
                        and not slot.get("car_maybe_gone_since")
                    ):
                        slot["car_maybe_gone_since"] = now_dt.isoformat()
                        logger.info(
                            "[PARKING] Entered MAYBE_GONE (absent %.1fs): camera=%s roi=%s",
                            absent_secs, camera_id, roi_name,
                        )

                    # Stage 2: if still absent past the grace + confirm
                    # windows without a return, fire parking_outtime.
                    # (A return would have cleared car_maybe_gone_since
                    # via the occupant branch above.)
                    maybe_since = slot.get("car_maybe_gone_since")
                    in_maybe_secs = (
                        _absent_seconds(maybe_since, now_dt) if maybe_since else 0.0
                    )
                    ready_to_fire = (
                        maybe_since
                        and in_maybe_secs >= (
                            CAR_RETURN_GRACE_SECONDS + CAR_CONFIRM_GONE_SECONDS
                        )
                    )

                    if ready_to_fire:
                        outtime = event_ts
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
                                "last_gun_seen_at": slot.get("last_gun_seen_at"),
                            },
                        )
                        events.append(evt)
                        publish_sync("parking_events", evt, task_id=task_id)
                        logger.info(
                            "[PARKING] Outtime confirmed (debounced %.0fs in MAYBE_GONE): "
                            "camera=%s roi=%s track=%s",
                            in_maybe_secs, camera_id, roi_name, slot["track_id"],
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

        logger.debug(
            "[PARKING] result: triggered=%s | events=%s",
            triggered, [e["event_type"] for e in events],
        )
        return {
            "triggered":       triggered,
            "matched_objects": tracked_cars + synthetic_llm_cars,
            "events":          events,
        }
