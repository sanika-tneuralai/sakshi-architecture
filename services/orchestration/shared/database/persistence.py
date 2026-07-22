"""
Orchestration DB helpers

Provides helpers that the orchestration layer uses at pipeline runtime:
  - get_camera_rois / get_camera_usecases / get_class_thresholds : config reads
  - upsert_charging_session       : session create / update from usecase events
  - persist_alerts_from_results   : alert deduplication and write
  - close_stale_sessions          : periodic stale session cleanup
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from sqlalchemy.exc import IntegrityError

from shared.database.connection import SessionLocal
from shared.database.models import ROIConfig, CameraUsecase, ChargingSession, Alert, Camera


# ---------------------------------------------------------------------------
# Camera registration helpers
# ---------------------------------------------------------------------------

def upsert_camera_rtsp(camera_id: str, rtsp_url: str, fps: int,
                       name: Optional[str] = None, location: Optional[str] = None) -> None:
    """Persist the RTSP URL + fps (+ optional name/location) for a camera_id.

    Creates the row if it doesn't exist; otherwise updates in place. name/location
    are only written when provided, so a plain /pipeline/start replay doesn't wipe
    them. Called from POST /pipeline/start so the orchestrator can push
    /camera/start to decode-detect on its own (now and again on every recovery).
    """
    session = SessionLocal()
    try:
        cam = session.query(Camera).filter(Camera.camera_id == camera_id).one_or_none()
        if cam is None:
            cam = Camera(camera_id=camera_id, rtsp_url=rtsp_url, fps=fps, name=name, location=location)
            session.add(cam)
        else:
            cam.rtsp_url = rtsp_url
            cam.fps = fps
            if name is not None:
                cam.name = name
            if location is not None:
                cam.location = location
        session.commit()
    finally:
        session.close()


def update_camera_meta(camera_id: str, name: Optional[str] = None,
                       location: Optional[str] = None, rtsp_url: Optional[str] = None,
                       fps: Optional[int] = None) -> bool:
    """Edit an existing camera row's user-facing fields in place.

    Only fields passed as non-None are touched, so a partial edit (e.g. just the
    station name) leaves the rest alone. For name/location an empty string is
    treated as "clear to NULL"; rtsp_url is never cleared (it's required to run).
    Returns False if no such camera_id exists. Powers the dashboard's inline
    edit on the onboarding list.
    """
    session = SessionLocal()
    try:
        cam = session.query(Camera).filter(Camera.camera_id == camera_id).one_or_none()
        if cam is None:
            return False
        if name is not None:
            cam.name = name.strip() or None
        if location is not None:
            cam.location = location.strip() or None
        if rtsp_url is not None and rtsp_url.strip():
            cam.rtsp_url = rtsp_url.strip()
        if fps is not None:
            cam.fps = fps
        session.commit()
        return True
    finally:
        session.close()


def upsert_camera_rois(camera_id: str, rois: List[Dict[str, Any]]) -> int:
    """Replace a camera's parking-zone (non-logo) ROIs with `rois`.

    Each roi: {"roi_id": str, "points": [[x,y], ...], "label": str|None}. roi_type
    is forced to 'parking' (any non-'logo' type; the usecase layer treats these as
    parking slots keyed by roi_id). Logo ROIs are left untouched. Returns the count
    written. Used by the dashboard's draw-ROI-on-snapshot onboarding step.
    """
    session = SessionLocal()
    try:
        session.query(ROIConfig).filter(
            ROIConfig.camera_id == camera_id,
            ROIConfig.roi_type != "logo",
        ).delete(synchronize_session=False)
        for r in rois:
            session.add(ROIConfig(
                camera_id=camera_id,
                roi_id=r["roi_id"],
                roi_type="parking",
                points=r["points"],
                label=r.get("label"),
            ))
        session.commit()
        return len(rois)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_registered_cameras() -> List[Dict[str, Any]]:
    """Return every camera row that has an rtsp_url set (with name/location).

    Used by the orchestrator on startup and by the decode-detect heartbeat to
    re-push /camera/start, and by the dashboard camera-management/overview views.
    Rows without an rtsp_url are skipped — register one via /pipeline/start first.
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(Camera)
            .filter(Camera.rtsp_url.isnot(None))
            .all()
        )
        return [
            {
                "camera_id": r.camera_id, "rtsp_url": r.rtsp_url, "fps": r.fps,
                "name": r.name, "location": r.location,
            }
            for r in rows
        ]
    finally:
        session.close()


def delete_camera(camera_id: str) -> Dict[str, Any]:
    """Remove a camera from the dashboard/registry.

    First drops the camera's config rows (ROIs + usecase mappings), then tries to
    delete the Camera row itself. A camera that has accumulated history — detections,
    usecase_results, alerts, analytics_daily, charging_sessions all FK to
    camera.camera_id — cannot be hard-deleted without violating those constraints
    (and we don't want to silently wipe operational history). In that case we fall
    back to a *soft delete*: clear rtsp_url so the row drops out of
    list_registered_cameras() — the orchestrator won't replay it and it disappears
    from the dashboard — while the historical rows stay intact.

    Returns {"found": bool, "hard_deleted": bool, "soft_deleted": bool}. A re-add
    with the same camera_id works either way (upsert_camera_rtsp updates the row
    in place if a soft-deleted one lingers).
    """
    session = SessionLocal()
    try:
        cam = session.query(Camera).filter(Camera.camera_id == camera_id).one_or_none()
        if cam is None:
            return {"found": False, "hard_deleted": False, "soft_deleted": False}

        # Drop config that only makes sense while the camera exists. Safe to delete
        # outright — these are admin config, not audit history.
        session.query(ROIConfig).filter(ROIConfig.camera_id == camera_id).delete(
            synchronize_session=False
        )
        session.query(CameraUsecase).filter(CameraUsecase.camera_id == camera_id).delete(
            synchronize_session=False
        )
        session.commit()

        # Try the hard delete of the camera row itself.
        try:
            session.delete(cam)
            session.commit()
            return {"found": True, "hard_deleted": True, "soft_deleted": False}
        except IntegrityError:
            # Has history rows (detections / sessions / alerts / analytics). Keep the
            # row for referential integrity but detach it from the live registry.
            session.rollback()
            cam = session.query(Camera).filter(Camera.camera_id == camera_id).one_or_none()
            if cam is not None:
                cam.rtsp_url = None
                session.commit()
            return {"found": True, "hard_deleted": False, "soft_deleted": True}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Minimum charging duration
#
# A vehicle's session must last at least this long for it to count as a real
# charging visit. Anything shorter is almost always tracker noise: a tracker
# drop + re-acquire that fragmented one physical visit into multiple rows,
# or a car that pulled in and pulled out without ever plugging in.
#
# Sessions under the floor are flagged session_status='discarded' rather
# than deleted. Keeping the row preserves audit trail (you can prove the
# discard happened) and lets dashboards / analytics / energy comparison
# filter them out by status without a schema migration.
#
# Tunable via env var; default 15 minutes.
# ---------------------------------------------------------------------------
MIN_SESSION_MINUTES = int(os.getenv("MIN_SESSION_MINUTES", "15"))


def _is_below_min_duration(in_time: datetime | None, out_time: datetime | None) -> bool:
    """True when (out_time - in_time) is non-null and below MIN_SESSION_MINUTES."""
    if in_time is None or out_time is None:
        return False
    # Defensive: both sides should be tz-aware (DateTime(timezone=True) columns),
    # but normalize anyway so a stray naive datetime doesn't raise mid-commit.
    if in_time.tzinfo is None:
        in_time = in_time.replace(tzinfo=timezone.utc)
    if out_time.tzinfo is None:
        out_time = out_time.replace(tzinfo=timezone.utc)
    return (out_time - in_time) < timedelta(minutes=MIN_SESSION_MINUTES)


# ---------------------------------------------------------------------------
# Config reads
# ---------------------------------------------------------------------------

def get_camera_rois(camera_id: str) -> List[Dict[str, Any]]:
    """
    Return all active ROI definitions for a camera.

    Each dict matches the shape expected by usecase rules:
        {
            "roi_id":   str,
            "roi_type": str,
            "points":   [[x, y], ...],
            "label":    str | None,
            "metadata": dict
        }

    Returns an empty list if no ROIs are configured or on DB error.
    """
    db = SessionLocal()
    try:
        rows = db.query(ROIConfig).filter(ROIConfig.camera_id == camera_id).all()
        return [
            {
                "roi_id":   row.roi_id,
                "roi_type": row.roi_type,
                "points":   row.points,
                "label":    row.label,
                "metadata": row.roi_metadata or {},
            }
            for row in rows
        ]
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[DB] Failed to fetch ROIs for camera {camera_id}: {e}"
        )
        return []
    finally:
        db.close()


def get_camera_usecases(camera_id: str) -> List[str]:
    """
    Return the list of enabled usecase IDs for a camera.

    Returns an empty list if no usecases are configured or on DB error.
    The caller is responsible for falling back to defaults when the list is empty.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(CameraUsecase)
            .filter(
                CameraUsecase.camera_id == camera_id,
                CameraUsecase.enabled == True,  # noqa: E712
            )
            .order_by(CameraUsecase.id)
            .all()
        )
        return [row.usecase_id for row in rows]
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[DB] Failed to fetch usecases for camera {camera_id}: {e}"
        )
        return []
    finally:
        db.close()


# Mapping from usecase_id → YOLO class names the usecase operates on.
_USECASE_CLASSES: Dict[str, list] = {
    "gun_detection":      ["gun"],
    "safety_monitoring":  ["fire", "smoke"],
    "parking_detection":  ["car", "motorcycle"],
    "parking_compliance": ["car", "motorcycle"],
    "vehicle_extraction": ["car"],   # car-only by design: motorcycles skip the extraction LLM
    "phone_detection":    ["cell phone"],
    "smoking_detection":  ["cigarette"],
    "mopping_detection":  ["mop"],
    "cash_detection":     ["cash", "cash_drawer"],
    "bag_detection":      ["backpack", "handbag", "suitcase"],
    "dress_code":         ["uniform_grey", "uniform_black", "uniform_beige", "uniform_blue", "uniform_red",
                           "untucked_shirt", "no_uniform"],
    "staff_detector":     ["grey_uniform", "black_uniform", "beige_uniform", "blue_uniform", "red_uniform"],
    "restricted_area":    ["uniform_grey", "uniform_black", "uniform_beige", "uniform_blue", "uniform_red",
                           "no_uniform", "violation_uniform"],
}


def get_class_thresholds(camera_id: str) -> Dict[str, float]:
    """
    Return per-class confidence thresholds for a camera, derived from
    per-usecase confidence_threshold values in camera_usecases.config.

    Returns an empty dict if no thresholds are configured or on DB error.
    The caller falls back to the global confidence_threshold when empty.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(CameraUsecase)
            .filter(
                CameraUsecase.camera_id == camera_id,
                CameraUsecase.enabled == True,  # noqa: E712
            )
            .all()
        )
        merged: Dict[str, float] = {}
        for row in rows:
            cfg = row.config or {}
            threshold = cfg.get("confidence_threshold")
            if threshold is None:
                continue
            for class_name in _USECASE_CLASSES.get(row.usecase_id, []):
                merged[class_name] = threshold
        return merged
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[DB] Failed to fetch class_thresholds for camera {camera_id}: {e}"
        )
        return {}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Charging session upsert
# ---------------------------------------------------------------------------

_persistence_logger = logging.getLogger(__name__)


def upsert_charging_session(camera_id: str, usecase_results: list) -> None:
    """
    Find or create an active ChargingSession for camera_id and update it with
    data extracted from the current pipeline iteration's usecase results.

    Session identity: UNIQUE(camera_id, slot_id, in_time) — enforced at the DB
    level. Any duplicate insert (e.g. from a Redis-lost cold start) silently
    fails due to ON CONFLICT DO NOTHING, so the existing row is preserved.

    Session lookup: open session at (camera_id, slot_id) — single stable key.

    Field updates: first-write-wins. A non-null field is never overwritten.

    Status derivation (on every update):
      active     — in_time set, no plug_time
      charging   — plug_time set, out_time not yet set
      completed  — out_time set AND plug_time set
      incomplete — out_time set, no plug_time
    """
    def _parse_dt(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # 1. Extract events from usecase results, grouped by slot_id           #
    #                                                                      #
    # Unauthorized parking (parking_compliance with                        #
    # metadata.source="unauthorized_parking") has no slot_id — those       #
    # events are bucketed by track_id in parking_by_track. Their session   #
    # row is written with slot_id=NULL.                                    #
    # ------------------------------------------------------------------ #
    parking_by_slot: dict = {}    # {slot_id: {"in_time": ts, "out_time": ts, "track_id": str}}
    parking_by_track: dict = {}   # {track_id: {"in_time": ts, "out_time": ts}}  — unauthorized only
    gun_by_slot: dict = {}        # {slot_id: {"plug_time": ts, "plug_out_time": ts, "gun_number": str, "track_id": str}}
    gun_by_track: dict = {}       # {track_id: {"gun_number": str}}  — unauthorized only, no plug times
    vehicle_by_slot: dict = {}    # {slot_id: {"car_number": str, "car_model": str, "vehicle_type": str}}
    vehicle_by_track: dict = {}   # {track_id: {"car_number": str, "car_model": str, "vehicle_type": str}}

    for result in usecase_results:
        usecase_id = result.get("usecase_id") or result.get("usecase_name", "")
        extras = result.get("extras") or {}

        _event_usecases = {"parking_detection", "parking_compliance", "gun_detection"}
        has_events = bool(extras.get("events")) or bool(extras.get("vehicle_details"))
        if not result.get("triggered") and not (usecase_id in _event_usecases and has_events):
            continue

        if usecase_id in ("parking_detection", "parking_compliance"):
            for evt in extras.get("events", result.get("events", [])):
                etype   = evt.get("event_type")
                ts      = evt.get("timestamp")
                meta    = evt.get("metadata", {})
                slot_id = meta.get("slot_id") or meta.get("roi")
                tid     = evt.get("track_id")
                if etype not in ("parking_intime", "parking_outtime"):
                    continue

                # Unauthorized parking: no slot_id, bucket by track_id
                if not slot_id:
                    if meta.get("source") != "unauthorized_parking" or not tid:
                        continue
                    if tid not in parking_by_track:
                        parking_by_track[tid] = {"in_time": None, "out_time": None, "vehicle_type": None}
                    # Carry the YOLO-derived vehicle_type stamped on the event —
                    # the only two-wheeler signal for a motorcycle (it skips the
                    # extraction LLM).
                    if parking_by_track[tid].get("vehicle_type") in (None, "unknown"):
                        parking_by_track[tid]["vehicle_type"] = meta.get("vehicle_type")
                    if etype == "parking_intime" and parking_by_track[tid]["in_time"] is None:
                        parking_by_track[tid]["in_time"] = ts
                    elif etype == "parking_outtime" and parking_by_track[tid]["out_time"] is None:
                        parking_by_track[tid]["out_time"] = ts
                    continue

                if slot_id not in parking_by_slot:
                    parking_by_slot[slot_id] = {"in_time": None, "out_time": None, "track_id": None, "last_gun_seen_at": None, "vehicle_type": None}
                # Carry the YOLO-derived vehicle_type stamped on the parking
                # event. Motorcycles skip vehicle_extraction (no LLM), so this is
                # the only vehicle_type signal for a two-wheeler — used below to
                # keep it out of charging_sessions entirely (car charging station).
                if parking_by_slot[slot_id].get("vehicle_type") in (None, "unknown"):
                    parking_by_slot[slot_id]["vehicle_type"] = meta.get("vehicle_type")
                if etype == "parking_intime" and parking_by_slot[slot_id]["in_time"] is None:
                    parking_by_slot[slot_id]["in_time"]  = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid
                elif etype == "parking_outtime" and parking_by_slot[slot_id]["out_time"] is None:
                    parking_by_slot[slot_id]["out_time"] = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid
                    # Last frame the gun was on the car ≈ real unplug moment.
                    # Used to record an accurate plug_out_time on departure
                    # instead of inferring = out_time (see inference below).
                    if parking_by_slot[slot_id].get("last_gun_seen_at") is None:
                        parking_by_slot[slot_id]["last_gun_seen_at"] = meta.get("last_gun_seen_at")

        elif usecase_id == "gun_detection":
            for evt in extras.get("events", result.get("events", [])):
                etype   = evt.get("event_type")
                ts      = evt.get("timestamp")
                meta    = evt.get("metadata", {})
                slot_id = meta.get("slot_id") or meta.get("roi")
                tid     = evt.get("track_id")

                # Gun detected near an unauthorized car (outside all ROIs).
                # Attach gun_number only; no plug_time/plug_out_time recorded
                # because slot-based plug semantics don't apply.
                if etype == "gun_unauthorized" and tid:
                    gun_name = meta.get("gun_name")
                    if gun_name:
                        gun_by_track.setdefault(tid, {"gun_number": None})
                        gun_by_track[tid]["gun_number"] = gun_by_track[tid]["gun_number"] or gun_name
                    continue

                if not slot_id:
                    continue
                if slot_id not in gun_by_slot:
                    gun_by_slot[slot_id] = {"plug_time": None, "plug_out_time": None, "gun_number": None, "track_id": None}
                if etype == "gun_plugin" and gun_by_slot[slot_id]["plug_time"] is None:
                    gun_by_slot[slot_id]["plug_time"]   = ts
                    gun_by_slot[slot_id]["gun_number"]  = gun_by_slot[slot_id]["gun_number"] or meta.get("gun_name")
                    gun_by_slot[slot_id]["track_id"]    = gun_by_slot[slot_id]["track_id"] or tid
                elif etype == "gun_plugout" and gun_by_slot[slot_id]["plug_out_time"] is None:
                    gun_by_slot[slot_id]["plug_out_time"] = ts
                    gun_by_slot[slot_id]["gun_number"]    = gun_by_slot[slot_id]["gun_number"] or meta.get("gun_name")
                    gun_by_slot[slot_id]["track_id"]      = gun_by_slot[slot_id]["track_id"] or tid

        elif usecase_id == "vehicle_extraction":
            for d in extras.get("vehicle_details", result.get("vehicle_details", [])):
                sid = d.get("slot_id")
                tid = d.get("track_id")
                entry = {
                    "car_number":   d.get("car_number"),
                    "car_model":    d.get("car_model"),
                    "vehicle_type": d.get("vehicle_type"),
                }
                if sid:
                    vehicle_by_slot[sid] = entry
                elif tid:
                    # Unauthorized-parking car: no slot, key by track_id
                    vehicle_by_track[tid] = entry

    # ------------------------------------------------------------------ #
    # 2. Upsert one session per slot_id seen this frame                    #
    # ------------------------------------------------------------------ #
    all_slots = set(parking_by_slot.keys()) | set(gun_by_slot.keys())
    if not all_slots and not parking_by_track and not gun_by_track:
        return

    for slot_id in all_slots:
        p = parking_by_slot.get(slot_id, {})
        g = gun_by_slot.get(slot_id, {})

        in_time       = _parse_dt(p.get("in_time"))
        out_time      = _parse_dt(p.get("out_time"))
        plug_time     = _parse_dt(g.get("plug_time"))
        plug_out_time = _parse_dt(g.get("plug_out_time"))
        last_gun_seen_at = _parse_dt(p.get("last_gun_seen_at"))
        gun_number    = g.get("gun_number")
        track_id      = p.get("track_id") or g.get("track_id")

        # Vehicle enrichment — slot_id is the only key (track_id fallback removed
        # because slot-anchored extraction always provides slot_id)
        car_number = None
        car_model  = None
        vehicle_type = None
        vd = vehicle_by_slot.get(slot_id)
        if vd:
            cn = vd.get("car_number")
            cm = vd.get("car_model")
            vt = vd.get("vehicle_type")
            if cn not in (None, "unreadable", "unknown"):
                car_number = cn
            if cm not in (None, "unknown"):
                car_model = cm
            if vt not in (None, "unknown"):
                vehicle_type = vt
        # Fall back to the YOLO-seeded vehicle_type carried on the parking event
        # — motorcycles skip the extraction LLM, so vehicle_by_slot is empty for
        # them.
        if vehicle_type is None:
            pvt = p.get("vehicle_type")
            if pvt not in (None, "unknown"):
                vehicle_type = pvt

        # This is a car charging station: two-wheelers never get a charging
        # session row. A motorcycle's only record is the non_ev_parking alert
        # raised by parking_compliance. Skip the slot before any row is created.
        if vehicle_type == "two_wheeler":
            _persistence_logger.info(
                f"[DB][{camera_id}] slot={slot_id} is a two-wheeler — "
                f"skipping charging_session (alert-only)"
            )
            continue

        _persistence_logger.info(
            f"[DB][{camera_id}] upsert slot={slot_id} track={track_id} "
            f"in_time={p.get('in_time')} out_time={p.get('out_time')} "
            f"plug_time={g.get('plug_time')} plug_out_time={g.get('plug_out_time')} "
            f"gun={gun_number} car={car_number}"
        )

        if not any([in_time, out_time, plug_time, plug_out_time, gun_number, car_number, car_model]):
            _persistence_logger.info(
                f"[DB][{camera_id}] Nothing actionable for slot={slot_id} — skipping"
            )
            continue

        db = SessionLocal()
        try:
            open_statuses = ("active", "charging")

            # Primary lookup: open session at (camera_id, slot_id)
            session: ChargingSession | None = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.slot_id == slot_id,
                    ChargingSession.session_status.in_(open_statuses),
                )
                .order_by(ChargingSession.session_id.desc())
                .first()
            )

            # Stale-predecessor guard: if the open session's in_time is older
            # than the current event's in_time by at least MIN_SESSION_MINUTES,
            # the row belongs to a previous car whose departure was missed by
            # detection (gun_plugout / parking_outtime never fired). Close it
            # at the new arrival's in_time and let the block below create a
            # fresh row for the actual new car. Without this, first-write-wins
            # silently merges the new car's events into the stale row.
            if (
                session is not None
                and in_time is not None
                and session.in_time is not None
                and (in_time - session.in_time) >= timedelta(minutes=MIN_SESSION_MINUTES)
            ):
                if session.out_time is None:
                    session.out_time = in_time
                if session.plug_time is not None and session.plug_out_time is None:
                    session.plug_out_time = session.out_time
                # Sessions that reached "charging" are real visits — don't
                # discard them based on the (synthesized) closure timestamp.
                if session.plug_time is not None:
                    session.session_status = "completed"
                elif _is_below_min_duration(session.in_time, session.out_time):
                    session.session_status = "discarded"
                else:
                    session.session_status = "completed"
                _persistence_logger.warning(
                    f"[DB] Force-closed stale predecessor session_id={session.session_id} "
                    f"slot={slot_id} predecessor_in={session.in_time} new_in={in_time} "
                    f"(gap >= {MIN_SESSION_MINUTES} min) — opening fresh row for new arrival"
                )
                db.flush()
                session = None

            # Create a new session only when in_time is present.
            # The UNIQUE(camera_id, slot_id, in_time) constraint at the DB level
            # ensures that even if parking_intime fires twice (e.g. after a full
            # Redis+service restart that bypassed bootstrap), only one row is kept.
            if session is None:
                if in_time is None:
                    _persistence_logger.debug(
                        f"[DB] Skipping new session for camera {camera_id} slot={slot_id} — no in_time yet"
                    )
                    continue
                session = ChargingSession(
                    camera_id=camera_id,
                    slot_id=slot_id,
                    track_id=track_id,
                    in_time=in_time,
                    session_status="active",
                )
                db.add(session)
                db.flush()
                _persistence_logger.info(
                    f"[DB] Created new ChargingSession session_id={session.session_id} "
                    f"slot={slot_id} track={track_id} for camera {camera_id}"
                )

            # ---- First-write-wins field updates ---- #
            if track_id    and session.track_id    is None: session.track_id    = track_id
            if gun_number  and session.gun_number  is None: session.gun_number  = gun_number
            if car_number  and session.car_number  is None: session.car_number  = car_number
            if car_model   and session.car_model   is None: session.car_model   = car_model
            if in_time     and session.in_time     is None: session.in_time     = in_time
            if plug_time   and session.plug_time   is None: session.plug_time   = plug_time
            if plug_out_time and session.plug_out_time is None: session.plug_out_time = plug_out_time
            if out_time    and session.out_time    is None: session.out_time    = out_time

            # If the car has left (out_time set) and was charging (plug_time set)
            # but no gun_plugout event fired, record plug_out_time. Prefer
            # last_gun_seen_at (the last frame the gun was actually on the car,
            # carried on the parking_outtime event ≈ the real unplug moment) —
            # this is accurate when the driver unplugged and drove off before a
            # gun_plugout could commit. Fall back to out_time as the upper-bound
            # proxy when last_gun_seen_at is unavailable (e.g. gun blind-spot).
            if (
                session.out_time is not None
                and session.plug_time is not None
                and session.plug_out_time is None
            ):
                session.plug_out_time = last_gun_seen_at or session.out_time

            # ---- Status derivation ---- #
            # A session is "completed" once both in_time and out_time exist —
            # plug events are optional metadata, not status drivers (some valid
            # sessions never produce plug detections, e.g. blind-spot guns or
            # cars that left without charging).
            #
            # Sub-floor visits get session_status='discarded' instead. These
            # are almost always tracker fragmentation or pull-in-pull-out
            # non-events; flagging them keeps the dashboard, analytics, and
            # energy comparison clean while preserving the row for audit.
            if session.in_time is not None and session.out_time is not None:
                if _is_below_min_duration(session.in_time, session.out_time):
                    session.session_status = "discarded"
                    _persistence_logger.info(
                        f"[DB] Discarded short session session_id={session.session_id} "
                        f"slot={session.slot_id} duration_min="
                        f"{(session.out_time - session.in_time).total_seconds() / 60:.1f} "
                        f"(< {MIN_SESSION_MINUTES} min) camera={camera_id}"
                    )
                else:
                    session.session_status = "completed"
            elif session.plug_time is not None:
                session.session_status = "charging"
            else:
                session.session_status = "active"

            db.commit()
            _persistence_logger.info(
                f"[DB] ChargingSession session_id={session.session_id} slot={session.slot_id} "
                f"track={session.track_id} status={session.session_status} | "
                f"in_time={session.in_time} plug_time={session.plug_time} out_time={session.out_time} "
                f"car={session.car_number} gun={session.gun_number} camera={camera_id}"
            )

        except Exception as e:
            db.rollback()
            _persistence_logger.warning(
                f"[DB] Failed to upsert ChargingSession for camera {camera_id} slot={slot_id}: {e}"
            )
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # 3. Unauthorized-parking sessions (no slot_id, keyed by track_id)    #
    #                                                                      #
    # Inappropriately parked cars (outside all ROIs). We capture in_time,  #
    # out_time, car_number, and car_model when available. gun_number is    #
    # attached opportunistically when a gun is detected near the car (no   #
    # slot semantics apply, so plug_time / plug_out_time stay NULL).       #
    # Session identity here is (camera_id, track_id, slot_id=NULL):        #
    # looked up via track_id among open sessions with slot_id IS NULL.     #
    # ------------------------------------------------------------------ #
    unauthorized_tracks = set(parking_by_track.keys()) | set(gun_by_track.keys())
    for track_id in unauthorized_tracks:
        p = parking_by_track.get(track_id, {})
        g = gun_by_track.get(track_id, {})

        in_time    = _parse_dt(p.get("in_time"))
        out_time   = _parse_dt(p.get("out_time"))
        gun_number = g.get("gun_number")

        car_number = None
        car_model  = None
        vehicle_type = None
        vd = vehicle_by_track.get(track_id)
        if vd:
            cn = vd.get("car_number")
            cm = vd.get("car_model")
            vt = vd.get("vehicle_type")
            if cn not in (None, "unreadable", "unknown"):
                car_number = cn
            if cm not in (None, "unknown"):
                car_model = cm
            if vt not in (None, "unknown"):
                vehicle_type = vt
        # Fall back to the YOLO-seeded vehicle_type carried on the event.
        if vehicle_type is None:
            pvt = p.get("vehicle_type")
            if pvt not in (None, "unknown"):
                vehicle_type = pvt

        # Car charging station: two-wheelers never get a charging session row,
        # in-slot or unauthorized. Skip before any row is created.
        if vehicle_type == "two_wheeler":
            _persistence_logger.info(
                f"[DB][{camera_id}] unauthorized track={track_id} is a two-wheeler — "
                f"skipping charging_session (alert-only)"
            )
            continue

        _persistence_logger.info(
            f"[DB][{camera_id}] upsert unauthorized track={track_id} "
            f"in_time={p.get('in_time')} out_time={p.get('out_time')} "
            f"gun={gun_number} car={car_number}"
        )

        if not any([in_time, out_time, car_number, car_model, gun_number]):
            continue

        db = SessionLocal()
        try:
            open_statuses = ("active", "charging")
            session: ChargingSession | None = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.slot_id.is_(None),
                    ChargingSession.track_id == track_id,
                    ChargingSession.session_status.in_(open_statuses),
                )
                .order_by(ChargingSession.session_id.desc())
                .first()
            )

            if session is None:
                # Only open a new unauthorized session when parking_compliance
                # has confirmed entry (in_time). A gun-only event for a track
                # we've never seen entering is ignored to avoid orphan rows.
                if in_time is None:
                    _persistence_logger.debug(
                        f"[DB] Skipping new unauthorized session for camera {camera_id} track={track_id} — no in_time yet"
                    )
                    continue
                session = ChargingSession(
                    camera_id=camera_id,
                    slot_id=None,
                    track_id=track_id,
                    in_time=in_time,
                    session_status="active",
                )
                db.add(session)
                db.flush()
                _persistence_logger.info(
                    f"[DB] Created unauthorized ChargingSession session_id={session.session_id} "
                    f"track={track_id} for camera {camera_id}"
                )

            if car_number and session.car_number is None: session.car_number = car_number
            if car_model  and session.car_model  is None: session.car_model  = car_model
            if gun_number and session.gun_number is None: session.gun_number = gun_number
            if in_time    and session.in_time    is None: session.in_time    = in_time
            if out_time   and session.out_time   is None: session.out_time   = out_time

            if session.in_time is not None and session.out_time is not None:
                if _is_below_min_duration(session.in_time, session.out_time):
                    session.session_status = "discarded"
                    _persistence_logger.info(
                        f"[DB] Discarded short unauthorized session session_id={session.session_id} "
                        f"track={session.track_id} duration_min="
                        f"{(session.out_time - session.in_time).total_seconds() / 60:.1f} "
                        f"(< {MIN_SESSION_MINUTES} min) camera={camera_id}"
                    )
                else:
                    session.session_status = "completed"
            else:
                session.session_status = "active"

            db.commit()
            _persistence_logger.info(
                f"[DB] Unauthorized ChargingSession session_id={session.session_id} "
                f"track={session.track_id} status={session.session_status} | "
                f"in_time={session.in_time} out_time={session.out_time} "
                f"gun={session.gun_number} car={session.car_number} camera={camera_id}"
            )

        except Exception as e:
            db.rollback()
            _persistence_logger.warning(
                f"[DB] Failed to upsert unauthorized ChargingSession camera={camera_id} track={track_id}: {e}"
            )
            raise
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Alert persistence
# ---------------------------------------------------------------------------

_NO_ALERT_USECASES = {"people_counter", "heatmap", "vehicle_extraction"}


def persist_alerts_from_results(camera_id: str, usecase_results: list) -> int:
    """
    Save Alert rows from triggered usecase results (except analytics-only rules).

    Deduplication for parking_detection and gun_detection:
      - One alert per (camera_id, slot_id, usecase_name, alert_type).
      - Checks the DB before inserting — skips if an identical alert already exists.

    parking_detection special handling:
      - parking_intime / parking_outtime → alert usecase_name='parking_detection'
      - multiple_cars_in_roi             → dropped (transient YOLO overlap artefact)

    Returns the number of alert rows written.
    """
    _persistence_logger.info(
        f"[DB][{camera_id}] DEBUG persist_alerts: received {len(usecase_results)} results, "
        f"triggered={[r.get('usecase_id', r.get('usecase_name')) for r in usecase_results if r.get('triggered')]}"
    )
    db = SessionLocal()
    written = 0
    try:
        for result in usecase_results:
            usecase_id = result.get("usecase_id") or result.get("usecase_name", "unknown")
            extras = result.get("extras") or {}

            _event_usecases = {"parking_detection", "gun_detection"}
            has_events = bool(extras.get("events"))
            if not result.get("triggered") and not (usecase_id in _event_usecases and has_events):
                continue
            if usecase_id in _NO_ALERT_USECASES:
                continue

            snapshot_url = result.get("snapshot_url")

            # ── parking_detection: split events by type ───────────────────
            if usecase_id == "parking_detection":
                events = extras.get("events", [])
                session_events   = [e for e in events if e.get("event_type") in ("parking_intime", "parking_outtime")]
                violation_events = [e for e in events if e.get("event_type") == "multiple_cars_in_roi"]

                for evt in session_events:
                    slot_id    = evt.get("metadata", {}).get("slot_id") or evt.get("metadata", {}).get("roi")
                    event_type = evt.get("event_type", "")
                    alert_type = f"parking_{event_type}"

                    if slot_id:
                        exists = (
                            db.query(Alert.alert_id)
                            .filter(
                                Alert.camera_id    == camera_id,
                                Alert.slot_id      == slot_id,
                                Alert.usecase_name == "parking_detection",
                                Alert.alert_type   == alert_type,
                            )
                            .first()
                        )
                        if exists:
                            continue

                    matched_count = result.get("matched_count", 0)
                    db.add(Alert(
                        camera_id    = camera_id,
                        slot_id      = slot_id,
                        usecase_name = "parking_detection",
                        alert_type   = alert_type,
                        message      = f"[parking_detection] {matched_count} object(s) detected | event: {event_type}",
                        status       = "sent",
                        snapshot_url = snapshot_url,
                        extras       = {"events": [evt]},
                    ))
                    written += 1

                # multiple_cars_in_roi is a transient YOLO overlap artefact,
                # not a parking rule violation — drop it from the alert log.

                continue

            # ── parking_compliance: one alert per (slot_id, event_type) ──
            if usecase_id == "parking_compliance":
                for viol in extras.get("violations", []):
                    meta       = viol.get("metadata", {})
                    slot_id    = meta.get("slot_id") or meta.get("roi")
                    event_type = viol.get("event_type", "")  # e.g. unauthorized_parking, wrong_parking

                    if not event_type:
                        continue

                    db.add(Alert(
                        camera_id    = camera_id,
                        slot_id      = slot_id,
                        usecase_name = "parking_compliance",
                        alert_type   = event_type,
                        message      = f"[parking_compliance] {event_type} | slot={slot_id}",
                        status       = "sent",
                        snapshot_url = snapshot_url,
                        extras       = {"violations": [viol]},
                    ))
                    written += 1

                continue

            # ── gun_detection: one alert per (slot_id, event_type) ────────
            if usecase_id == "gun_detection":
                for evt in extras.get("events", []):
                    event_type = evt.get("event_type", "")
                    # gun_unauthorized is a data-attribution ping for the
                    # unauthorized-parking session path — not an alertable
                    # event on its own. The parking_compliance alert already
                    # covers the violation.
                    if event_type == "gun_unauthorized":
                        continue
                    slot_id    = evt.get("metadata", {}).get("slot_id") or evt.get("metadata", {}).get("roi")
                    alert_type = event_type  # gun_plugin / gun_plugout

                    if slot_id:
                        exists = (
                            db.query(Alert.alert_id)
                            .filter(
                                Alert.camera_id    == camera_id,
                                Alert.slot_id      == slot_id,
                                Alert.usecase_name == "gun_detection",
                                Alert.alert_type   == alert_type,
                            )
                            .first()
                        )
                        if exists:
                            continue

                    meta = evt.get("metadata", {})
                    db.add(Alert(
                        camera_id    = camera_id,
                        slot_id      = slot_id,
                        usecase_name = "gun_detection",
                        alert_type   = alert_type,
                        message      = f"[gun_detection] {event_type} | gun={meta.get('gun_name')} slot={slot_id}",
                        status       = "sent",
                        snapshot_url = snapshot_url,
                        extras       = {"events": [evt]},
                    ))
                    written += 1

                continue

            # ── Generic path for all other usecases ───────────────────────
            matched_count = result.get("matched_count", 0)
            message = f"[{usecase_id}] {matched_count} object(s) detected"
            if "events" in extras and extras["events"]:
                event_types = list({e.get("event_type", "") for e in extras["events"]})
                message += f" | events: {', '.join(event_types)}"
            elif "violations" in extras and extras["violations"]:
                reasons = list({v.get("metadata", {}).get("reason", "") for v in extras["violations"]})
                message += f" | violations: {', '.join(r for r in reasons if r)}"

            db.add(Alert(
                camera_id    = camera_id,
                slot_id      = None,
                usecase_name = usecase_id,
                alert_type   = f"{usecase_id}_triggered",
                message      = message,
                status       = "sent",
                snapshot_url = snapshot_url,
                extras       = extras if extras else None,
            ))
            written += 1

        db.commit()
        _persistence_logger.debug(f"[DB] Persisted {written} alert(s) for camera {camera_id}")
    except Exception as e:
        db.rollback()
        _persistence_logger.warning(f"[DB] Failed to persist alerts for camera {camera_id}: {e}")
    finally:
        db.close()

    return written


# ---------------------------------------------------------------------------
# Stale session cleanup
# ---------------------------------------------------------------------------

SESSION_STALE_HOURS = 4


def close_stale_sessions(stale_hours: int = SESSION_STALE_HOURS) -> int:
    """
    Close any ChargingSession that has been open (active or charging) for longer
    than stale_hours without receiving an out_time.

    Sessions closed here get an inferred out_time (created_at + stale_hours) and
    are marked 'completed' (or 'discarded' if the inferred duration is below
    MIN_SESSION_MINUTES — rare, but possible when in_time was post-dated).
    The inferred out_time lets downstream analytics (energy comparison) still
    produce a duration window when detection missed the car leaving.

    Called periodically by the orchestration pipeline (every 60 iterations).
    Returns the number of sessions closed.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=stale_hours)
    db = SessionLocal()
    closed = 0
    try:
        stale = (
            db.query(ChargingSession)
            .filter(
                ChargingSession.session_status.in_(("active", "charging")),
                ChargingSession.created_at <= cutoff,
            )
            .all()
        )
        for session in stale:
            if session.out_time is None:
                # Anchor inferred out_time to updated_at — the last time the
                # pipeline had evidence of this session (last event fired for
                # this slot). For a car that silently left without producing
                # parking_outtime, updated_at is roughly when detection lost
                # the track, which is tighter and more truthful than a flat
                # created_at + stale_hours. Falls back to created_at + stale_hours
                # when updated_at is unavailable.
                last_seen = session.updated_at
                if last_seen is not None and last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                if last_seen is None:
                    created = session.created_at
                    if created is not None and created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    inferred_out = (created + timedelta(hours=stale_hours)) if created else now
                else:
                    inferred_out = last_seen
                # Clamp out_time >= plug_time so the window doesn't go negative
                # for sessions that reached "charging" before going silent.
                if session.plug_time is not None:
                    plug_t = session.plug_time
                    if plug_t.tzinfo is None:
                        plug_t = plug_t.replace(tzinfo=timezone.utc)
                    if inferred_out < plug_t:
                        inferred_out = plug_t
                session.out_time = inferred_out
            # Mirror the upsert path: if the session was charging but never produced
            # a gun_plugout, anchor plug_out_time to the (now-inferred) out_time so
            # downstream energy/duration math has a closed window.
            if session.plug_time is not None and session.plug_out_time is None:
                session.plug_out_time = session.out_time
            # Sessions that reached "charging" are real charging visits even if
            # the synthesized out_time leaves the window short — the sweep itself
            # is admitting we lost track of the true out_time, so the min-duration
            # filter (designed for active short visits) doesn't apply here.
            if session.plug_time is not None:
                session.session_status = "completed"
            elif _is_below_min_duration(session.in_time, session.out_time):
                session.session_status = "discarded"
            else:
                session.session_status = "completed"
            closed += 1
            _persistence_logger.warning(
                f"[DB] Closed stale session session_id={session.session_id} "
                f"slot={session.slot_id} camera={session.camera_id} "
                f"created_at={session.created_at} out_time={session.out_time} "
                f"status={session.session_status} (stale > {stale_hours}h)"
            )
        if closed:
            db.commit()
        _persistence_logger.info(f"[DB] Stale session sweep: closed {closed} session(s)")
    except Exception as e:
        db.rollback()
        _persistence_logger.warning(f"[DB] Failed to close stale sessions: {e}")
    finally:
        db.close()
    return closed
