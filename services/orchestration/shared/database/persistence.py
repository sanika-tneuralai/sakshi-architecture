"""
Orchestration DB helpers

Provides read helpers that the orchestration layer uses at pipeline runtime
to fetch per-camera ROI geometry and enabled usecases from the database.

Usage:
    from shared.database.persistence import get_camera_rois, get_camera_usecases

    rois    = get_camera_rois("cam_01")
    usecases = get_camera_usecases("cam_01")
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from shared.database.connection import SessionLocal
from shared.database.models import ROIConfig, CameraUsecase, ChargingSession, Alert


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
        import logging
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
    The caller is responsible for falling back to defaults when the list
    is empty.
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
        import logging
        logging.getLogger(__name__).warning(
            f"[DB] Failed to fetch usecases for camera {camera_id}: {e}"
        )
        return []
    finally:
        db.close()


# Mapping from usecase_id → the YOLO class names that usecase operates on.
# Each usecase's confidence_threshold in the config column applies to all
# classes listed here. Derived from the rule implementations in the usecase service.
_USECASE_CLASSES: Dict[str, list] = {
    "gun_detection":      ["gun"],
    "safety_monitoring":  ["fire", "smoke"],
    "parking_detection":  ["car"],
    "parking_compliance": ["car"],
    "vehicle_extraction": ["car"],
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

    For each enabled usecase row, reads config.confidence_threshold and maps
    it to all YOLO class names that usecase operates on (via _USECASE_CLASSES).

    Example DB row:
        camera_id='camera_01', usecase_id='gun_detection', enabled=true,
        config={"confidence_threshold": 0.3}
        → produces {"gun_plugged_in": 0.3, "gun_plugged_out": 0.3}

        camera_id='camera_01', usecase_id='safety_monitoring', enabled=true,
        config={"confidence_threshold": 0.4}
        → produces {"fire": 0.4, "smoke": 0.4}

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
        import logging
        logging.getLogger(__name__).warning(
            f"[DB] Failed to fetch class_thresholds for camera {camera_id}: {e}"
        )
        return {}
    finally:
        db.close()


_persistence_logger = logging.getLogger(__name__)


def upsert_charging_session(camera_id: str, usecase_results: list) -> None:
    """
    Find or create an active ChargingSession for camera_id and update it with
    data extracted from the current pipeline iteration's usecase results.

    Session identity is (camera_id, slot_id) where slot_id = ROI name.
    This is a physical, stable identifier that never resets, unlike track_id
    which changes on tracker reset or Redis TTL expiry.

    Events are sourced from three usecases:
      - parking_detection : provides in_time / out_time, grouped by slot_id
      - gun_detection      : provides gun_number, plug_time, plug_out_time, grouped by slot_id
      - vehicle_extraction : provides car_number, car_model per track_id (enrichment only)

    Session lookup: (camera_id, slot_id) on open sessions — single stable key.
    car_number and car_model are enrichment fields, never used for session lookup.

    Session status transitions:
      - 'active'     : in_time set, no plug_time yet
      - 'charging'   : plug_time set, no plug_out_time yet
      - 'completed'  : plug_out_time AND out_time both set
      - 'incomplete' : out_time set but no plug_time

    A session is considered "open" when session_status is 'active' or 'charging'.
    Once out_time is written the status is finalised and the session is closed.
    A subsequent car arriving at the same slot creates a new session row.
    """
    def _parse_dt(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # 1. Extract events from usecase results, grouped by slot_id           #
    # ------------------------------------------------------------------ #
    # {slot_id: {"in_time": ts, "out_time": ts, "track_id": str}}
    parking_by_slot: dict = {}
    # {slot_id: {"plug_time": ts, "plug_out_time": ts, "gun_number": str, "track_id": str}}
    gun_by_slot: dict = {}
    # {slot_id: {"car_number": str, "car_model": str}} — primary enrichment key (from vehicle_extraction slot_id)
    vehicle_by_slot: dict = {}
    # {track_id: {"car_number": str, "car_model": str}} — fallback enrichment key
    vehicle_by_track: dict = {}

    for result in usecase_results:
        usecase_id = result.get("usecase_id") or result.get("usecase_name", "")
        extras = result.get("extras") or {}

        _event_usecases = {"parking_detection", "parking_compliance", "gun_detection"}
        has_events = bool(extras.get("events")) or bool(extras.get("vehicle_details"))
        if not result.get("triggered") and not (usecase_id in _event_usecases and has_events):
            continue

        if usecase_id in ("parking_detection", "parking_compliance"):
            for evt in extras.get("events", result.get("events", [])):
                etype = evt.get("event_type")
                ts = evt.get("timestamp")
                meta = evt.get("metadata", {})
                slot_id = meta.get("slot_id") or meta.get("roi")
                tid = evt.get("track_id")
                if etype not in ("parking_intime", "parking_outtime") or not slot_id:
                    continue
                if slot_id not in parking_by_slot:
                    parking_by_slot[slot_id] = {"in_time": None, "out_time": None, "track_id": None}
                if etype == "parking_intime" and parking_by_slot[slot_id]["in_time"] is None:
                    parking_by_slot[slot_id]["in_time"] = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid
                elif etype == "parking_outtime" and parking_by_slot[slot_id]["out_time"] is None:
                    parking_by_slot[slot_id]["out_time"] = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid

        elif usecase_id == "gun_detection":
            for evt in extras.get("events", result.get("events", [])):
                etype = evt.get("event_type")
                ts = evt.get("timestamp")
                meta = evt.get("metadata", {})
                slot_id = meta.get("slot_id") or meta.get("roi")
                tid = evt.get("track_id")
                if not slot_id:
                    continue
                if slot_id not in gun_by_slot:
                    gun_by_slot[slot_id] = {"plug_time": None, "plug_out_time": None, "gun_number": None, "track_id": None}
                if etype == "gun_plugin" and gun_by_slot[slot_id]["plug_time"] is None:
                    gun_by_slot[slot_id]["plug_time"] = ts
                    gun_by_slot[slot_id]["gun_number"] = gun_by_slot[slot_id]["gun_number"] or meta.get("gun_name")
                    gun_by_slot[slot_id]["track_id"] = gun_by_slot[slot_id]["track_id"] or tid
                elif etype == "gun_plugout" and gun_by_slot[slot_id]["plug_out_time"] is None:
                    gun_by_slot[slot_id]["plug_out_time"] = ts
                    gun_by_slot[slot_id]["gun_number"] = gun_by_slot[slot_id]["gun_number"] or meta.get("gun_name")
                    gun_by_slot[slot_id]["track_id"] = gun_by_slot[slot_id]["track_id"] or tid

        elif usecase_id == "vehicle_extraction":
            for d in extras.get("vehicle_details", result.get("vehicle_details", [])):
                tid = d.get("track_id")
                sid = d.get("slot_id")
                entry = {
                    "car_number": d.get("car_number"),
                    "car_model": d.get("car_model"),
                }
                # Index by slot_id (primary) and track_id (fallback) so the
                # upsert loop can always find enrichment data regardless of
                # whether the trackers in parking_detection and vehicle_extraction
                # assigned the same track_id to the car.
                if sid:
                    vehicle_by_slot[sid] = entry
                if tid:
                    vehicle_by_track[tid] = entry

    # ------------------------------------------------------------------ #
    # 2. Upsert one session per slot_id seen this frame                    #
    # ------------------------------------------------------------------ #
    all_slots = set(parking_by_slot.keys()) | set(gun_by_slot.keys())
    if not all_slots:
        return  # nothing actionable this frame

    for slot_id in all_slots:
        p = parking_by_slot.get(slot_id, {})
        g = gun_by_slot.get(slot_id, {})

        in_time = _parse_dt(p.get("in_time"))
        out_time = _parse_dt(p.get("out_time"))
        plug_time = _parse_dt(g.get("plug_time"))
        plug_out_time = _parse_dt(g.get("plug_out_time"))
        gun_number = g.get("gun_number")
        track_id = p.get("track_id") or g.get("track_id")

        # Enrich from vehicle_extraction: slot_id match is reliable (same spatial key);
        # track_id is a fallback for when vehicle_extraction predates the slot_id field.
        car_number = None
        car_model = None
        vd = vehicle_by_slot.get(slot_id) or (vehicle_by_track.get(track_id) if track_id else None)
        if vd:
            cn = vd.get("car_number")
            cm = vd.get("car_model")
            if cn not in (None, "unreadable", "unknown"):
                car_number = cn
            if cm not in (None, "unknown"):
                car_model = cm

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

            # Primary lookup: (camera_id, slot_id) on open sessions only
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

            # Issue #11: Redis restart guard — before creating a new session,
            # check if this slot had a session closed within the last 5 minutes.
            # A Redis restart resets tracker state, causing parking_intime to re-fire
            # for a car that is still physically parked. Re-opening the recent session
            # prevents a duplicate row for the same physical visit.
            if session is None and in_time is not None:
                restart_window = datetime.now(timezone.utc) - timedelta(minutes=5)
                recent = (
                    db.query(ChargingSession)
                    .filter(
                        ChargingSession.camera_id == camera_id,
                        ChargingSession.slot_id == slot_id,
                        ChargingSession.updated_at >= restart_window,
                    )
                    .order_by(ChargingSession.session_id.desc())
                    .first()
                )
                if recent and recent.session_status in ("active", "charging", "completed", "incomplete"):
                    session = recent
                    session.session_status = "active" if session.plug_time is None else "charging"
                    _persistence_logger.info(
                        f"[DB] Re-opened recent session session_id={session.session_id} "
                        f"slot={slot_id} after likely Redis restart for camera {camera_id}"
                    )

            # Create a new session only when in_time is present
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
                    session_status="active",
                )
                db.add(session)
                db.flush()
                _persistence_logger.info(
                    f"[DB] Created new ChargingSession session_id={session.session_id} "
                    f"slot={slot_id} track={track_id} for camera {camera_id}"
                )

            # ------------------------------------------------------------ #
            # 3. Apply updates                                               #
            # ------------------------------------------------------------ #
            if track_id and session.track_id is None:
                session.track_id = track_id
            if gun_number and session.gun_number is None:
                session.gun_number = gun_number
            if car_number and session.car_number is None:
                session.car_number = car_number
            if car_model and session.car_model is None:
                session.car_model = car_model
            if in_time and session.in_time is None:
                session.in_time = in_time
            if plug_time and session.plug_time is None:
                session.plug_time = plug_time
            if plug_out_time and session.plug_out_time is None:
                session.plug_out_time = plug_out_time
            if out_time and session.out_time is None:
                session.out_time = out_time

            # ------------------------------------------------------------ #
            # 4. Derive session status                                       #
            # ------------------------------------------------------------ #
            if session.out_time is not None:
                session.session_status = "completed" if session.plug_time is not None else "incomplete"
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


# Usecases that are analytics-only and should NOT generate alerts.
_NO_ALERT_USECASES = {"people_counter", "heatmap", "vehicle_extraction"}


def persist_alerts_from_results(camera_id: str, usecase_results: list) -> int:
    """
    Save Alert rows from triggered usecase results (except analytics-only rules).

    Called by the orchestration pipeline after every usecase evaluation so the
    dashboard can display alerts for parking_detection, gun_detection,
    parking_compliance, safety_monitoring, and any other triggered rule.

    Deduplication for event-driven usecases (parking_detection, gun_detection):
      - One alert per (camera_id, slot_id, usecase_name) per event type.
      - parking_intime and parking_outtime each produce at most one alert per slot.
      - gun_plugin and gun_plugout each produce at most one alert per slot.
      - Checks the DB before inserting — skips if an identical alert already exists.

    Special handling for parking_detection:
      - parking_intime / parking_outtime events → persisted as parking_detection alerts
      - multiple_cars_in_roi events → persisted as separate parking_compliance alerts
        so they surface correctly on the Parking Compliance dashboard section.

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

            # parking_detection and gun_detection fire exit events (outtime, plugout) on the
            # frame the object leaves — triggered=False at that point. Still need to persist.
            _event_usecases = {"parking_detection", "gun_detection"}
            has_events = bool(extras.get("events"))
            if not result.get("triggered") and not (usecase_id in _event_usecases and has_events):
                continue
            if usecase_id in _NO_ALERT_USECASES:
                _persistence_logger.info(f"[DB][{camera_id}] DEBUG: skipping {usecase_id} (in _NO_ALERT_USECASES)")
                continue

            _persistence_logger.info(f"[DB][{camera_id}] DEBUG: processing alert for usecase={usecase_id} | extras keys={list(extras.keys())}")
            snapshot_url = result.get("snapshot_url")

            # ── Special case: split parking_detection events by type ──────────
            # multiple_cars_in_roi violations must appear on the Parking
            # Compliance dashboard, so they are stored under usecase_name
            # "parking_compliance" with their own alert rows.
            if usecase_id == "parking_detection":
                events = extras.get("events", [])

                session_events = [e for e in events if e.get("event_type") in ("parking_intime", "parking_outtime")]
                violation_events = [e for e in events if e.get("event_type") == "multiple_cars_in_roi"]

                # One alert per (camera_id, slot_id, event_type)
                for evt in session_events:
                    slot_id = evt.get("metadata", {}).get("slot_id") or evt.get("metadata", {}).get("roi")
                    event_type = evt.get("event_type", "")
                    alert_type = f"parking_{event_type}"

                    # Deduplicate: skip if this slot already has this alert type
                    if slot_id:
                        exists = (
                            db.query(Alert.alert_id)
                            .filter(
                                Alert.camera_id == camera_id,
                                Alert.slot_id == slot_id,
                                Alert.usecase_name == "parking_detection",
                                Alert.alert_type == alert_type,
                            )
                            .first()
                        )
                        if exists:
                            _persistence_logger.debug(
                                f"[DB][{camera_id}] Skipping duplicate alert slot={slot_id} type={alert_type}"
                            )
                            continue

                    matched_count = result.get("matched_count", 0)
                    message = f"[parking_detection] {matched_count} object(s) detected | event: {event_type}"
                    alert = Alert(
                        camera_id=camera_id,
                        slot_id=slot_id,
                        usecase_name="parking_detection",
                        alert_type=alert_type,
                        message=message,
                        status="sent",
                        snapshot_url=snapshot_url,
                        extras={"events": [evt]},
                    )
                    db.add(alert)
                    written += 1

                # Persist each multiple_cars_in_roi as a parking_compliance violation
                for evt in violation_events:
                    meta = evt.get("metadata", {})
                    slot_id = meta.get("slot_id") or meta.get("roi")
                    roi = meta.get("roi", "unknown")
                    count = meta.get("count", 2)
                    message = f"[parking_compliance] {count} cars detected in {roi}"
                    alert = Alert(
                        camera_id=camera_id,
                        slot_id=slot_id,
                        usecase_name="parking_compliance",
                        alert_type="multiple_cars_in_roi",
                        message=message,
                        status="sent",
                        snapshot_url=snapshot_url,
                        extras={"violations": [evt]},
                    )
                    db.add(alert)
                    written += 1

                continue  # handled above — skip the generic path below

            # ── gun_detection: one alert per (slot_id, event_type) ────────────
            if usecase_id == "gun_detection":
                for evt in extras.get("events", []):
                    slot_id = evt.get("metadata", {}).get("slot_id") or evt.get("metadata", {}).get("roi")
                    event_type = evt.get("event_type", "")
                    alert_type = event_type  # gun_plugin / gun_plugout

                    if slot_id:
                        exists = (
                            db.query(Alert.alert_id)
                            .filter(
                                Alert.camera_id == camera_id,
                                Alert.slot_id == slot_id,
                                Alert.usecase_name == "gun_detection",
                                Alert.alert_type == alert_type,
                            )
                            .first()
                        )
                        if exists:
                            _persistence_logger.debug(
                                f"[DB][{camera_id}] Skipping duplicate alert slot={slot_id} type={alert_type}"
                            )
                            continue

                    meta = evt.get("metadata", {})
                    message = f"[gun_detection] {event_type} | gun={meta.get('gun_name')} slot={slot_id}"
                    alert = Alert(
                        camera_id=camera_id,
                        slot_id=slot_id,
                        usecase_name="gun_detection",
                        alert_type=alert_type,
                        message=message,
                        status="sent",
                        snapshot_url=snapshot_url,
                        extras={"events": [evt]},
                    )
                    db.add(alert)
                    written += 1

                continue  # handled above — skip the generic path below

            # ── Generic path for all other usecases ───────────────────────────
            matched_count = result.get("matched_count", 0)
            message = f"[{usecase_id}] {matched_count} object(s) detected"
            if "events" in extras and extras["events"]:
                event_types = list({e.get("event_type", "") for e in extras["events"]})
                message += f" | events: {', '.join(event_types)}"
            elif "violations" in extras and extras["violations"]:
                reasons = list({v.get("metadata", {}).get("reason", "") for v in extras["violations"]})
                message += f" | violations: {', '.join(r for r in reasons if r)}"

            alert = Alert(
                camera_id=camera_id,
                slot_id=None,
                usecase_name=usecase_id,
                alert_type=f"{usecase_id}_triggered",
                message=message,
                status="sent",
                snapshot_url=snapshot_url,
                extras=extras if extras else None,
            )
            db.add(alert)
            written += 1

        db.commit()
        _persistence_logger.debug(f"[DB] Persisted {written} alert(s) for camera {camera_id}")
    except Exception as e:
        db.rollback()
        _persistence_logger.warning(f"[DB] Failed to persist alerts for camera {camera_id}: {e}")
    finally:
        db.close()

    return written


# Issue #8: Stale session timeout
SESSION_STALE_HOURS = 4


def close_stale_sessions(stale_hours: int = SESSION_STALE_HOURS) -> int:
    """
    Close any ChargingSession that has been open (active or charging) for longer
    than stale_hours without receiving an out_time.

    This handles cameras going offline, model permanently losing a car, or service
    crashes that prevent parking_outtime from ever firing — without this, sessions
    stay open forever and block future cars at the same slot from getting a fresh session.

    Sessions closed here are marked 'incomplete' since there was no clean plug_out/out_time.

    Called periodically by the orchestration pipeline (e.g. every 60 iterations).
    Returns the number of sessions closed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=stale_hours)
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
            session.session_status = "incomplete"
            closed += 1
            _persistence_logger.warning(
                f"[DB] Closed stale session session_id={session.session_id} "
                f"slot={session.slot_id} camera={session.camera_id} "
                f"created_at={session.created_at} (stale > {stale_hours}h)"
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
