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
from datetime import datetime
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

    Events are sourced from three usecases:
      - parking_detection : provides in_time / out_time via extras
      - gun_detection      : provides gun_number, plug_time, plug_out_time via extras
      - vehicle_extraction : provides car_number, car_model via extras

    Session lookup priority:
      1. Match on camera_id + gun_number (when gun_detection fired)
      2. Match on camera_id + car_number (when vehicle_extraction fired)
      3. Most-recent open session for camera_id

    Session status transitions:
      - 'active'     : in_time set, no plug_time yet
      - 'charging'   : plug_time set, no plug_out_time yet
      - 'completed'  : plug_out_time AND out_time both set
      - 'incomplete' : out_time set but no plug_time

    A session is considered "open" when session_status is 'active' or 'charging'.
    Once out_time is written the status is finalised and the session is closed.
    """
    # ------------------------------------------------------------------ #
    # 1. Extract events from usecase results                               #
    #                                                                      #
    # Each usecase returns its data at the TOP LEVEL of the result dict    #
    # (not nested under "extras"). Relevant structures:                    #
    #                                                                      #
    # parking_detection:                                                   #
    #   result["events"] = [                                               #
    #     {"event_type": "parking_intime",  "timestamp": "...", ...},      #
    #     {"event_type": "parking_outtime", "timestamp": "...", ...},      #
    #   ]                                                                  #
    #                                                                      #
    # gun_detection:                                                       #
    #   result["events"] = [                                               #
    #     {"event_type": "gun_plugin",  "timestamp": "...",                #
    #      "metadata": {"gun_name": "Gun 1", ...}},                        #
    #     {"event_type": "gun_plugout", "timestamp": "...",                #
    #      "metadata": {"gun_name": "Gun 1", ...}},                        #
    #   ]                                                                  #
    #                                                                      #
    # vehicle_extraction:                                                  #
    #   result["vehicle_details"] = [                                      #
    #     {"car_number": "MH12AB1234", "car_model": "Toyota Innova", ...}  #
    #   ]                                                                  #
    # ------------------------------------------------------------------ #
    in_time_raw = None
    out_time_raw = None
    plug_time_raw = None
    plug_out_time_raw = None
    gun_number = None
    car_number = None
    car_model = None

    for result in usecase_results:
        usecase_id = result.get("usecase_id") or result.get("usecase_name", "")
        if not result.get("triggered"):
            continue

        # Rule-specific fields are nested under "extras" by the usecase service engine.
        # Fall back to top-level for backwards compatibility.
        extras = result.get("extras") or {}

        if usecase_id in ("parking_detection", "parking_compliance"):
            # parking_detection fires intime/outtime for cars inside ROIs.
            # parking_compliance also fires intime/outtime for unauthorized
            # cars (outside all ROIs) so their sessions are tracked too.
            for evt in extras.get("events", result.get("events", [])):
                etype = evt.get("event_type")
                ts = evt.get("timestamp")
                if etype == "parking_intime" and in_time_raw is None:
                    in_time_raw = ts
                elif etype == "parking_outtime" and out_time_raw is None:
                    out_time_raw = ts

        elif usecase_id == "gun_detection":
            for evt in extras.get("events", result.get("events", [])):
                etype = evt.get("event_type")
                ts = evt.get("timestamp")
                meta = evt.get("metadata", {})
                if etype == "gun_plugin" and plug_time_raw is None:
                    plug_time_raw = ts
                    gun_number = gun_number or meta.get("gun_name")
                elif etype == "gun_plugout" and plug_out_time_raw is None:
                    plug_out_time_raw = ts
                    gun_number = gun_number or meta.get("gun_name")

        elif usecase_id == "vehicle_extraction":
            details = extras.get("vehicle_details", result.get("vehicle_details", []))
            if details:
                # Only use a readable car_number — skip "unreadable"/"unknown" fallbacks
                # so they don't pollute session lookup keys or create junk sessions.
                readable = next(
                    (d for d in details if d.get("car_number") not in (None, "unreadable")),
                    None,
                )
                if readable:
                    car_number = car_number or readable.get("car_number")
                    car_model = car_model or readable.get("car_model")
                elif not car_model:
                    # Accept model even if plate is unreadable, but only if model is known
                    first = details[0]
                    if first.get("car_model") not in (None, "unknown"):
                        car_model = first.get("car_model")

    def _parse_dt(value) -> datetime | None:
        """Convert a string or datetime to a timezone-naive datetime (or None)."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    in_time = _parse_dt(in_time_raw)
    out_time = _parse_dt(out_time_raw)
    plug_time = _parse_dt(plug_time_raw)
    plug_out_time = _parse_dt(plug_out_time_raw)

    # Nothing actionable in this iteration — skip DB work entirely.
    if not any([in_time, out_time, plug_time, plug_out_time, gun_number, car_number, car_model]):
        return

    # ------------------------------------------------------------------ #
    # 2. Find or create an open session                                    #
    # ------------------------------------------------------------------ #
    db = SessionLocal()
    try:
        open_statuses = ("active", "charging")
        session: ChargingSession | None = None

        # Priority 1: match by gun_number
        if gun_number:
            session = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.gun_number == gun_number,
                    ChargingSession.session_status.in_(open_statuses),
                )
                .order_by(ChargingSession.session_id.desc())
                .first()
            )

        # Priority 2: match by car_number (only when readable)
        if session is None and car_number:
            session = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.car_number == car_number,
                    ChargingSession.session_status.in_(open_statuses),
                )
                .order_by(ChargingSession.session_id.desc())
                .first()
            )

        # Priority 3: most-recent open session for this camera
        if session is None:
            session = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.session_status.in_(open_statuses),
                )
                .order_by(ChargingSession.session_id.desc())
                .first()
            )

        # Create a new session only when in_time is present.
        # vehicle_extraction alone (no parking event) must never create a
        # new session — it should only enrich an already-open session.
        if session is None:
            if in_time is None:
                _persistence_logger.debug(
                    f"[DB] Skipping new session for camera {camera_id} — "
                    f"no in_time yet (vehicle_extraction only)"
                )
                return
            session = ChargingSession(
                camera_id=camera_id,
                session_status="active",
            )
            db.add(session)
            db.flush()  # Populate session_id before we update fields below
            _persistence_logger.info(
                f"[DB] Created new ChargingSession session_id={session.session_id} "
                f"for camera {camera_id}"
            )

        # ---------------------------------------------------------------- #
        # 3. Apply updates                                                  #
        # ---------------------------------------------------------------- #
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

        # ---------------------------------------------------------------- #
        # 4. Derive session status                                          #
        # ---------------------------------------------------------------- #
        if session.out_time is not None:
            # Session is over — finalise
            if session.plug_time is not None:
                session.session_status = "completed"
            else:
                session.session_status = "incomplete"
        elif session.plug_time is not None:
            session.session_status = "charging"
        else:
            session.session_status = "active"

        db.commit()
        _persistence_logger.info(
            f"[DB] ChargingSession session_id={session.session_id} "
            f"status={session.session_status} | in_time={session.in_time} "
            f"plug_time={session.plug_time} out_time={session.out_time} "
            f"car={session.car_number} gun={session.gun_number} camera={camera_id}"
        )

    except Exception as e:
        db.rollback()
        _persistence_logger.warning(
            f"[DB] Failed to upsert ChargingSession for camera {camera_id}: {e}"
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

    Special handling for parking_detection:
      - parking_intime / parking_outtime events → persisted as parking_detection alerts
      - multiple_cars_in_roi events → persisted as separate parking_compliance alerts
        so they surface correctly on the Parking Compliance dashboard section.

    Returns the number of alert rows written.
    """
    db = SessionLocal()
    written = 0
    try:
        for result in usecase_results:
            usecase_id = result.get("usecase_id") or result.get("usecase_name", "unknown")

            if not result.get("triggered"):
                continue
            if usecase_id in _NO_ALERT_USECASES:
                continue

            extras = result.get("extras") or {}
            snapshot_b64 = result.get("snapshot_b64")

            # ── Special case: split parking_detection events by type ──────────
            # multiple_cars_in_roi violations must appear on the Parking
            # Compliance dashboard, so they are stored under usecase_name
            # "parking_compliance" with their own alert rows.
            if usecase_id == "parking_detection":
                events = extras.get("events", [])

                session_events = [e for e in events if e.get("event_type") in ("parking_intime", "parking_outtime")]
                violation_events = [e for e in events if e.get("event_type") == "multiple_cars_in_roi"]

                # Persist session events as parking_detection alert
                if session_events:
                    matched_count = result.get("matched_count", 0)
                    event_types = list({e.get("event_type", "") for e in session_events})
                    message = f"[parking_detection] {matched_count} object(s) detected | events: {', '.join(event_types)}"
                    alert = Alert(
                        camera_id=camera_id,
                        usecase_name="parking_detection",
                        alert_type="parking_detection_triggered",
                        message=message,
                        status="sent",
                        snapshot_b64=snapshot_b64,
                        extras={"events": session_events},
                    )
                    db.add(alert)
                    written += 1

                # Persist each multiple_cars_in_roi as a parking_compliance violation
                for evt in violation_events:
                    meta = evt.get("metadata", {})
                    roi = meta.get("roi", "unknown")
                    count = meta.get("count", 2)
                    message = f"[parking_compliance] {count} cars detected in {roi}"
                    alert = Alert(
                        camera_id=camera_id,
                        usecase_name="parking_compliance",
                        alert_type="multiple_cars_in_roi",
                        message=message,
                        status="sent",
                        snapshot_b64=snapshot_b64,
                        extras={"violations": [evt]},
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
                usecase_name=usecase_id,
                alert_type=f"{usecase_id}_triggered",
                message=message,
                status="sent",
                snapshot_b64=snapshot_b64,
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
