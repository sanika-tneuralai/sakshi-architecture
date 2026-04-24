"""
Orchestration DB helpers

Provides helpers that the orchestration layer uses at pipeline runtime:
  - get_camera_rois / get_camera_usecases / get_class_thresholds : config reads
  - upsert_charging_session       : session create / update from usecase events
  - persist_alerts_from_results   : alert deduplication and write
  - close_stale_sessions          : periodic stale session cleanup
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from shared.database.connection import SessionLocal
from shared.database.models import ROIConfig, CameraUsecase, ChargingSession, Alert


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
    vehicle_by_slot: dict = {}    # {slot_id: {"car_number": str, "car_model": str}}
    vehicle_by_track: dict = {}   # {track_id: {"car_number": str, "car_model": str}}

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
                        parking_by_track[tid] = {"in_time": None, "out_time": None}
                    if etype == "parking_intime" and parking_by_track[tid]["in_time"] is None:
                        parking_by_track[tid]["in_time"] = ts
                    elif etype == "parking_outtime" and parking_by_track[tid]["out_time"] is None:
                        parking_by_track[tid]["out_time"] = ts
                    continue

                if slot_id not in parking_by_slot:
                    parking_by_slot[slot_id] = {"in_time": None, "out_time": None, "track_id": None}
                if etype == "parking_intime" and parking_by_slot[slot_id]["in_time"] is None:
                    parking_by_slot[slot_id]["in_time"]  = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid
                elif etype == "parking_outtime" and parking_by_slot[slot_id]["out_time"] is None:
                    parking_by_slot[slot_id]["out_time"] = ts
                    parking_by_slot[slot_id]["track_id"] = parking_by_slot[slot_id]["track_id"] or tid

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
                    "car_number": d.get("car_number"),
                    "car_model":  d.get("car_model"),
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
        gun_number    = g.get("gun_number")
        track_id      = p.get("track_id") or g.get("track_id")

        # Vehicle enrichment — slot_id is the only key (track_id fallback removed
        # because slot-anchored extraction always provides slot_id)
        car_number = None
        car_model  = None
        vd = vehicle_by_slot.get(slot_id)
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

            # ---- Status derivation ---- #
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
        vd = vehicle_by_track.get(track_id)
        if vd:
            cn = vd.get("car_number")
            cm = vd.get("car_model")
            if cn not in (None, "unreadable", "unknown"):
                car_number = cn
            if cm not in (None, "unknown"):
                car_model = cm

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

            if session.out_time is not None:
                # Unauthorized sessions never get plug_time ⇒ always 'incomplete' when closed.
                session.session_status = "incomplete"
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
    are marked 'completed' if plug_time exists, otherwise 'incomplete'. The
    inferred out_time lets downstream analytics (energy comparison) still
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
                # Prefer created_at + stale_hours over now() so the inferred window
                # stays anchored to the session's own timeline rather than wall clock.
                created = session.created_at
                if created is not None and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                session.out_time = (created + timedelta(hours=stale_hours)) if created else now
            session.session_status = "completed" if session.plug_time is not None else "incomplete"
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
