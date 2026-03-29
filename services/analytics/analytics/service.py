"""
Analytics aggregation service.
"""
from datetime import date, timedelta, datetime
from typing import Optional
from sqlalchemy import func, and_
from sqlalchemy.dialects.postgresql import insert
from shared.database.connection import SessionLocal
from shared.database.models import Detection, Alert, AnalyticsDaily, ChargingSession


def aggregate_daily_analytics(target_date: date = None):
    """
    Aggregate daily analytics for all cameras.
    
    Args:
        target_date: Date to aggregate (default: yesterday)
    """
    if target_date is None:
        target_date = (datetime.utcnow().date() - timedelta(days=1))
    
    print(f"[ANALYTICS AGGREGATION] Starting daily aggregation for {target_date}")
    
    db = SessionLocal()
    try:
        # Get all unique camera_ids from detections
        camera_ids = db.query(Detection.camera_id).filter(
            func.date(Detection.timestamp) == target_date
        ).distinct().all()
        
        camera_ids = [c[0] for c in camera_ids]
        print(f"[ANALYTICS AGGREGATION] Found {len(camera_ids)} cameras with activity")
        
        for camera_id in camera_ids:
            # Count total detections
            total_detections = db.query(func.count(Detection.detection_id)).filter(
                and_(
                    Detection.camera_id == camera_id,
                    func.date(Detection.timestamp) == target_date
                )
            ).scalar() or 0
            
            # Count ROI violations (inside_roi = True)
            roi_violations = db.query(func.count(Detection.detection_id)).filter(
                and_(
                    Detection.camera_id == camera_id,
                    func.date(Detection.timestamp) == target_date,
                    Detection.inside_roi == True
                )
            ).scalar() or 0
            
            # Count alerts sent
            alerts_sent = db.query(func.count(Alert.alert_id)).filter(
                and_(
                    Alert.camera_id == camera_id,
                    func.date(Alert.timestamp) == target_date,
                    Alert.status == 'sent'
                )
            ).scalar() or 0
            
            # Insert or update analytics_daily
            stmt = insert(AnalyticsDaily).values(
                date=target_date,
                camera_id=camera_id,
                total_detections=total_detections,
                roi_violations=roi_violations,
                alerts_sent=alerts_sent
            )
            stmt = stmt.on_conflict_do_update(
                constraint='uix_date_camera',
                set_=dict(
                    total_detections=total_detections,
                    roi_violations=roi_violations,
                    alerts_sent=alerts_sent
                )
            )
            db.execute(stmt)
            
            print(f"[ANALYTICS AGGREGATION] {camera_id}: detections={total_detections}, roi={roi_violations}, alerts={alerts_sent}")
        
        db.commit()
        print(f"[ANALYTICS AGGREGATION] Successfully aggregated data for {len(camera_ids)} cameras")
        
    except Exception as e:
        db.rollback()
        print(f"[ANALYTICS AGGREGATION] Error during aggregation: {str(e)}")
        raise
    finally:
        db.close()


def get_daily_analytics(camera_id: str = None, start_date: date = None, end_date: date = None):
    """
    Get daily analytics from analytics_daily table.
    
    Args:
        camera_id: Filter by camera (optional)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
    """
    print(f"[ANALYTICS QUERY] Fetching daily analytics")
    
    db = SessionLocal()
    try:
        query = db.query(AnalyticsDaily)
        
        if camera_id:
            query = query.filter(AnalyticsDaily.camera_id == camera_id)
            print(f"[ANALYTICS QUERY] Filtered by camera_id: {camera_id}")
        
        if start_date:
            query = query.filter(AnalyticsDaily.date >= start_date)
            print(f"[ANALYTICS QUERY] Filtered by start_date: {start_date}")
        
        if end_date:
            query = query.filter(AnalyticsDaily.date <= end_date)
            print(f"[ANALYTICS QUERY] Filtered by end_date: {end_date}")
        
        results = query.order_by(AnalyticsDaily.date.desc()).all()
        print(f"[ANALYTICS QUERY] Found {len(results)} records")
        
        return results
    finally:
        db.close()


def get_alert_analytics(camera_id: str = None, start_date: date = None, end_date: date = None):
    """
    Get alert analytics aggregated by camera and usecase.
    
    Args:
        camera_id: Filter by camera (optional)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
    """
    print(f"[ANALYTICS QUERY] Fetching alert analytics")
    
    db = SessionLocal()
    try:
        query = db.query(
            Alert.camera_id,
            Alert.usecase_name,
            func.count(Alert.alert_id).label('total_alerts'),
            func.sum(func.case((Alert.status == 'sent', 1), else_=0)).label('alerts_sent'),
            func.sum(func.case((Alert.status == 'failed', 1), else_=0)).label('alerts_failed')
        )
        
        if camera_id:
            query = query.filter(Alert.camera_id == camera_id)
            print(f"[ANALYTICS QUERY] Filtered by camera_id: {camera_id}")
        
        if start_date:
            query = query.filter(func.date(Alert.timestamp) >= start_date)
            print(f"[ANALYTICS QUERY] Filtered by start_date: {start_date}")
        
        if end_date:
            query = query.filter(func.date(Alert.timestamp) <= end_date)
            print(f"[ANALYTICS QUERY] Filtered by end_date: {end_date}")
        
        results = query.group_by(Alert.camera_id, Alert.usecase_name).all()
        print(f"[ANALYTICS QUERY] Found {len(results)} alert records")
        
        return results
    finally:
        db.close()


def get_detection_analytics(camera_id: str = None, start_date: date = None, end_date: date = None):
    """
    Get detection analytics aggregated by camera.
    
    Args:
        camera_id: Filter by camera (optional)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
    """
    print(f"[ANALYTICS QUERY] Fetching detection analytics")
    
    db = SessionLocal()
    try:
        query = db.query(
            Detection.camera_id,
            func.count(Detection.detection_id).label('total_detections'),
            func.sum(func.case((Detection.inside_roi == True, 1), else_=0)).label('roi_detections'),
            func.sum(func.case((Detection.inside_roi == False, 1), else_=0)).label('non_roi_detections')
        )
        
        if camera_id:
            query = query.filter(Detection.camera_id == camera_id)
            print(f"[ANALYTICS QUERY] Filtered by camera_id: {camera_id}")
        
        if start_date:
            query = query.filter(func.date(Detection.timestamp) >= start_date)
            print(f"[ANALYTICS QUERY] Filtered by start_date: {start_date}")
        
        if end_date:
            query = query.filter(func.date(Detection.timestamp) <= end_date)
            print(f"[ANALYTICS QUERY] Filtered by end_date: {end_date}")
        
        results = query.group_by(Detection.camera_id).all()
        print(f"[ANALYTICS QUERY] Found {len(results)} detection records")

        return results
    finally:
        db.close()


# ── Charging Session Analytics ────────────────────────────────────────────────

def _duration_minutes(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    """Return duration in minutes between two timestamps, or None if either is missing."""
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    return round(delta / 60, 2) if delta >= 0 else None


def get_charging_sessions(
    camera_id: Optional[str] = None,
    gun_number: Optional[str] = None,
    car_model: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
):
    """
    Return raw charging session records with derived duration fields.

    Filters:
        camera_id  – restrict to one camera
        gun_number – restrict to one charging gun
        car_model  – restrict to one car model
        start_date – sessions whose in_time >= start_date
        end_date   – sessions whose in_time <= end_date
        status     – 'active', 'charging', 'completed', 'incomplete'
    """
    print("[ANALYTICS QUERY] Fetching charging sessions")
    db = SessionLocal()
    try:
        query = db.query(ChargingSession)

        if camera_id:
            query = query.filter(ChargingSession.camera_id == camera_id)
        if gun_number:
            query = query.filter(ChargingSession.gun_number == gun_number)
        if car_model:
            query = query.filter(ChargingSession.car_model == car_model)
        if status:
            query = query.filter(ChargingSession.session_status == status)
        if start_date:
            query = query.filter(func.date(ChargingSession.in_time) >= start_date)
        if end_date:
            query = query.filter(func.date(ChargingSession.in_time) <= end_date)

        sessions = query.order_by(ChargingSession.in_time.desc()).all()
        print(f"[ANALYTICS QUERY] Found {len(sessions)} charging session records")
        return sessions
    finally:
        db.close()


def get_charging_pattern_analysis(
    camera_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """
    Comprehensive charging pattern and loss analysis segmented by car model and gun.

    A "loss session" is one where the car left without ever being plugged in
    (session_status = 'incomplete').

    Returns a dict with keys:
        by_car_model  – list of per-model aggregates
        by_gun        – list of per-gun aggregates
        total_*       – overall summary counters
    """
    print("[ANALYTICS QUERY] Running charging pattern analysis")
    db = SessionLocal()
    try:
        query = db.query(ChargingSession)
        if camera_id:
            query = query.filter(ChargingSession.camera_id == camera_id)
        if start_date:
            query = query.filter(func.date(ChargingSession.in_time) >= start_date)
        if end_date:
            query = query.filter(func.date(ChargingSession.in_time) <= end_date)

        sessions = query.all()

        # ── Per car-model aggregation ─────────────────────────────────────────
        model_buckets: dict = {}
        for s in sessions:
            key = s.car_model or "unknown"
            if key not in model_buckets:
                model_buckets[key] = []
            model_buckets[key].append(s)

        by_car_model = []
        for model_name, group in model_buckets.items():
            completed = [s for s in group if s.session_status == "completed"]
            losses = [s for s in group if s.session_status == "incomplete"]

            charging_durations = [
                _duration_minutes(s.plug_time, s.plug_out_time)
                for s in completed
                if _duration_minutes(s.plug_time, s.plug_out_time) is not None
            ]
            wait_before = [
                _duration_minutes(s.in_time, s.plug_time)
                for s in completed
                if _duration_minutes(s.in_time, s.plug_time) is not None
            ]
            wait_after = [
                _duration_minutes(s.plug_out_time, s.out_time)
                for s in completed
                if _duration_minutes(s.plug_out_time, s.out_time) is not None
            ]
            total_dur = [
                _duration_minutes(s.in_time, s.out_time)
                for s in group
                if _duration_minutes(s.in_time, s.out_time) is not None
            ]

            by_car_model.append({
                "car_model": model_name,
                "total_sessions": len(group),
                "completed_sessions": len(completed),
                "avg_charging_duration_minutes": round(sum(charging_durations) / len(charging_durations), 2) if charging_durations else None,
                "avg_wait_before_plug_minutes": round(sum(wait_before) / len(wait_before), 2) if wait_before else None,
                "avg_wait_after_plug_out_minutes": round(sum(wait_after) / len(wait_after), 2) if wait_after else None,
                "avg_total_session_minutes": round(sum(total_dur) / len(total_dur), 2) if total_dur else None,
                "loss_sessions": len(losses),
                "loss_rate_pct": round(len(losses) / len(group) * 100, 2) if group else 0.0,
            })

        # ── Per gun aggregation ───────────────────────────────────────────────
        gun_buckets: dict = {}
        for s in sessions:
            key = (s.camera_id, s.gun_number or "unknown")
            if key not in gun_buckets:
                gun_buckets[key] = []
            gun_buckets[key].append(s)

        # Compute the date range span in minutes to derive utilization
        if start_date and end_date:
            range_minutes = (end_date - start_date).days * 24 * 60 + 24 * 60
        else:
            # Estimate from actual session data
            all_in = [s.in_time for s in sessions if s.in_time]
            all_out = [s.out_time for s in sessions if s.out_time]
            if all_in and all_out:
                range_minutes = (max(all_out) - min(all_in)).total_seconds() / 60
            else:
                range_minutes = None

        by_gun = []
        for (cam_id, gun_name), group in gun_buckets.items():
            completed = [s for s in group if s.session_status == "completed"]
            losses = [s for s in group if s.session_status == "incomplete"]

            charging_durations = [
                _duration_minutes(s.plug_time, s.plug_out_time)
                for s in completed
                if _duration_minutes(s.plug_time, s.plug_out_time) is not None
            ]
            wait_before = [
                _duration_minutes(s.in_time, s.plug_time)
                for s in completed
                if _duration_minutes(s.in_time, s.plug_time) is not None
            ]
            wait_after = [
                _duration_minutes(s.plug_out_time, s.out_time)
                for s in completed
                if _duration_minutes(s.plug_out_time, s.out_time) is not None
            ]
            total_dur = [
                _duration_minutes(s.in_time, s.out_time)
                for s in group
                if _duration_minutes(s.in_time, s.out_time) is not None
            ]

            total_charging_minutes = sum(charging_durations) if charging_durations else 0
            utilization = (
                round(total_charging_minutes / range_minutes * 100, 2)
                if range_minutes and range_minutes > 0
                else 0.0
            )

            by_gun.append({
                "gun_number": gun_name,
                "camera_id": cam_id,
                "total_sessions": len(group),
                "completed_sessions": len(completed),
                "avg_charging_duration_minutes": round(sum(charging_durations) / len(charging_durations), 2) if charging_durations else None,
                "avg_wait_before_plug_minutes": round(sum(wait_before) / len(wait_before), 2) if wait_before else None,
                "avg_wait_after_plug_out_minutes": round(sum(wait_after) / len(wait_after), 2) if wait_after else None,
                "avg_total_session_minutes": round(sum(total_dur) / len(total_dur), 2) if total_dur else None,
                "loss_sessions": len(losses),
                "loss_rate_pct": round(len(losses) / len(group) * 100, 2) if group else 0.0,
                "utilization_rate_pct": utilization,
            })

        # ── Overall summary ───────────────────────────────────────────────────
        total_completed = len([s for s in sessions if s.session_status == "completed"])
        total_loss = len([s for s in sessions if s.session_status == "incomplete"])

        print(f"[ANALYTICS QUERY] Pattern analysis complete: {len(sessions)} sessions, "
              f"{len(by_car_model)} models, {len(by_gun)} guns")

        return {
            "by_car_model": sorted(by_car_model, key=lambda x: x["total_sessions"], reverse=True),
            "by_gun": sorted(by_gun, key=lambda x: (x["camera_id"], x["gun_number"])),
            "total_sessions": len(sessions),
            "total_completed": total_completed,
            "total_loss": total_loss,
            "overall_loss_rate_pct": round(total_loss / len(sessions) * 100, 2) if sessions else 0.0,
        }
    finally:
        db.close()
