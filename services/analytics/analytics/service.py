"""
Analytics aggregation service.
"""
from datetime import date, timedelta, datetime
from sqlalchemy import func, and_
from sqlalchemy.dialects.postgresql import insert
from shared.database.connection import SessionLocal
from shared.database.models import Detection, Alert, AnalyticsDaily


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
