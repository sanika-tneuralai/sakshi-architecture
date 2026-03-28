"""
Database persistence helpers.
"""
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from shared.database.connection import SessionLocal
from shared.database.models import Camera, Detection, UsecaseResult, Alert


def persist_camera(camera_id: str, name: str = None, location: str = None):
    """
    Persist camera information (insert if not exists).
    
    Args:
        camera_id: Camera identifier
        name: Camera name (optional)
        location: Camera location (optional)
    """
    db = SessionLocal()
    try:
        stmt = insert(Camera).values(
            camera_id=camera_id,
            name=name or camera_id,
            location=location or "Unknown"
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=['camera_id'])
        db.execute(stmt)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] Error persisting camera: {str(e)}")
    finally:
        db.close()


def persist_detection(camera_id: str, object_type: str, confidence: float, inside_roi: bool, screenshot_path: str = None):
    """
    Persist detection to database.
    
    Args:
        camera_id: Camera identifier
        object_type: Detected object class name
        confidence: Detection confidence score
        inside_roi: Whether detection is inside ROI
        screenshot_path: Path to detection screenshot (optional)
        
    Returns:
        detection_id if successful, None otherwise
    """
    db = SessionLocal()
    try:
        detection = Detection(
            camera_id=camera_id,
            object_type=object_type,
            confidence=confidence,
            inside_roi=inside_roi,
            screenshot_path=screenshot_path
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)
        return detection.detection_id
    except Exception as e:
        db.rollback()
        print(f"[DB] Error persisting detection: {str(e)}")
        return None
    finally:
        db.close()


def persist_usecase_result(camera_id: str, usecase_name: str, triggered: bool, detection_id: int = None):
    """
    Persist usecase evaluation result.
    
    Args:
        camera_id: Camera identifier
        usecase_name: Usecase name
        triggered: Whether usecase was triggered
        detection_id: Associated detection ID (optional)
    """
    db = SessionLocal()
    try:
        result = UsecaseResult(
            camera_id=camera_id,
            usecase_name=usecase_name,
            detection_id=detection_id,
            triggered=triggered
        )
        db.add(result)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] Error persisting usecase result: {str(e)}")
    finally:
        db.close()


def persist_alert(
    camera_id: str,
    usecase_name: str,
    alert_type: str,
    status: str = 'sent',
    detection_id: int = None,
    screenshot_path: str = None,
    snapshot_b64: str = None,
    extras: dict = None,
):
    """
    Persist alert to database.

    Args:
        camera_id: Camera identifier
        usecase_name: Usecase name that triggered alert
        alert_type: Type of alert
        status: Alert status ('sent' or 'failed')
        detection_id: ID of associated detection (optional)
        screenshot_path: Path to detection screenshot (optional)
        snapshot_b64: Base64 JPEG frame captured at alert time (optional)
        extras: Rule-specific data dict — events, violations, vehicle_details, etc. (optional)
    """
    db = SessionLocal()
    try:
        alert = Alert(
            camera_id=camera_id,
            usecase_name=usecase_name,
            alert_type=alert_type,
            status=status,
            detection_id=detection_id,
            screenshot_path=screenshot_path,
            snapshot_b64=snapshot_b64,
            extras=extras,
        )
        db.add(alert)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] Error persisting alert: {str(e)}")
    finally:
        db.close()
