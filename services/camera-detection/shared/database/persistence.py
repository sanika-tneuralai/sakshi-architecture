"""
Database persistence helpers.
"""
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from shared.database.connection import SessionLocal
from shared.database.models import Camera, Detection


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


def persist_detection(camera_id: str, object_type: str, confidence: float, screenshot_path: str = None):
    """
    Persist detection to database.

    Args:
        camera_id: Camera identifier
        object_type: Detected object class name
        confidence: Detection confidence score
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


