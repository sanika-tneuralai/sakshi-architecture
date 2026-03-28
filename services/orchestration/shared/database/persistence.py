"""
Orchestration DB helpers

Provides read helpers that the orchestration layer uses at pipeline runtime
to fetch per-camera ROI geometry and enabled usecases from the database.

Usage:
    from shared.database.persistence import get_camera_rois, get_camera_usecases

    rois    = get_camera_rois("cam_01")
    usecases = get_camera_usecases("cam_01")
"""
from typing import List, Dict, Any

from shared.database.connection import SessionLocal
from shared.database.models import ROIConfig, CameraUsecase


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
