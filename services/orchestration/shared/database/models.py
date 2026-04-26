"""
Shared Database Models

This module contains all database models used across the GOEC services.
Each service can import only the models it needs.

Models:
    - Camera: Camera configuration and metadata
    - Detection: Object detection results
    - UsecaseResult: Use case evaluation results
    - Alert: Alert records
    - AnalyticsDaily: Daily aggregated analytics
    - ROIConfig: Per-camera ROI polygon definitions
    - CameraUsecase: Per-camera enabled usecase configuration
    - ChargingSession: EV charging session lifecycle records

Usage:
    from shared.database.models import Camera, ChargingSession
"""
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Date, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.sql import func
from shared.database.connection import Base


class Camera(Base):
    __tablename__ = "camera"

    camera_id = Column(String(255), primary_key=True)
    name = Column(String(255))
    location = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Detection(Base):
    __tablename__ = "detections"

    detection_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    object_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    inside_roi = Column(Boolean, nullable=False, default=False)
    screenshot_path = Column(String(500), nullable=True)


class UsecaseResult(Base):
    __tablename__ = "usecase_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    usecase_name = Column(String(100), nullable=False)
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    triggered = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Alert(Base):
    """
    Alert records for triggered usecases.

    slot_id is populated for parking_detection and gun_detection alerts
    (deduplication key). It is NULL for all other usecase alerts.
    """
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    slot_id = Column(String(100), nullable=True, index=True)
    usecase_name = Column(String(100), nullable=False)
    alert_type = Column(String(100), nullable=False)
    message = Column(String(500), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    status = Column(String(20), nullable=False, default='sent')
    snapshot_url = Column(String(512), nullable=True)
    extras = Column(JSON, nullable=True)


class AnalyticsDaily(Base):
    __tablename__ = "analytics_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    total_detections = Column(Integer, nullable=False, default=0)
    roi_violations = Column(Integer, nullable=False, default=0)
    alerts_sent = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint('date', 'camera_id', name='uix_date_camera'),
    )


class ROIConfig(Base):
    """
    Per-camera ROI polygon definitions.

    Each row defines one ROI polygon. The orchestration layer fetches these
    at pipeline runtime and forwards them to the usecase service.
    """
    __tablename__ = "roi_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    roi_id = Column(String(100), nullable=False)
    roi_type = Column(String(100), nullable=False)
    points = Column(JSON, nullable=False)       # [[x1,y1], [x2,y2], ...]
    label = Column(String(255), nullable=True)
    roi_metadata = Column(JSON, nullable=True, default=dict)

    __table_args__ = (
        UniqueConstraint('camera_id', 'roi_id', name='uix_camera_roi'),
    )


class CameraUsecase(Base):
    """
    Per-camera enabled usecase list with per-usecase config overrides.
    """
    __tablename__ = "camera_usecases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    usecase_id = Column(String(100), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    config = Column(JSON, nullable=True, default=dict)

    __table_args__ = (
        UniqueConstraint('camera_id', 'usecase_id', name='uix_camera_usecases_mapping'),
    )


class ChargingSession(Base):
    """
    EV charging session lifecycle record.

    Assembled from parking_detection, vehicle_extraction, and gun_detection
    usecase results during the orchestration pipeline.

    Session identity: UNIQUE(camera_id, slot_id, in_time).
      - camera_id + slot_id identify the physical charging bay.
      - in_time is the confirmed entry timestamp (millisecond precision).
      - Together they are guaranteed unique per physical visit.
    The unique constraint is enforced at the DB level so any duplicate insert
    (e.g. from a Redis-lost cold-start edge case) silently does nothing.

    Status transitions:
      active     — in_time set, no plug_time yet
      charging   — plug_time set, out_time not yet set
      completed  — plug_out_time AND out_time both set
      incomplete — out_time set but no plug_time
      discarded  — in_time + out_time both set, but duration is below
                   MIN_SESSION_MINUTES (default 20). Almost always tracker
                   fragmentation or a non-charging visit. Row is preserved
                   for audit but excluded from default dashboard listings,
                   analytics, and energy meter comparison.

    All fields except (camera_id, slot_id, session_status) are written once
    (first-write-wins). The upsert logic never overwrites a non-null field.
    """
    __tablename__ = "charging_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    slot_id = Column(String(100), nullable=True, index=True)
    track_id = Column(String(50), nullable=True, index=True)
    gun_number = Column(String(100), nullable=True)
    car_number = Column(String(100), nullable=True)
    car_model = Column(String(255), nullable=True)
    in_time = Column(DateTime(timezone=True), nullable=True)
    plug_time = Column(DateTime(timezone=True), nullable=True)
    plug_out_time = Column(DateTime(timezone=True), nullable=True)
    out_time = Column(DateTime(timezone=True), nullable=True)
    session_status = Column(String(50), nullable=False, default='active')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        # Unique session identity: one row per physical car visit per slot.
        # Prevents duplicate sessions from Redis-lost cold starts.
        # in_time has millisecond precision — collision probability is zero.
        UniqueConstraint('camera_id', 'slot_id', 'in_time', name='uix_session_identity'),
    )
