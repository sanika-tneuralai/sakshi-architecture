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

Usage:
    # Import all models
    from shared.database.models import Camera, Detection, UsecaseResult, Alert, AnalyticsDaily
    
    # Or import specific models needed by your service
    from shared.database.models import Camera, Detection
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.sql import func
from shared.database.connection import Base


class Camera(Base):
    """
    Camera model for storing camera configuration and metadata.
    
    Attributes:
        camera_id (str): Unique camera identifier (primary key)
        name (str): Human-readable camera name
        location (str): Camera location description
        created_at (datetime): Timestamp when camera was added
    """
    __tablename__ = "cameras"
    
    camera_id = Column(String(255), primary_key=True)
    name = Column(String(255))
    location = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Detection(Base):
    """
    Detection model for storing object detection results.
    
    Attributes:
        detection_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to cameras table
        timestamp (datetime): When detection occurred
        object_type (str): Type of detected object (e.g., 'person', 'vehicle')
        confidence (float): Detection confidence score (0.0 to 1.0)
        inside_roi (bool): Whether object is inside region of interest
        screenshot_path (str): Optional path to detection screenshot
    """
    __tablename__ = "detections"
    
    detection_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    object_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    inside_roi = Column(Boolean, nullable=False, default=False)
    screenshot_path = Column(String(500), nullable=True)


class UsecaseResult(Base):
    """
    UsecaseResult model for storing use case evaluation results.
    
    Attributes:
        result_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to cameras table
        usecase_name (str): Name of the evaluated use case
        detection_id (int): Optional foreign key to detections table
        triggered (bool): Whether the use case was triggered
        timestamp (datetime): When evaluation occurred
    """
    __tablename__ = "usecase_results"
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    usecase_name = Column(String(100), nullable=False)
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    triggered = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Alert(Base):
    """
    Alert model for storing alert records.
    
    Attributes:
        alert_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to cameras table
        usecase_name (str): Name of the use case that triggered alert
        alert_type (str): Type of alert (e.g., 'email', 'sms', 'webhook')
        timestamp (datetime): When alert was triggered
        status (str): Alert status ('sent' or 'failed')
        detection_id (int): Optional foreign key to detections table
        screenshot_path (str): Optional path to alert screenshot
    """
    __tablename__ = "alerts"
    
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    usecase_name = Column(String(100), nullable=False)
    alert_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    status = Column(String(20), nullable=False)  # 'sent' or 'failed'
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    screenshot_path = Column(String(500), nullable=True)


class AnalyticsDaily(Base):
    """
    AnalyticsDaily model for storing daily aggregated analytics.
    
    Attributes:
        id (int): Auto-incrementing primary key
        date (date): Date of the analytics record
        camera_id (str): Foreign key to cameras table
        total_detections (int): Total number of detections for the day
        roi_violations (int): Number of ROI violations for the day
        alerts_sent (int): Number of alerts sent for the day
    
    Constraints:
        Unique constraint on (date, camera_id) combination
    """
    __tablename__ = "analytics_daily"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    total_detections = Column(Integer, nullable=False, default=0)
    roi_violations = Column(Integer, nullable=False, default=0)
    alerts_sent = Column(Integer, nullable=False, default=0)
    
    __table_args__ = (
        UniqueConstraint('date', 'camera_id', name='uix_date_camera'),
    )


class ROIConfig(Base):
    """
    ROIConfig model for storing per-camera region-of-interest definitions.

    Each row defines one ROI polygon for a camera. The orchestration layer
    fetches these at runtime and forwards them to the usecase service so that
    rules such as parking_detection, restricted_area, and people_counter can
    evaluate detections against the correct geometry.

    Attributes:
        id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to cameras table
        roi_id (str): Logical identifier used by rules (e.g. "roi_1")
        roi_type (str): Rule-domain tag (e.g. "parking_zone", "restricted_zone",
                        "counting_zone", "safety_zone")
        points (list): JSON array of [x, y] coordinate pairs defining the polygon
        label (str): Human-readable name shown in dashboards / logs
        roi_metadata (dict): Rule-specific configuration stored as JSON
                             e.g. {"max_occupancy": 5} for people_counter
                                  {"allowed_hours": "08:00-20:00"} for parking_compliance
    """
    __tablename__ = "roi_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    roi_id = Column(String(100), nullable=False)
    roi_type = Column(String(100), nullable=False)
    points = Column(JSON, nullable=False)           # [[x1,y1], [x2,y2], ...]
    label = Column(String(255), nullable=True)
    roi_metadata = Column(JSON, nullable=True, default=dict)

    __table_args__ = (
        UniqueConstraint('camera_id', 'roi_id', name='uix_camera_roi'),
    )


class CameraUsecase(Base):
    """
    CameraUsecase model for storing which usecases are enabled per camera
    and any per-usecase configuration overrides.

    The orchestration layer queries this table to build the `usecases` list
    that is sent to the usecase evaluation service, replacing the hard-coded
    default list in CameraConfig.

    Attributes:
        id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to cameras table
        usecase_id (str): Usecase rule identifier (e.g. "parking_detection")
        enabled (bool): Whether this usecase should run for this camera
        config (dict): Per-usecase JSON config overrides
                       e.g. {"confidence_threshold": 0.6, "max_occupancy": 10}
    """
    __tablename__ = "camera_usecases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    usecase_id = Column(String(100), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    config = Column(JSON, nullable=True, default=dict)

    __table_args__ = (
        UniqueConstraint('camera_id', 'usecase_id', name='uix_camera_usecases_mapping'),
    )
