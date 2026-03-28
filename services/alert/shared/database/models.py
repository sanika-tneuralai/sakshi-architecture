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
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, UniqueConstraint, JSON, Text
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
    alert_type = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    status = Column(String(20), nullable=False)  # 'sent' or 'failed'
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    screenshot_path = Column(String(500), nullable=True)
    snapshot_b64 = Column(Text, nullable=True)        # base64 JPEG frame at alert time
    extras = Column(JSON, nullable=True)              # rule-specific data (events, violations, etc.)


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
