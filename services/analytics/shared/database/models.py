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
    - ChargingSession: EV charging session lifecycle records

Usage:
    # Import all models
    from shared.database.models import Camera, Detection, UsecaseResult, Alert, AnalyticsDaily, ChargingSession

    # Or import specific models needed by your service
    from shared.database.models import Camera, Detection
"""
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Date, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from shared.database.connection import Base
from sqlalchemy import JSON


class Camera(Base):
    """
    Camera model for storing camera configuration and metadata.

    Attributes:
        camera_id (str): Unique camera identifier (primary key)
        name (str): Human-readable camera name
        location (str): Camera location description
        created_at (datetime): Timestamp when camera was added
    """
    __tablename__ = "camera"

    camera_id = Column(String(255), primary_key=True)
    name = Column(String(255))
    location = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Detection(Base):
    """
    Detection model for storing object detection results.

    Attributes:
        detection_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to camera table
        timestamp (datetime): When detection occurred
        object_type (str): Type of detected object (e.g., 'person', 'vehicle')
        confidence (float): Detection confidence score (0.0 to 1.0)
        inside_roi (bool): Whether object is inside region of interest
        screenshot_path (str): Optional path to detection screenshot
    """
    __tablename__ = "detections"

    detection_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
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
        camera_id (str): Foreign key to camera table
        usecase_name (str): Name of the evaluated use case
        detection_id (int): Optional foreign key to detections table
        triggered (bool): Whether the use case was triggered
        timestamp (datetime): When evaluation occurred
    """
    __tablename__ = "usecase_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    usecase_name = Column(String(100), nullable=False)
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    triggered = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Alert(Base):
    """
    Alert model for storing alert records.

    Attributes:
        alert_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to camera table
        usecase_name (str): Name of the use case that triggered alert
        alert_type (str): Type of alert (e.g., 'parking_detection_triggered')
        message (str): Human-readable alert message
        timestamp (datetime): When alert was triggered
        status (str): Alert status ('sent' or 'failed')
        detection_id (int): Optional foreign key to detections table
        screenshot_path (str): Optional path to alert screenshot
        snapshot_b64 (str): Base64-encoded JPEG frame captured at alert time
        extras (dict): Rule-specific data (events, violations, vehicle_details, etc.)
    """
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
    usecase_name = Column(String(100), nullable=False)
    alert_type = Column(String(100), nullable=False)
    message = Column(String(500), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    status = Column(String(20), nullable=False, default='sent')
    detection_id = Column(Integer, ForeignKey("detections.detection_id"), nullable=True)
    screenshot_path = Column(String(500), nullable=True)
    snapshot_b64 = Column(Text, nullable=True)
    extras = Column(JSON, nullable=True)


class AnalyticsDaily(Base):
    """
    AnalyticsDaily model for storing daily aggregated analytics.

    Attributes:
        id (int): Auto-incrementing primary key
        date (date): Date of the analytics record
        camera_id (str): Foreign key to camera table
        total_detections (int): Total number of detections for the day
        roi_violations (int): Number of ROI violations for the day
        alerts_sent (int): Number of alerts sent for the day

    Constraints:
        Unique constraint on (date, camera_id) combination
    """
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


class ChargingSession(Base):
    """
    ChargingSession model for storing EV charging session records.

    Captures the full lifecycle of a vehicle at an EV charging station,
    assembled from parking_detection, vehicle_extraction, and gun_detection
    usecase results.

    Attributes:
        session_id (int): Auto-incrementing primary key
        camera_id (str): Foreign key to camera table
        gun_number (str): Charging gun identifier (e.g. "Gun 1", "Gun 2")
        car_number (str): Vehicle license plate number
        car_model (str): Vehicle make/model
        in_time (datetime): When the car entered the parking ROI
        plug_time (datetime): When the charging gun was plugged in
        plug_out_time (datetime): When the charging gun was plugged out
        out_time (datetime): When the car exited the parking ROI
        session_status (str): 'active', 'charging', 'completed', 'incomplete'
        created_at (datetime): Record creation timestamp
        updated_at (datetime): Record last-update timestamp
    """
    __tablename__ = "charging_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(255), ForeignKey("camera.camera_id"), nullable=False, index=True)
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
