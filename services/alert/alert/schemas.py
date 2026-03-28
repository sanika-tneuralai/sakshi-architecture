"""
Pydantic schemas for alert module.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AlertRequest(BaseModel):
    """Alert-ready payload from Usecase API (legacy single-alert endpoint)"""
    camera_id: str = Field(..., description="Camera ID")
    usecase_id: str = Field(..., description="Usecase identifier")
    alert_required: bool = Field(..., description="Whether alert is required")
    alert_type: str = Field(..., description="Type of alert")
    alert_objects: List[dict] = Field(..., description="Objects that triggered the alert")
    alert_count: int = Field(..., description="Number of objects in alert")


class PipelineAlertRequest(BaseModel):
    """Alert request from orchestrator with multiple usecase results"""
    camera_id: str = Field(..., description="Camera ID")
    usecase_results: List[Dict[str, Any]] = Field(..., description="Results from usecase evaluation")


class AlertDetail(BaseModel):
    """Details of a single alert that was fired"""
    usecase_id: str                             # e.g. "parking_detection"
    alert_type: str                             # e.g. "parking_detection_triggered"
    alert_count: int                            # number of matched objects
    message: str                                # human-readable summary
    timestamp: str                              # ISO-8601 UTC when the alert fired
    snapshot_b64: Optional[str] = None         # base64 JPEG frame from detection
    extras: Optional[Dict[str, Any]] = None    # rule-specific data:
                                                #   parking_detection  → events (intime/outtime)
                                                #   gun_detection      → events (plugin/plugout)
                                                #   parking_compliance → violations
                                                #   people_counter     → occupancy details
                                                #   any future rule    → whatever it returns


class AlertResponse(BaseModel):
    """Response after processing a single alert (legacy endpoint)"""
    camera_id: str
    alert_sent: bool
    alert_type: str
    alert_count: int
    message: str


class PipelineAlertResponse(BaseModel):
    """Response after processing multiple alerts from a pipeline iteration"""
    camera_id: str
    total_alerts_sent: int
    alerts_sent: List[AlertDetail]


class AlertRecord(BaseModel):
    """Alert record returned from the /alert/list endpoint"""
    alert_id: int
    camera_id: str
    usecase_name: str
    alert_type: str
    timestamp: str
    status: str
    snapshot_b64: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None
    screenshot_path: Optional[str] = None


class AlertListResponse(BaseModel):
    """Response for the /alert/list endpoint"""
    alerts: List[AlertRecord]
    total: int
