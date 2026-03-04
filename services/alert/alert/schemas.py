"""
Pydantic schemas for alert module.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class AlertRequest(BaseModel):
    """Alert-ready payload from Usecase API"""
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
    """Details of a single alert sent"""
    usecase_id: str
    alert_type: str
    alert_count: int
    message: str


class AlertResponse(BaseModel):
    """Response after processing alert"""
    camera_id: str
    alert_sent: bool
    alert_type: str
    alert_count: int
    message: str


class PipelineAlertResponse(BaseModel):
    """Response after processing multiple alerts"""
    camera_id: str
    total_alerts_sent: int
    alerts_sent: List[AlertDetail]


class AlertRecord(BaseModel):
    """Alert record from database"""
    alert_id: int
    camera_id: str
    usecase_name: str
    alert_type: str
    timestamp: str
    status: str
    screenshot_path: str = None


class AlertListResponse(BaseModel):
    """Response for alert list"""
    alerts: List[AlertRecord]
    total: int

