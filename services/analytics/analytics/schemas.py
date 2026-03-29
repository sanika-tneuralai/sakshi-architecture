"""
Analytics Pydantic schemas.
"""
from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, List


# Analytics Daily Response
class DailyAnalytics(BaseModel):
    date: date
    camera_id: str
    total_detections: int
    roi_violations: int
    alerts_sent: int


class DailyAnalyticsResponse(BaseModel):
    data: list[DailyAnalytics]
    total_records: int


# Alert Analytics Response
class AlertAnalytics(BaseModel):
    camera_id: str
    usecase_name: str
    total_alerts: int
    alerts_sent: int
    alerts_failed: int


class AlertAnalyticsResponse(BaseModel):
    data: list[AlertAnalytics]
    total_records: int


# Detection Analytics Response
class DetectionAnalytics(BaseModel):
    camera_id: str
    total_detections: int
    roi_detections: int
    non_roi_detections: int
    roi_violation_rate: float


class DetectionAnalyticsResponse(BaseModel):
    data: list[DetectionAnalytics]
    total_records: int


# ── Charging Pattern & Loss Analysis ─────────────────────────────────────────

class ChargingSessionRecord(BaseModel):
    """Single completed or in-progress charging session."""
    session_id: int
    camera_id: str
    gun_number: Optional[str]
    car_number: Optional[str]
    car_model: Optional[str]
    in_time: Optional[datetime]
    plug_time: Optional[datetime]
    plug_out_time: Optional[datetime]
    out_time: Optional[datetime]
    session_status: str
    # Derived durations (minutes)
    wait_before_plug_minutes: Optional[float]   # in_time → plug_time
    charging_duration_minutes: Optional[float]  # plug_time → plug_out_time
    wait_after_plug_out_minutes: Optional[float] # plug_out_time → out_time
    total_session_minutes: Optional[float]       # in_time → out_time


class ChargingSessionResponse(BaseModel):
    data: List[ChargingSessionRecord]
    total_records: int


class ChargingPatternByModel(BaseModel):
    """Charging pattern metrics aggregated per car model."""
    car_model: str
    total_sessions: int
    completed_sessions: int
    avg_charging_duration_minutes: Optional[float]
    avg_wait_before_plug_minutes: Optional[float]
    avg_wait_after_plug_out_minutes: Optional[float]
    avg_total_session_minutes: Optional[float]
    loss_sessions: int          # sessions where car left without charging
    loss_rate_pct: float        # loss_sessions / total_sessions * 100


class ChargingPatternByGun(BaseModel):
    """Charging pattern metrics aggregated per charging gun."""
    gun_number: str
    camera_id: str
    total_sessions: int
    completed_sessions: int
    avg_charging_duration_minutes: Optional[float]
    avg_wait_before_plug_minutes: Optional[float]
    avg_wait_after_plug_out_minutes: Optional[float]
    avg_total_session_minutes: Optional[float]
    loss_sessions: int
    loss_rate_pct: float
    utilization_rate_pct: float  # charging_time / available_time * 100 (over date range)


class ChargingPatternAnalysisResponse(BaseModel):
    """Combined charging pattern analysis response."""
    by_car_model: List[ChargingPatternByModel]
    by_gun: List[ChargingPatternByGun]
    total_sessions: int
    total_completed: int
    total_loss: int
    overall_loss_rate_pct: float
