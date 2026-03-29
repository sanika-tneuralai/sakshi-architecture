"""
Analytics API endpoints.
"""
from fastapi import APIRouter, Query
from datetime import date
from typing import Optional

from analytics.schemas import (
    DailyAnalyticsResponse, DailyAnalytics,
    AlertAnalyticsResponse, AlertAnalytics,
    DetectionAnalyticsResponse, DetectionAnalytics,
    ChargingSessionResponse, ChargingSessionRecord,
    ChargingPatternAnalysisResponse, ChargingPatternByModel, ChargingPatternByGun,
)
from analytics.service import (
    get_daily_analytics,
    get_alert_analytics,
    get_detection_analytics,
    get_charging_sessions,
    get_charging_pattern_analysis,
    _duration_minutes,
)


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/daily", response_model=DailyAnalyticsResponse)
def get_daily(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get daily aggregated analytics from analytics_daily table.
    
    **Filters:**
    - **camera_id**: Filter by specific camera
    - **start_date**: Start date for date range
    - **end_date**: End date for date range
    
    **Response:**
    - Aggregated daily analytics with detections, ROI violations, and alerts
    """
    print(f"\n[API] GET /analytics/daily called")
    print(f"[API] Filters: camera_id={camera_id}, start_date={start_date}, end_date={end_date}")
    
    results = get_daily_analytics(camera_id, start_date, end_date)
    
    data = [
        DailyAnalytics(
            date=r.date,
            camera_id=r.camera_id,
            total_detections=r.total_detections,
            roi_violations=r.roi_violations,
            alerts_sent=r.alerts_sent
        )
        for r in results
    ]
    
    print(f"[API] Returning {len(data)} daily analytics records\n")
    
    return DailyAnalyticsResponse(
        data=data,
        total_records=len(data)
    )


@router.get("/alerts", response_model=AlertAnalyticsResponse)
def get_alerts(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get alert analytics aggregated by camera and usecase.
    
    **Filters:**
    - **camera_id**: Filter by specific camera
    - **start_date**: Start date for date range
    - **end_date**: End date for date range
    
    **Response:**
    - Alert counts per camera and usecase (sent/failed)
    """
    print(f"\n[API] GET /analytics/alerts called")
    print(f"[API] Filters: camera_id={camera_id}, start_date={start_date}, end_date={end_date}")
    
    results = get_alert_analytics(camera_id, start_date, end_date)
    
    data = [
        AlertAnalytics(
            camera_id=r.camera_id,
            usecase_name=r.usecase_name,
            total_alerts=r.total_alerts,
            alerts_sent=r.alerts_sent,
            alerts_failed=r.alerts_failed
        )
        for r in results
    ]
    
    print(f"[API] Returning {len(data)} alert analytics records\n")
    
    return AlertAnalyticsResponse(
        data=data,
        total_records=len(data)
    )


@router.get("/detections", response_model=DetectionAnalyticsResponse)
def get_detections(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get detection analytics aggregated by camera.
    
    **Filters:**
    - **camera_id**: Filter by specific camera
    - **start_date**: Start date for date range
    - **end_date**: End date for date range
    
    **Response:**
    - Detection counts with ROI violation statistics
    """
    print(f"\n[API] GET /analytics/detections called")
    print(f"[API] Filters: camera_id={camera_id}, start_date={start_date}, end_date={end_date}")
    
    results = get_detection_analytics(camera_id, start_date, end_date)
    
    data = [
        DetectionAnalytics(
            camera_id=r.camera_id,
            total_detections=r.total_detections,
            roi_detections=r.roi_detections,
            non_roi_detections=r.non_roi_detections,
            roi_violation_rate=round((r.roi_detections / r.total_detections * 100), 2) if r.total_detections > 0 else 0.0
        )
        for r in results
    ]
    
    print(f"[API] Returning {len(data)} detection analytics records\n")

    return DetectionAnalyticsResponse(
        data=data,
        total_records=len(data)
    )


# ── Charging Session & Pattern Analysis endpoints ─────────────────────────────

@router.get("/charging/sessions", response_model=ChargingSessionResponse)
def get_sessions(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    gun_number: Optional[str] = Query(None, description="Filter by gun number (e.g. 'Gun 1')"),
    car_model: Optional[str] = Query(None, description="Filter by car model"),
    start_date: Optional[date] = Query(None, description="Filter sessions with in_time >= date"),
    end_date: Optional[date] = Query(None, description="Filter sessions with in_time <= date"),
    status: Optional[str] = Query(None, description="Filter by status: active, charging, completed, incomplete"),
):
    """
    List raw charging session records with computed duration fields.

    Each session covers a vehicle's full lifecycle at a charging station:
    - **in_time** – car entered parking ROI
    - **plug_time** – gun plugged in
    - **plug_out_time** – gun unplugged
    - **out_time** – car left parking ROI

    Derived durations (minutes) are computed on the fly.
    """
    print(f"\n[API] GET /analytics/charging/sessions called")
    sessions = get_charging_sessions(camera_id, gun_number, car_model, start_date, end_date, status)

    data = [
        ChargingSessionRecord(
            session_id=s.session_id,
            camera_id=s.camera_id,
            gun_number=s.gun_number,
            car_number=s.car_number,
            car_model=s.car_model,
            in_time=s.in_time,
            plug_time=s.plug_time,
            plug_out_time=s.plug_out_time,
            out_time=s.out_time,
            session_status=s.session_status,
            wait_before_plug_minutes=_duration_minutes(s.in_time, s.plug_time),
            charging_duration_minutes=_duration_minutes(s.plug_time, s.plug_out_time),
            wait_after_plug_out_minutes=_duration_minutes(s.plug_out_time, s.out_time),
            total_session_minutes=_duration_minutes(s.in_time, s.out_time),
        )
        for s in sessions
    ]

    print(f"[API] Returning {len(data)} charging session records\n")
    return ChargingSessionResponse(data=data, total_records=len(data))


@router.get("/charging/patterns", response_model=ChargingPatternAnalysisResponse)
def get_charging_patterns(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Comprehensive charging pattern and loss analysis.

    Returns two segmented views:

    **By car model** – how different vehicle types use the chargers:
    - Average charging duration, wait times, total session time
    - Loss rate (cars that left without charging)

    **By charging gun** – how each physical charger is being used:
    - Same duration metrics per gun
    - Utilization rate (% of time range the gun was actively charging)
    - Loss rate per gun

    **Loss analysis:** A "loss session" is when a car arrived (in_time set) but
    left without ever being plugged in (session_status = 'incomplete'). This
    represents a missed charging opportunity and potential revenue loss.
    """
    print(f"\n[API] GET /analytics/charging/patterns called")
    result = get_charging_pattern_analysis(camera_id, start_date, end_date)

    by_car_model = [ChargingPatternByModel(**m) for m in result["by_car_model"]]
    by_gun = [ChargingPatternByGun(**g) for g in result["by_gun"]]

    print(f"[API] Pattern analysis: {result['total_sessions']} sessions, "
          f"{len(by_car_model)} models, {len(by_gun)} guns\n")

    return ChargingPatternAnalysisResponse(
        by_car_model=by_car_model,
        by_gun=by_gun,
        total_sessions=result["total_sessions"],
        total_completed=result["total_completed"],
        total_loss=result["total_loss"],
        overall_loss_rate_pct=result["overall_loss_rate_pct"],
    )
