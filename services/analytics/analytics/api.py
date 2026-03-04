"""
Analytics API endpoints.
"""
from fastapi import APIRouter, Query
from datetime import date
from typing import Optional

from analytics.schemas import (
    DailyAnalyticsResponse, DailyAnalytics,
    AlertAnalyticsResponse, AlertAnalytics,
    DetectionAnalyticsResponse, DetectionAnalytics
)
from analytics.service import (
    get_daily_analytics,
    get_alert_analytics,
    get_detection_analytics
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
