"""
Alert API endpoints.
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from alert.schemas import AlertRequest, AlertResponse, PipelineAlertRequest, PipelineAlertResponse, AlertDetail, AlertListResponse, AlertRecord
from alert.service import process_alert, process_pipeline_alerts
from shared.database.connection import SessionLocal
from shared.database.models import Alert


router = APIRouter(prefix="/alert", tags=["alert"])


@router.post("/send", response_model=PipelineAlertResponse)
def send_alert(request: PipelineAlertRequest):
    """
    Process and send alerts based on usecase evaluation results.
    
    **Process:**
    1. Receives usecase results from orchestrator
    2. Evaluates each triggered usecase for alerting
    3. Sends appropriate alerts
    4. Returns alert status
    
    **Request Body:**
    - **camera_id**: Camera identifier
    - **usecase_results**: List of usecase evaluation results
    
    **Response:**
    - Alert status with details of all alerts sent
    """
    print(f"\n[API] ============== FUNCTION ENTRY: send_alert ==============")
    print(f"[API] POST /alert/send called")
    print(f"[API] Request received for camera_id: {request.camera_id}")
    print(f"[API] Usecase results count: {len(request.usecase_results)}")
    
    response = process_pipeline_alerts(request)
    
    print(f"[API] Preparing to return response...")
    print(f"[API] Total alerts sent: {response.total_alerts_sent}")
    print(f"[API] ============== FUNCTION EXIT: send_alert ==============\n")
    
    return response


@router.post("/send-single", response_model=AlertResponse)
def send_single_alert(request: AlertRequest):
    """
    Process and send a single alert (legacy endpoint).
    
    **Process:**
    1. Receives alert-ready payload from Usecase API
    2. Evaluates alert rule: usecase_id == "person_in_roi" AND alert_required == true
    3. Simulates sending alert (print-only)
    4. Returns alert status
    
    **Request Body:**
    - **camera_id**: Camera identifier
    - **usecase_id**: Usecase that triggered the alert
    - **alert_required**: Whether alert is required
    - **alert_type**: Type of alert
    - **alert_objects**: Objects that triggered the alert
    - **alert_count**: Number of objects
    
    **Response:**
    - Alert status with sent confirmation
    """
    print(f"\n[API] ============== FUNCTION ENTRY: send_single_alert ==============")
    print(f"[API] POST /alert/send-single called")
    print(f"[API] Request received for camera_id: {request.camera_id}")
    
    response = process_alert(request)
    
    print(f"[API] Preparing to return response...")
    print(f"[API] Response: {response.model_dump()}")
    print(f"[API] ============== FUNCTION EXIT: send_single_alert ==============\n")
    
    return response


@router.get("/list", response_model=AlertListResponse)
def get_alerts(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    usecase_name: Optional[str] = Query(None, description="Filter by usecase name"),
    limit: int = Query(100, description="Max records to return", ge=1, le=1000)
):
    """
    Get alert records.
    
    **Query Parameters:**
    - **camera_id**: Filter by camera (optional)
    - **usecase_name**: Filter by usecase (optional)
    - **limit**: Maximum records (default: 100)
    
    **Response:**
    - List of alerts with timestamps and screenshot paths
    """
    db = SessionLocal()
    try:
        query = db.query(Alert)
        
        if camera_id:
            query = query.filter(Alert.camera_id == camera_id)
        if usecase_name:
            query = query.filter(Alert.usecase_name == usecase_name)
        
        query = query.order_by(Alert.timestamp.desc()).limit(limit)
        results = query.all()
        
        alerts = [
            AlertRecord(
                alert_id=r.alert_id,
                camera_id=r.camera_id,
                usecase_name=r.usecase_name,
                alert_type=r.alert_type,
                timestamp=r.timestamp.isoformat() if r.timestamp else None,
                status=r.status,
                screenshot_path=r.screenshot_path,
                snapshot_b64=r.snapshot_b64,
                extras=r.extras,
            )
            for r in results
        ]
        
        return AlertListResponse(alerts=alerts, total=len(alerts))
    finally:
        db.close()
