"""
Alert API endpoints.
"""
from fastapi import APIRouter

from alert.schemas import AlertRequest, AlertResponse, PipelineAlertRequest, PipelineAlertResponse
from alert.service import process_alert, process_pipeline_alerts


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


