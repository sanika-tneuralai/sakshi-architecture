"""
Usecase API endpoints.
"""
from fastapi import APIRouter, HTTPException

from usecase.schemas import UsecaseRequest, UsecaseResponse
from usecase.service import evaluate_usecases


router = APIRouter(prefix="/usecase", tags=["usecase"])


@router.post("/evaluate", response_model=UsecaseResponse)
def evaluate_usecase(request: UsecaseRequest):
    """
    Evaluate multiple usecases against detection output.
    
    **Scalable Multi-Usecase Architecture:**
    - Accepts ONE detection output
    - Evaluates MULTIPLE usecases in single request
    - Returns results for ALL usecases
    - Detection runs ONCE, usecases run on same data
    
    **Process:**
    1. Takes detection API output as input
    2. Evaluates each requested usecase independently
    3. Returns aggregated results for all usecases
    
    **Request Body:**
    - **camera_id**: Camera identifier
    - **detection_output**: Complete detection API response JSON
    - **usecases**: List of usecase IDs to evaluate (e.g., ["person_in_roi", "crowd_in_roi"])
    
    **Response:**
    - **camera_id**: Camera identifier
    - **results**: List of results, one per usecase
    
    **Available Usecases:**
    - person_in_roi: Triggers when any person is detected in ROI
    - crowd_in_roi: Triggers when 3+ persons are detected in ROI
    - restricted_zone_breach: Triggers when any vehicle is detected in ROI
    
    **Example Request:**
    ```json
    {
      "camera_id": "s1_cam_1",
      "detection_output": {...},
      "usecases": ["person_in_roi", "crowd_in_roi"]
    }
    ```
    """
    print(f"\n[API] ====================================================================")
    print(f"[API] USECASE API - MULTI-USECASE EVALUATION")
    print(f"[API] ====================================================================")
    print(f"[API] Endpoint: POST /usecase/evaluate")
    print(f"[API] Request received")
    print(f"[API] Camera ID: {request.camera_id}")
    print(f"[API] Usecases requested: {len(request.usecases)}")
    print(f"[API] Usecase IDs: {', '.join(request.usecases)}")
    
    num_detections = len(request.detection_output.get('detections', [])) if isinstance(request.detection_output, dict) else 0
    print(f"[API] Detection output contains: {num_detections} detection(s)")
    print(f"[API] ====================================================================\n")
    
    if not request.usecases or len(request.usecases) == 0:
        print(f"[API] ERROR: No usecases specified in request")
        raise HTTPException(status_code=400, detail="At least one usecase must be specified")
    
    try:
        print(f"[API] Calling orchestrator: evaluate_usecases()")
        print(f"[API] This will evaluate {len(request.usecases)} usecase(s) on the SAME detection output")
        
        # Call service orchestrator to evaluate all usecases
        result = evaluate_usecases(
            camera_id=request.camera_id,
            detection_output=request.detection_output,
            usecases=request.usecases
        )
        
        print(f"\n[API] ====================================================================")
        print(f"[API] ORCHESTRATOR COMPLETED")
        print(f"[API] ====================================================================")
        print(f"[API] Results received for {len(result['results'])} usecase(s)")
        
        triggered_count = sum(1 for r in result['results'] if r.triggered)
        print(f"[API] Triggered usecases: {triggered_count}/{len(result['results'])}")
        
        for uc_result in result['results']:
            status = "✓ TRIGGERED" if uc_result.triggered else "✗ Not triggered"
            print(f"[API]   - {uc_result.usecase_id}: {status} ({uc_result.matched_count} matches)")
        
        print(f"[API] Returning multi-usecase response")
        print(f"[API] Response status: 200 OK")
        print(f"[API] ====================================================================\n")
        
        return result
        
    except Exception as e:
        print(f"\n[API] !!!!! EXCEPTION CAUGHT !!!!!")
        print(f"[API] ERROR: Usecase evaluation failed")
        print(f"[API] Exception type: {type(e).__name__}")
        print(f"[API] Exception message: {str(e)}")
        print(f"[API] Raising HTTPException with status 500")
        print(f"[API] ====================================================================\n")
        raise HTTPException(status_code=500, detail=f"Usecase evaluation failed: {str(e)}")

