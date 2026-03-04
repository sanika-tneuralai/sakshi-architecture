"""
Orchestration Service - Pipeline Controller

This service orchestrates the complete detection pipeline by making HTTP calls to:
- Camera-Detection Service (camera operations + object detection)
- Usecase Service (usecase evaluation)
- Alert Service (alert management)

It does NOT contain any camera/detection/usecase logic itself.
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel, Field
import logging
import uvicorn
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestration.log')
    ]
)

logger = logging.getLogger(__name__)

# Service URLs from environment variables
CAMERA_DETECTION_URL = os.getenv("CAMERA_DETECTION_URL", "http://localhost:8000")
USECASE_SERVICE_URL = os.getenv("USECASE_SERVICE_URL", "http://localhost:8001")
ALERT_SERVICE_URL = os.getenv("ALERT_SERVICE_URL", "http://localhost:8002")

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "0.5"))


def create_session_with_retry():
    """Create a requests session with retry logic"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# Global session with retry logic
http_session = create_session_with_retry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Orchestration Service")
    logger.info("=" * 60)
    logger.info(f"Camera-Detection URL: {CAMERA_DETECTION_URL}")
    logger.info(f"Usecase Service URL: {USECASE_SERVICE_URL}")
    logger.info(f"Alert Service URL: {ALERT_SERVICE_URL}")
    logger.info(f"Request Timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"Max Retries: {MAX_RETRIES}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Orchestration Service")
    http_session.close()
    logger.info("Orchestration Service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Orchestration Service",
    description="""
    ## Pipeline Orchestration Service
    
    Coordinates the complete detection pipeline by orchestrating calls to:
    - **Camera-Detection Service**: Frame extraction and object detection
    - **Usecase Service**: Usecase evaluation and rule processing
    - **Alert Service**: Alert generation and notification
    
    ### Features:
    - **Pipeline Execution**: End-to-end pipeline orchestration
    - **Error Handling**: Robust error handling with retries
    - **Service Communication**: HTTP-based inter-service communication
    - **Configurable**: Service URLs configurable via environment variables
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class PipelineRequest(BaseModel):
    """Request schema for pipeline execution"""
    camera_id: str = Field(..., description="Camera identifier")
    usecases: Optional[List[str]] = Field(
        None, 
        description="List of usecases to evaluate. If None, uses defaults."
    )
    confidence_threshold: Optional[float] = Field(
        0.5, 
        ge=0.0, 
        le=1.0,
        description="Detection confidence threshold"
    )


# ============================================================================
# ORCHESTRATION ENDPOINTS
# ============================================================================

@app.post("/pipeline/execute", tags=["pipeline"])
def execute_pipeline(request: PipelineRequest):
    """
    Orchestrate the complete pipeline: Camera → Detection → Usecase → Alert
    
    This endpoint orchestrates the complete pipeline by making HTTP calls to:
    1. Camera-Detection Service (get frame + run detection)
    2. Usecase Service (evaluate usecases)
    3. Alert Service (send alerts if triggered)
    
    **Request Body:**
    - **camera_id**: Camera identifier (required)
    - **usecases**: List of usecase IDs (optional, defaults to all)
    - **confidence_threshold**: Detection confidence 0.0-1.0 (optional, default: 0.5)
    
    **Example:**
    ```json
    {
      "camera_id": "s1_cam_1",
      "confidence_threshold": 0.5,
      "usecases": ["person_in_roi", "crowd_in_roi"]
    }
    ```
    
    **Returns:**
    Combined results from all services including detections, usecase evaluations, and alerts.
    """
    logger.info("\n" + "="*80)
    logger.info("[ORCHESTRATOR] PIPELINE EXECUTION STARTED")
    logger.info("="*80)
    logger.info(f"[ORCHESTRATOR] Camera ID: {request.camera_id}")
    logger.info(f"[ORCHESTRATOR] Usecases: {request.usecases or ['person_in_roi', 'crowd_in_roi', 'restricted_zone_breach']}")
    logger.info(f"[ORCHESTRATOR] Confidence Threshold: {request.confidence_threshold}")
    logger.info("="*80 + "\n")
    
    # Default usecases if none provided
    usecases = request.usecases or ["person_in_roi", "crowd_in_roi", "restricted_zone_breach"]
    
    try:
        # STEP 1: Get frame from Camera-Detection Service
        logger.info(f"[ORCHESTRATOR] STEP 1/4: Calling Camera API")
        logger.info(f"[ORCHESTRATOR] Endpoint: GET {CAMERA_DETECTION_URL}/camera/frame/{request.camera_id}")
        
        camera_response = http_session.get(
            f"{CAMERA_DETECTION_URL}/camera/frame/{request.camera_id}",
            timeout=REQUEST_TIMEOUT
        )
        logger.info(f"[ORCHESTRATOR] Camera API Response: {camera_response.status_code}")
        
        if camera_response.status_code != 200:
            logger.error(f"[ORCHESTRATOR] ERROR: Camera API failed")
            raise HTTPException(
                status_code=camera_response.status_code, 
                detail=f"Camera API failed: {camera_response.text}"
            )
        
        camera_data = camera_response.json()
        logger.info(f"[ORCHESTRATOR] Camera API Success")
        logger.info(f"[ORCHESTRATOR]   - Frame status: {camera_data.get('status')}")
        logger.info(f"[ORCHESTRATOR]   - Backend: {camera_data.get('backend')}")
        logger.info("")
        
        # STEP 2: Run Detection via Camera-Detection Service
        logger.info(f"[ORCHESTRATOR] STEP 2/4: Calling Detection API")
        logger.info(f"[ORCHESTRATOR] Endpoint: POST {CAMERA_DETECTION_URL}/detection/detect")
        
        detection_payload = {
            "camera_id": request.camera_id,
            "confidence_threshold": request.confidence_threshold
        }
        logger.info(f"[ORCHESTRATOR] Detection payload: {detection_payload}")
        
        detection_response = http_session.post(
            f"{CAMERA_DETECTION_URL}/detection/detect",
            json=detection_payload,
            timeout=REQUEST_TIMEOUT
        )
        logger.info(f"[ORCHESTRATOR] Detection API Response: {detection_response.status_code}")
        
        if detection_response.status_code != 200:
            logger.error(f"[ORCHESTRATOR] ERROR: Detection API failed")
            raise HTTPException(
                status_code=detection_response.status_code,
                detail=f"Detection API failed: {detection_response.text}"
            )
        
        detection_data = detection_response.json()
        logger.info(f"[ORCHESTRATOR] Detection API Success")
        logger.info(f"[ORCHESTRATOR]   - Total detections: {detection_data.get('total_detections_count')}")
        logger.info(f"[ORCHESTRATOR]   - ROI detections: {detection_data.get('roi_detections_count')}")
        logger.info(f"[ORCHESTRATOR]   - Processing time: {detection_data.get('processing_time_ms')}ms")
        logger.info("")
        
        # STEP 3: Evaluate Usecases via Usecase Service
        logger.info(f"[ORCHESTRATOR] STEP 3/4: Calling Usecase API")
        logger.info(f"[ORCHESTRATOR] Endpoint: POST {USECASE_SERVICE_URL}/usecase/evaluate")
        
        usecase_payload = {
            "camera_id": request.camera_id,
            "detection_output": detection_data,
            "usecases": usecases
        }
        logger.info(f"[ORCHESTRATOR] Evaluating {len(usecases)} usecases")
        
        usecase_response = http_session.post(
            f"{USECASE_SERVICE_URL}/usecase/evaluate",
            json=usecase_payload,
            timeout=REQUEST_TIMEOUT
        )
        logger.info(f"[ORCHESTRATOR] Usecase API Response: {usecase_response.status_code}")
        
        if usecase_response.status_code != 200:
            logger.error(f"[ORCHESTRATOR] ERROR: Usecase API failed")
            raise HTTPException(
                status_code=usecase_response.status_code,
                detail=f"Usecase API failed: {usecase_response.text}"
            )
        
        usecase_data = usecase_response.json()
        logger.info(f"[ORCHESTRATOR] Usecase API Success")
        logger.info(f"[ORCHESTRATOR]   - Results count: {len(usecase_data.get('results', []))}")
        
        triggered_usecases = [r for r in usecase_data.get('results', []) if r.get('triggered')]
        logger.info(f"[ORCHESTRATOR]   - Triggered usecases: {len(triggered_usecases)}/{len(usecase_data.get('results', []))}")
        
        for result in usecase_data.get('results', []):
            status = "✓ TRIGGERED" if result.get('triggered') else "✗ Not triggered"
            logger.info(f"[ORCHESTRATOR]     {result.get('usecase_id')}: {status}")
        logger.info("")
        
        # STEP 4: Send Alerts via Alert Service
        logger.info(f"[ORCHESTRATOR] STEP 4/4: Calling Alert API")
        logger.info(f"[ORCHESTRATOR] Endpoint: POST {ALERT_SERVICE_URL}/alert/send")
        
        alert_payload = {
            "camera_id": request.camera_id,
            "usecase_results": usecase_data.get('results', [])
        }
        logger.info(f"[ORCHESTRATOR] Processing alerts for {len(triggered_usecases)} triggered usecases")
        
        alert_response = http_session.post(
            f"{ALERT_SERVICE_URL}/alert/send",
            json=alert_payload,
            timeout=REQUEST_TIMEOUT
        )
        logger.info(f"[ORCHESTRATOR] Alert API Response: {alert_response.status_code}")
        
        if alert_response.status_code != 200:
            logger.warning(f"[ORCHESTRATOR] WARNING: Alert API failed (non-critical)")
            logger.warning(f"[ORCHESTRATOR] Alert error: {alert_response.text}")
            alert_data = {"alerts_sent": [], "status": "failed"}
        else:
            alert_data = alert_response.json()
            logger.info(f"[ORCHESTRATOR] Alert API Success")
            logger.info(f"[ORCHESTRATOR]   - Alerts sent: {len(alert_data.get('alerts_sent', []))}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("[ORCHESTRATOR] PIPELINE EXECUTION COMPLETED")
        logger.info("="*80 + "\n")
        
        # Return combined results
        return {
            "status": "success",
            "camera_id": request.camera_id,
            "pipeline_results": {
                "camera": {
                    "status": camera_data.get('status'),
                    "backend": camera_data.get('backend')
                },
                "detection": {
                    "total_detections": detection_data.get('total_detections_count'),
                    "roi_detections": detection_data.get('roi_detections_count'),
                    "processing_time_ms": detection_data.get('processing_time_ms')
                },
                "usecases": {
                    "evaluated": len(usecase_data.get('results', [])),
                    "triggered": len(triggered_usecases),
                    "results": usecase_data.get('results', [])
                },
                "alerts": {
                    "sent": len(alert_data.get('alerts_sent', [])),
                    "details": alert_data.get('alerts_sent', [])
                }
            }
        }
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[ORCHESTRATOR] ERROR: Connection failed - {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service connection failed: {str(e)}"
        )
    except requests.exceptions.Timeout as e:
        logger.error(f"[ORCHESTRATOR] ERROR: Request timeout - {str(e)}")
        raise HTTPException(
            status_code=504,
            detail=f"Service timeout: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] ERROR: Pipeline execution failed - {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline execution failed: {str(e)}"
        )


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the orchestration service and connectivity to dependent services.
    """
    health_status = {
        "status": "healthy",
        "service": "orchestration",
        "version": "1.0.0",
        "services": {}
    }
    
    # Check connectivity to dependent services
    services = {
        "camera_detection": CAMERA_DETECTION_URL,
        "usecase": USECASE_SERVICE_URL,
        "alert": ALERT_SERVICE_URL
    }
    
    for service_name, service_url in services.items():
        try:
            response = http_session.get(f"{service_url}/health", timeout=5)
            health_status["services"][service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": service_url
            }
        except Exception as e:
            health_status["services"][service_name] = {
                "status": "unreachable",
                "url": service_url,
                "error": str(e)
            }
    
    # Overall health is unhealthy if any service is down
    if any(s["status"] != "healthy" for s in health_status["services"].values()):
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/", tags=["root"])
async def root():
    """API Root - Welcome and Quick Links"""
    return {
        "message": "Orchestration Service - Pipeline Controller",
        "version": "1.0.0",
        "status": "operational",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "execute_pipeline": "/pipeline/execute"
        },
        "configured_services": {
            "camera_detection": CAMERA_DETECTION_URL,
            "usecase": USECASE_SERVICE_URL,
            "alert": ALERT_SERVICE_URL
        }
    }


# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )
