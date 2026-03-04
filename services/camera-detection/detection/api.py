"""
Detection API endpoints.
"""
from fastapi import APIRouter, HTTPException
import logging
import requests
from typing import Optional, Dict, Any

from detection.schemas import DetectionRequest, DetectionResponse, DetectionStats
from detection.service import get_detection_service
from camera.service import camera_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detection", tags=["detection"])


def fetch_camera_config(camera_id: str, config_api_url: str = "http://localhost:8000") -> Optional[Dict[str, Any]]:
    """
    Fetch camera configuration from Configuration API.
    
    Args:
        camera_id: Camera identifier
        config_api_url: Base URL of Configuration API
        
    Returns:
        Camera configuration dict or None if unavailable
    """
    print(f"[DETECTION CONFIG] Attempting to fetch configuration for camera_id: {camera_id}")
    print(f"[DETECTION CONFIG] Configuration API URL: {config_api_url}/config/camera/{camera_id}")
    
    try:
        response = requests.get(
            f"{config_api_url}/config/camera/{camera_id}",
            timeout=2.0
        )
        print(f"[DETECTION CONFIG] Configuration API response status: {response.status_code}")
        
        if response.status_code == 200:
            config = response.json()
            print(f"[DETECTION CONFIG] Configuration fetched successfully")
            print(f"[DETECTION CONFIG]   - confidence_threshold: {config.get('confidence_threshold')}")
            print(f"[DETECTION CONFIG]   - detection_model: {config.get('detection_model')}")
            print(f"[DETECTION CONFIG]   - roi_count: {len(config.get('rois', []))}")
            return config
        elif response.status_code == 404:
            print(f"[DETECTION CONFIG] No configuration found for camera_id: {camera_id}")
            return None
        else:
            print(f"[DETECTION CONFIG] Unexpected status code: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"[DETECTION CONFIG] WARNING: Configuration API timeout for camera_id: {camera_id}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[DETECTION CONFIG] WARNING: Configuration API connection error for camera_id: {camera_id}")
        return None
    except Exception as e:
        print(f"[DETECTION CONFIG] WARNING: Failed to fetch configuration: {str(e)}")
        return None


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Run object detection on current frame from a camera.
    
    **Process:**
    1. Gets current frame from camera manager
    2. Runs YOLO detection
    3. Filters detections by ROI
    4. Returns detection results
    
    **Request Body:**
    - **camera_id**: Camera to detect from (required)
    - **confidence_threshold**: Min confidence (default: 0.5)
    - **iou_threshold**: IOU for NMS (default: 0.45)
    - **classes**: Filter specific class IDs (optional)
    
    **Response:**
    - Detection results with ROI filtering
    - Processing time
    - Frame metadata
    """
    print(f"\n[DETECTION API] POST /detection/detect")
    print(f"[DETECTION API] Request for camera: {request.camera_id}")
    print(f"[DETECTION API] Confidence threshold: {request.confidence_threshold}")
    
    logger.info(f"Detection request for camera: {request.camera_id}")
    
    # Fetch camera configuration from Configuration API
    print(f"[DETECTION] Fetching camera configuration from Configuration API")
    camera_config = fetch_camera_config(request.camera_id)
    
    # Determine confidence threshold: Config API > Request > Default
    if camera_config and 'confidence_threshold' in camera_config:
        confidence_threshold = camera_config['confidence_threshold']
        print(f"[DETECTION] Using confidence_threshold from Configuration API: {confidence_threshold}")
    else:
        confidence_threshold = request.confidence_threshold
        print(f"[DETECTION] Using confidence_threshold from request (fallback): {confidence_threshold}")
    
    # Log detection model from config (selection only, no loading)
    if camera_config and 'detection_model' in camera_config:
        detection_model = camera_config['detection_model']
        print(f"[DETECTION] Detection model from Configuration API: {detection_model}")
    else:
        print(f"[DETECTION] No detection model in configuration, using default")
    
    # Get camera object
    print(f"[DETECTION API] Retrieving camera stream from camera manager")
    camera = camera_manager.get_camera_stream(request.camera_id)
    if not camera:
        print(f"[DETECTION API] ERROR: Camera {request.camera_id} not found or not running")
        logger.error(f"Camera {request.camera_id} not found or not running")
        raise HTTPException(status_code=404, detail=f"Camera {request.camera_id} not found or not running")
    
    print(f"[DETECTION API] Camera stream obtained")
    
    # Get frame from camera
    print(f"[DETECTION API] Retrieving frame from camera")
    frame = camera.get_frame()
    if frame is None:
        print(f"[DETECTION API] ERROR: No frame available from camera {request.camera_id}")
        logger.error(f"No frame available from camera {request.camera_id}")
        raise HTTPException(status_code=400, detail="No frame available from camera")
    
    print(f"[DETECTION API] Frame retrieved successfully")
    
    # Get ROI data from camera
    roi_points = camera.roi_points
    roi_mask = camera.roi_mask
    print(f"[DETECTION] Using ROI from camera object: roi_points={roi_points is not None}, roi_mask={roi_mask is not None}")
    
    # Run detection with configuration values
    print(f"[DETECTION] Running detection with confidence_threshold: {confidence_threshold}")
    detection_service = get_detection_service()
    result = detection_service.detect(
        frame=frame,
        camera_id=request.camera_id,
        roi_points=roi_points,
        roi_mask=roi_mask,
        confidence_threshold=confidence_threshold,
        iou_threshold=request.iou_threshold,
        classes=request.classes
    )
    
    print(f"[DETECTION API] Detection completed")
    print(f"[DETECTION API]   - Total detections: {result.total_detections_count}")
    print(f"[DETECTION API]   - ROI detections: {result.roi_detections_count}")
    print(f"[DETECTION API]   - Processing time: {result.processing_time_ms}ms")
    
    logger.info(f"Detection completed: {result.total_detections_count} total, {result.roi_detections_count} in ROI")
    
    print(f"[DETECTION API] ✓ Detection completed\n")
    print(f"✓ detect_objects completed for {request.camera_id}")
    
    return result


@router.get("/stats/{camera_id}", response_model=DetectionStats)
async def get_detection_stats(camera_id: str):
    """
    Get detection statistics for a camera.
    
    **Response:**
    - Total frames processed
    - Total detections
    - ROI detections
    - Average processing time
    """
    logger.info(f"Stats request for camera: {camera_id}")
    
    detection_service = get_detection_service()
    stats = detection_service.get_stats(camera_id)
    
    if not stats:
        logger.warning(f"No detection stats found for camera {camera_id}")
        raise HTTPException(status_code=404, detail=f"No detection stats for camera {camera_id}")
    
    print(f"✓ get_detection_stats completed for {camera_id}")
    return stats


@router.delete("/stats/{camera_id}")
async def reset_detection_stats(camera_id: str):
    """
    Reset detection statistics for a camera.
    """
    logger.info(f"Reset stats for camera: {camera_id}")
    
    detection_service = get_detection_service()
    detection_service.reset_stats(camera_id)
    
    print(f"✓ reset_detection_stats completed for {camera_id}")
    return {"status": "success", "message": f"Stats reset for camera {camera_id}"}


@router.get("/health")
async def detection_health():
    """
    Check detection service health.
    """
    try:
        detection_service = get_detection_service()
        device_info = {
            "device": detection_service.device,
            "model_path": detection_service.model_path,
            "model_loaded": detection_service.model is not None
        }
        print("✓ detection_health completed")
        return {
            "status": "healthy",
            "service": "detection",
            **device_info
        }
    except Exception as e:
        logger.error(f"Detection health check failed: {e}")
        print(f"✓ detection_health completed: unhealthy - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


print("✓ detection.api module loaded")
