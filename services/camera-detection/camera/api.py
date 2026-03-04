from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
import cv2
import base64

from camera.schemas import RTSPConfig, MultiStreamConfig, CameraStatus
from camera.service import camera_manager
from shared.common.utils import validate_rtsp_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/camera", tags=["camera"])


@router.post("/start", response_model=dict)
async def start_camera(config: RTSPConfig):
    """
    Start a single DeepStream camera
    
    **Single Camera Mode:**
    - Best for < 10 cameras
    - Each camera runs independently
    - Lower latency per camera
    - Higher resource usage
    
    **Request Body:**
    - **camera_id**: Unique identifier (required)
    - **rtsp_url**: RTSP stream URL (required)
    - **fps**: Frame rate 1-30 (default: 5)
    - **roi_points**: Optional polygon ROI [[x1,y1], [x2,y2], ...]
    
    **Example:**
    ```json
    {
      "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
      "camera_id": "queue_cam_1",
      "fps": 5,
      "roi_points": [[1055.55, 536.47], [951.55, 452.47], [1101.55, 347.47], [1228.55, 413.47]]
    }
    ```
    """
    try:
        # Validate RTSP URL
        if not validate_rtsp_url(config.rtsp_url):
            raise HTTPException(status_code=400, detail="Invalid RTSP URL format")
        
        await camera_manager.start_single_camera(config)
        
        logger.info(f"✓ start_camera completed for {config.camera_id}")
        return {
            "status": "success",
            "message": f"Camera {config.camera_id} started successfully",
            "camera_id": config.camera_id,
            "fps": config.fps,
            "has_roi": config.roi_points is not None,
            "backend": "opencv-ffmpeg"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-multi", response_model=dict)
async def start_multi_stream(config: MultiStreamConfig):
    """
    Start multiple cameras in batched DeepStream mode
    
    **Multi-Stream Mode:**
    - **Recommended for 100+ cameras**
    - Processes cameras in batches using GPU
    - Much higher efficiency
    - Lower latency overall
    - Better resource utilization
    
    **Request Body:**
    - **streams**: List of camera configurations (required)
    - **batch_size**: Batch size 1-128 (default: 30)
    - **width**: Stream width (default: 1920)
    - **height**: Stream height (default: 1080)
    
    **Example:**
    ```json
    {
      "batch_size": 30,
      "width": 1920,
      "height": 1080,
      "streams": [
        {
          "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
          "camera_id": "queue_cam_1",
          "fps": 5,
          "roi_points": [[1055.55, 536.47], [951.55, 452.47], [1101.55, 347.47], [1228.55, 413.47]]
        },
        {
          "rtsp_url": "rtsp://admin:admin@192.168.1.101:554/cam/realmonitor?channel=1&subtype=1",
          "camera_id": "queue_cam_2",
          "fps": 5
        }
      ]
    }
    ```
    """
    try:
        # Validate all RTSP URLs
        for stream in config.streams:
            if not validate_rtsp_url(stream.rtsp_url):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid RTSP URL for camera {stream.camera_id}"
                )
        
        await camera_manager.start_multi_stream(
            streams=config.streams,
            batch_size=config.batch_size,
            width=config.width,
            height=config.height
        )
        
        logger.info(f"✓ start_multi_stream completed for {len(config.streams)} cameras")
        return {
            "status": "success",
            "message": f"Started {len(config.streams)} cameras in multi-stream mode",
            "camera_ids": [s.camera_id for s in config.streams],
            "batch_size": config.batch_size,
            "resolution": f"{config.width}x{config.height}",
            "backend": "deepstream-multi",
            "recommended_for": "100+ cameras"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting multi-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{camera_id}", response_model=CameraStatus)
async def get_camera_status(camera_id: str):
    """
    Get detailed status of a specific camera
    
    **Path Parameters:**
    - **camera_id**: Camera identifier
    
    **Returns:**
    - Camera operational status
    - Frame count
    - Last frame timestamp
    - Backend type (single/multi)
    """
    status = camera_manager.get_camera_status(camera_id)
    
    if not status:
        raise HTTPException(
            status_code=404, 
            detail=f"Camera {camera_id} not found"
        )
    
    logger.info(f"✓ get_camera_status completed for {camera_id}")
    return status


@router.get("/list", response_model=dict)
async def list_cameras():
    """
    List all active cameras
    
    **Returns:**
    - List of single-stream cameras
    - List of multi-stream cameras
    - Total count
    - Current mode (single-stream/multi-stream/idle)
    """
    logger.info("✓ list_cameras completed")
    return camera_manager.list_cameras()


@router.get("/frame/{camera_id}")
async def get_frame(
    camera_id: str,
    include_metadata: bool = Query(default=True, description="Include frame metadata")
):
    """
    Get latest frame from camera for detection API
    
    **Path Parameters:**
    - **camera_id**: Camera identifier
    
    **Query Parameters:**
    - **include_metadata**: Whether to include detailed metadata (default: true)
    
    **Returns:**
    - Frame information and metadata
    - Ready for detection processing
    
    **Note:** Actual frame data is passed internally to detection API
    """
    print(f"\n[CAMERA API] GET /camera/frame/{camera_id}")
    print(f"[CAMERA API] Request for camera: {camera_id}")
    
    camera = camera_manager.get_camera_stream(camera_id)
    
    if not camera:
        print(f"[CAMERA API] ERROR: Camera {camera_id} not found")
        raise HTTPException(
            status_code=404, 
            detail=f"Camera {camera_id} not found"
        )
    
    try:
        if hasattr(camera, 'get_preprocessed_frame'):
            # Single camera
            print(f"[CAMERA API] Processing single-stream camera")
            frame_data = await camera.get_preprocessed_frame()
            
            if frame_data is None:
                print(f"[CAMERA API] ERROR: No frame available (camera initializing)")
                raise HTTPException(
                    status_code=503, 
                    detail="No frame available yet. Camera may still be initializing."
                )
            
            # Encode frame to JPEG and base64
            frame = frame_data['frame']
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            print(f"[CAMERA API] Frame encoded successfully")
            print(f"[CAMERA API]   - Frame shape: {frame_data['shape']}")
            print(f"[CAMERA API]   - Frame count: {frame_data['frame_count']}")
            print(f"[CAMERA API]   - Has ROI: {frame_data['roi_points'] is not None}")
            
            response = {
                "camera_id": camera_id,
                "frame": frame_base64,
                "status": "frame_ready",
                "backend": "opencv-ffmpeg"
            }
            
            if include_metadata:
                response.update({
                    "timestamp": frame_data['timestamp'],
                    "shape": list(frame_data['shape']),
                    "roi_points": frame_data['roi_points'],
                    "frame_count": frame_data['frame_count'],
                    "has_roi_mask": frame_data['roi_mask'] is not None
                })
            
            logger.info(f"✓ get_frame completed for {camera_id} (single-stream)")
            print(f"[CAMERA API] ✓ Frame retrieval completed\n")
            return response
            
        else:
            # Multi-stream camera
            print(f"[CAMERA API] Processing multi-stream camera")
            status = camera.get_camera_status(camera_id)
            if not status:
                print(f"[CAMERA API] ERROR: Camera not found in multi-stream")
                raise HTTPException(
                    status_code=404, 
                    detail="Camera not found in multi-stream"
                )
            
            print(f"[CAMERA API] Multi-stream status retrieved")
            print(f"[CAMERA API]   - Frame count: {status['frame_count']}")
            print(f"[CAMERA API]   - Stream index: {status['stream_index']}")
            
            response = {
                "camera_id": camera_id,
                "status": "frame_ready",
                "backend": "deepstream-multi"
            }
            
            if include_metadata:
                response.update({
                    "frame_count": status['frame_count'],
                    "last_frame_time": status['last_frame_time'],
                    "stream_index": status['stream_index']
                })
            
            logger.info(f"✓ get_frame completed for {camera_id} (multi-stream)")
            print(f"[CAMERA API] ✓ Frame retrieval completed\n")
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stop/{camera_id}")
async def stop_camera(camera_id: str):
    """
    Stop a single camera
    
    **Path Parameters:**
    - **camera_id**: Camera identifier
    
    **Note:** Cannot stop individual cameras in multi-stream mode
    """
    try:
        await camera_manager.stop_camera(camera_id)
        
        logger.info(f"✓ stop_camera completed for {camera_id}")
        return {
            "status": "success",
            "message": f"Camera {camera_id} stopped successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stop-multi")
async def stop_multi_stream():
    """
    Stop all cameras in multi-stream mode
    
    **Returns:**
    - List of stopped cameras
    - Count of stopped cameras
    
    **Note:** This stops the entire multi-stream pipeline
    """
    try:
        camera_ids = await camera_manager.stop_multi_stream()
        
        logger.info(f"✓ stop_multi_stream completed ({len(camera_ids)} cameras stopped)")
        return {
            "status": "success",
            "message": "Multi-stream mode stopped",
            "stopped_cameras": camera_ids,
            "count": len(camera_ids)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping multi-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stop-all")
async def stop_all_cameras():
    """
    Stop all cameras (both single and multi-stream)
    
    **Emergency Stop:**
    - Stops all running cameras
    - Cleans up all resources
    - Resets camera manager to idle state
    """
    try:
        await camera_manager.stop_all()
        
        logger.info("✓ stop_all_cameras completed")
        return {
            "status": "success",
            "message": "All cameras stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping all cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    **Returns:**
    - Service health status
    - Active cameras count
    - Current mode
    """
    camera_list = camera_manager.list_cameras()
    
    logger.info("✓ health_check completed")
    return {
        "status": "healthy",
        "service": "camera-management-api",
        "mode": camera_list['mode'],
        "active_cameras": camera_list['total_count'],
        "single_stream": len(camera_list['single_stream_cameras']),
        "multi_stream": len(camera_list['multi_stream_cameras'])
    }


# Helper function for detection API integration
def get_camera_stream(camera_id: str):
    """
    Get camera stream object for detection API
    
    Args:
        camera_id: Camera identifier
    
    Returns:
        Camera stream object or None
    """
    logger.info(f"✓ get_camera_stream completed for {camera_id}")
    return camera_manager.get_camera_stream(camera_id)