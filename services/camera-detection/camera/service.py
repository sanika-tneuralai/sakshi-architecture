import logging
import threading
import time
from typing import Dict, Optional, List
from camera.streams.opencv import OpenCVCamera
from camera.schemas import RTSPConfig, CameraStatus

try:
    from camera.streams.multi_stream import MultiStreamManager, PYDS_AVAILABLE
except ImportError:
    MultiStreamManager = None
    PYDS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Central manager for all camera operations
    Handles both single and multi-stream modes
    """
    
    def __init__(self):
        self.single_cameras: Dict[str, OpenCVCamera] = {}
        self.multi_stream_manager: Optional[MultiStreamManager] = None
        self.pipeline_workers: Dict[str, threading.Thread] = {}
        self.pipeline_stop_events: Dict[str, threading.Event] = {}
        
    async def start_single_camera(self, config: RTSPConfig) -> bool:
        """Start a single camera stream"""
        if config.camera_id in self.single_cameras:
            raise ValueError(f"Camera {config.camera_id} already exists")
        
        if self.multi_stream_manager and self.multi_stream_manager.is_running:
            raise ValueError("Cannot start single camera while multi-stream mode is active")
        
        camera = OpenCVCamera(
            camera_id=config.camera_id,
            rtsp_url=config.rtsp_url,
            fps=config.fps,
            roi_points=config.roi_points
        )
        
        await camera.start()
        self.single_cameras[config.camera_id] = camera
        
        # Start background pipeline worker
        self._start_pipeline_worker(config.camera_id)
        
        logger.info(f"Started single camera: {config.camera_id}")
        logger.info(f"✓ CameraManager.start_single_camera completed for {config.camera_id}")
        return True
    
    async def start_multi_stream(self, streams: List[RTSPConfig], batch_size: int, width: int, height: int) -> bool:
        """Start multi-stream mode for multiple cameras"""
        if not PYDS_AVAILABLE or MultiStreamManager is None:
            raise ValueError(
                "Multi-stream mode not available. pyds module is missing. "
                "Use single-camera mode instead with /camera/start endpoint."
            )
        
        if self.multi_stream_manager and self.multi_stream_manager.is_running:
            raise ValueError("Multi-stream manager already running")
        
        if self.single_cameras:
            raise ValueError("Cannot start multi-stream while single cameras are active. Stop them first.")
        
        self.multi_stream_manager = MultiStreamManager(batch_size=batch_size)
        await self.multi_stream_manager.start(streams, width, height)
        
        logger.info(f"Started multi-stream mode with {len(streams)} cameras")
        logger.info(f"✓ CameraManager.start_multi_stream completed for {len(streams)} cameras")
        return True
    
    async def stop_camera(self, camera_id: str) -> bool:
        """Stop a single camera"""
        if camera_id not in self.single_cameras:
            raise ValueError(f"Camera {camera_id} not found")
        
        # Stop pipeline worker first
        self._stop_pipeline_worker(camera_id)
        
        camera = self.single_cameras[camera_id]
        await camera.stop()
        del self.single_cameras[camera_id]
        
        logger.info(f"Stopped camera: {camera_id}")
        logger.info(f"✓ CameraManager.stop_camera completed for {camera_id}")
        return True
    
    async def stop_multi_stream(self) -> List[str]:
        """Stop multi-stream mode and return list of stopped cameras"""
        if not self.multi_stream_manager:
            raise ValueError("No multi-stream manager active")
        
        camera_ids = list(self.multi_stream_manager.cameras.keys())
        await self.multi_stream_manager.stop()
        self.multi_stream_manager = None
        
        logger.info(f"Stopped multi-stream mode with {len(camera_ids)} cameras")
        logger.info(f"✓ CameraManager.stop_multi_stream completed ({len(camera_ids)} cameras)")
        return camera_ids
    
    async def stop_all(self):
        """Stop all cameras (single and multi-stream)"""
        # Stop single cameras
        for camera_id in list(self.single_cameras.keys()):
            await self.stop_camera(camera_id)
        
        # Stop multi-stream
        if self.multi_stream_manager:
            await self.stop_multi_stream()
        
        logger.info("Stopped all cameras")
        logger.info("✓ CameraManager.stop_all completed")
    
    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        """Get status of a specific camera"""
        # Check single cameras
        if camera_id in self.single_cameras:
            camera = self.single_cameras[camera_id]
            status_dict = camera.get_status()
            return CameraStatus(**status_dict)
        
        # Check multi-stream
        if self.multi_stream_manager:
            status_dict = self.multi_stream_manager.get_camera_status(camera_id)
            if status_dict:
                return CameraStatus(**status_dict)
        
        logger.info(f"✓ CameraManager.get_camera_status completed for {camera_id}")
        return None
    
    def list_cameras(self) -> Dict:
        """List all active cameras"""
        single = list(self.single_cameras.keys())
        multi = []
        
        if self.multi_stream_manager and self.multi_stream_manager.is_running:
            multi = list(self.multi_stream_manager.cameras.keys())
        
        result = {
            'single_stream_cameras': single,
            'multi_stream_cameras': multi,
            'total_count': len(single) + len(multi),
            'mode': 'multi-stream' if multi else 'single-stream' if single else 'idle'
        }
        logger.info("✓ CameraManager.list_cameras completed")
        return result
    
    def get_camera_stream(self, camera_id: str):
        """Get camera stream object for detection API"""
        if camera_id in self.single_cameras:
            return self.single_cameras[camera_id]
        
        if self.multi_stream_manager:
            return self.multi_stream_manager
        
        logger.info(f"✓ CameraManager.get_camera_stream completed for {camera_id}")
        return None
    
    def _start_pipeline_worker(self, camera_id: str):
        """Start background pipeline worker for a camera"""
        if camera_id in self.pipeline_workers:
            logger.warning(f"Pipeline worker already running for {camera_id}")
            return
        
        stop_event = threading.Event()
        self.pipeline_stop_events[camera_id] = stop_event
        
        worker = threading.Thread(
            target=self._pipeline_worker,
            args=(camera_id, stop_event),
            daemon=True,
            name=f"pipeline-{camera_id}"
        )
        self.pipeline_workers[camera_id] = worker
        worker.start()
        
        logger.info(f"Started pipeline worker for {camera_id}")
    
    def _stop_pipeline_worker(self, camera_id: str):
        """Stop background pipeline worker for a camera"""
        if camera_id not in self.pipeline_stop_events:
            return
        
        # Signal worker to stop
        self.pipeline_stop_events[camera_id].set()
        
        # Wait for worker to finish
        if camera_id in self.pipeline_workers:
            self.pipeline_workers[camera_id].join(timeout=5.0)
            del self.pipeline_workers[camera_id]
        
        del self.pipeline_stop_events[camera_id]
        logger.info(f"Stopped pipeline worker for {camera_id}")
    
    def _pipeline_worker(self, camera_id: str, stop_event: threading.Event):
        """Background worker that runs pipeline continuously"""
        print(f"[WORKER] Pipeline worker started for {camera_id}")
        
        while not stop_event.is_set():
            try:
                # Run pipeline once
                _run_pipeline_once(camera_id)
                
                # Throttle: run every 2 seconds
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"[WORKER] Pipeline error for {camera_id}: {str(e)}")
                # Continue running even on errors
                time.sleep(5)  # Wait longer on error
        
        print(f"[WORKER] Pipeline worker stopped for {camera_id}")


def _run_pipeline_once(camera_id: str):
    """Run complete pipeline once for a camera (direct service calls)"""
    from detection.service import get_detection_service
    from usecase.service import evaluate_usecases
    from alert.service import process_pipeline_alerts
    from alert.schemas import PipelineAlertRequest
    
    print(f"\n[PIPELINE] Running pipeline for {camera_id}")
    
    try:
        # Step 1: Get camera and frame
        camera = camera_manager.get_camera_stream(camera_id)
        if not camera:
            return
        
        frame = camera.get_frame()
        if frame is None:
            return
        
        # Step 2: Run detection
        detection_service = get_detection_service()
        roi_points = camera.roi_points if hasattr(camera, 'roi_points') else None
        roi_mask = camera.roi_mask if hasattr(camera, 'roi_mask') else None
        
        detection_result = detection_service.detect(
            frame=frame,
            camera_id=camera_id,
            roi_points=roi_points,
            roi_mask=roi_mask,
            confidence_threshold=0.5,
            iou_threshold=0.45,
            classes=None
        )
        
        detection_output = detection_result.model_dump() if hasattr(detection_result, 'model_dump') else detection_result.dict()
        
        print(f"[PIPELINE] Detection: {detection_result.total_detections_count} total, {detection_result.roi_detections_count} in ROI")
        
        # Step 3: Evaluate usecases
        usecases = ["person_in_roi", "crowd_in_roi", "restricted_zone_breach"]
        usecase_result = evaluate_usecases(
            camera_id=camera_id,
            detection_output=detection_output,
            usecases=usecases
        )
        
        triggered_count = sum(1 for r in usecase_result['results'] if r.triggered)
        print(f"[PIPELINE] Usecases: {triggered_count}/{len(usecases)} triggered")
        
        # Step 4: Send alerts
        alert_request = PipelineAlertRequest(
            camera_id=camera_id,
            usecase_results=[r.model_dump() if hasattr(r, 'model_dump') else r.dict() for r in usecase_result['results']]
        )
        alert_result = process_pipeline_alerts(alert_request)
        
        print(f"[PIPELINE] Alerts: {alert_result.total_alerts_sent} sent\n")
        
    except Exception as e:
        logger.error(f"[PIPELINE] Error: {str(e)}")
        raise


# Global camera manager instance
camera_manager = CameraManager()