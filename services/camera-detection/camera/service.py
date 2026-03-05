import logging
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


# Global camera manager instance
camera_manager = CameraManager()