import logging
import threading
from typing import Dict, Optional, List
from camera.streams.opencv import OpenCVCamera
from camera.schemas import RTSPConfig, CameraStatus
from shared.common.config import Config

try:
    from camera.streams.multi_stream import MultiStreamManager, PYDS_AVAILABLE
except ImportError:
    MultiStreamManager = None
    PYDS_AVAILABLE = False

# DeepStream backend (imports pyds/gi — only available inside the DeepStream
# container). Guarded so the service still imports on a dev box without it.
try:
    from camera.streams.deepstream_pipeline import DeepStreamPipeline, DeepStreamCameraView
    DEEPSTREAM_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    DeepStreamPipeline = None
    DeepStreamCameraView = None
    DEEPSTREAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Central manager for all camera operations
    Handles both single and multi-stream modes
    """
    
    def __init__(self):
        self.single_cameras: Dict[str, OpenCVCamera] = {}
        self.multi_stream_manager: Optional[MultiStreamManager] = None

        # DeepStream backend state. When INFERENCE_BACKEND=deepstream, all
        # cameras are served by ONE batched pipeline that is (re)built whenever
        # the camera set changes (debounced). ds_sources is the desired set.
        self.backend = Config.get_inference_backend()
        self.ds_pipeline: Optional["DeepStreamPipeline"] = None
        self.ds_sources: Dict[str, dict] = {}          # camera_id -> {rtsp_url, fps}
        self._ds_lock = threading.Lock()
        self._ds_timer: Optional[threading.Timer] = None

    @property
    def _deepstream_mode(self) -> bool:
        return self.backend == "deepstream"

    async def start_single_camera(self, config: RTSPConfig) -> bool:
        """Start a single camera stream"""
        # DeepStream: register the camera and (re)build the batched pipeline.
        if self._deepstream_mode:
            if not DEEPSTREAM_AVAILABLE:
                raise ValueError(
                    "INFERENCE_BACKEND=deepstream but the DeepStream pipeline "
                    "module failed to import (pyds/gi missing). Run inside the "
                    "DeepStream container."
                )
            self._ds_add_source(config.camera_id, config.rtsp_url, config.fps or 5)
            logger.info(f"✓ start_single_camera queued {config.camera_id} for DeepStream pipeline")
            return True

        if config.camera_id in self.single_cameras:
            raise ValueError(f"Camera {config.camera_id} already exists")

        if self.multi_stream_manager and self.multi_stream_manager.is_running:
            raise ValueError("Cannot start single camera while multi-stream mode is active")

        camera = OpenCVCamera(
            camera_id=config.camera_id,
            rtsp_url=config.rtsp_url,
            fps=config.fps
        )

        try:
            await camera.start()
        except Exception as e:
            raise ValueError(f"Failed to start camera {config.camera_id}: {e}") from e

        self.single_cameras[config.camera_id] = camera

        logger.info(f"Started single camera: {config.camera_id}")
        logger.info(f"✓ CameraManager.start_single_camera completed for {config.camera_id}")
        return True

    # ------------------------------------------------------------------
    # DeepStream batched-pipeline reconcile (Phase 1b.3)
    # ------------------------------------------------------------------
    def _ds_add_source(self, camera_id: str, rtsp_url: str, fps: int) -> None:
        with self._ds_lock:
            self.ds_sources[camera_id] = {"rtsp_url": rtsp_url, "fps": fps}
        self._ds_schedule_reconcile()

    def _ds_remove_source(self, camera_id: str) -> None:
        with self._ds_lock:
            self.ds_sources.pop(camera_id, None)
        self._ds_schedule_reconcile()

    def _ds_schedule_reconcile(self) -> None:
        """Debounce: coalesce a burst of start/stop calls into one rebuild."""
        delay = Config.get_deepstream_reconcile_debounce()
        with self._ds_lock:
            if self._ds_timer is not None:
                self._ds_timer.cancel()
            self._ds_timer = threading.Timer(delay, self._ds_reconcile)
            self._ds_timer.daemon = True
            self._ds_timer.start()

    def _ds_reconcile(self) -> None:
        """Rebuild the batched pipeline to match the current desired set."""
        with self._ds_lock:
            sources = dict(self.ds_sources)

        # Tear down the old pipeline (brief blip on all streams — accepted for
        # the stable-camera-set deployment model).
        if self.ds_pipeline is not None:
            try:
                self.ds_pipeline.stop()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error stopping DeepStream pipeline during reconcile: {e}")
            self.ds_pipeline = None

        if not sources:
            logger.info("DeepStream reconcile: no sources, pipeline left down")
            return

        try:
            pipe = DeepStreamPipeline(
                config_path=Config.get_nvinfer_config_path(),
                width=Config.get_deepstream_width(),
                height=Config.get_deepstream_height(),
                publish_fps=Config.get_deepstream_publish_fps(),
                max_batch=Config.get_deepstream_max_batch(),
            )
            for cam_id, cfg in sources.items():
                pipe.add_source(cam_id, cfg["rtsp_url"], cfg["fps"])
            pipe.start()
            self.ds_pipeline = pipe
            logger.info(f"✓ DeepStream pipeline (re)built with {len(sources)} camera(s)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to (re)build DeepStream pipeline: {e}", exc_info=True)
            self.ds_pipeline = None
    
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
        if self._deepstream_mode:
            if camera_id not in self.ds_sources:
                raise ValueError(f"Camera {camera_id} not found")
            self._ds_remove_source(camera_id)
            logger.info(f"✓ stop_camera removed {camera_id} from DeepStream pipeline")
            return True

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
        """Stop all cameras (single, multi-stream, and DeepStream)"""
        # Stop single cameras
        for camera_id in list(self.single_cameras.keys()):
            await self.stop_camera(camera_id)

        # Stop multi-stream
        if self.multi_stream_manager:
            await self.stop_multi_stream()

        # Stop DeepStream pipeline
        if self._deepstream_mode:
            with self._ds_lock:
                if self._ds_timer is not None:
                    self._ds_timer.cancel()
                    self._ds_timer = None
                self.ds_sources.clear()
            if self.ds_pipeline is not None:
                self.ds_pipeline.stop()
                self.ds_pipeline = None

        logger.info("Stopped all cameras")
        logger.info("✓ CameraManager.stop_all completed")
    
    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        """Get status of a specific camera"""
        # DeepStream cameras
        if self._deepstream_mode and camera_id in self.ds_sources:
            if self.ds_pipeline is not None:
                status_dict = self.ds_pipeline.get_status(camera_id)
                if status_dict:
                    return CameraStatus(**status_dict)
            # Registered but pipeline not up yet (still within debounce/build).
            src = self.ds_sources[camera_id]
            return CameraStatus(
                camera_id=camera_id, is_running=False, backend="deepstream",
                fps=src["fps"], frame_count=0, rtsp_url=src["rtsp_url"],
            )

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
        """List all active cameras with detailed information"""
        cameras_info = []

        # DeepStream cameras
        if self._deepstream_mode:
            for camera_id, src in self.ds_sources.items():
                st = self.ds_pipeline.get_status(camera_id) if self.ds_pipeline else None
                cameras_info.append({
                    'camera_id': camera_id,
                    'status': 'running' if (st and st.get('is_running')) else 'starting',
                    'fps': src['fps'],
                    'frame_count': st.get('frame_count', 0) if st else 0,
                    'rtsp_url': src['rtsp_url'],
                    'backend': 'deepstream',
                })
            result = {'cameras': cameras_info, 'total': len(cameras_info)}
            logger.info("✓ CameraManager.list_cameras completed (deepstream)")
            return result

        # Get info from single cameras
        for camera_id, camera in self.single_cameras.items():
            status_dict = camera.get_status()
            cameras_info.append({
                'camera_id': camera_id,
                'status': 'running' if status_dict.get('is_running') else 'stopped',
                'fps': status_dict.get('fps', 5),
                'frame_count': status_dict.get('frame_count', 0),
                'rtsp_url': status_dict.get('rtsp_url', ''),
                'backend': status_dict.get('backend', 'opencv-ffmpeg')
            })
        
        # Get info from multi-stream cameras if active
        if self.multi_stream_manager and self.multi_stream_manager.is_running:
            for camera_id in self.multi_stream_manager.cameras.keys():
                status_dict = self.multi_stream_manager.get_camera_status(camera_id)
                if status_dict:
                    cameras_info.append({
                        'camera_id': camera_id,
                        'status': 'running' if status_dict.get('is_running') else 'stopped',
                        'fps': status_dict.get('fps', 5),
                        'frame_count': status_dict.get('frame_count', 0),
                        'rtsp_url': status_dict.get('rtsp_url', ''),
                        'backend': 'deepstream-multi'
                    })
        
        result = {
            'cameras': cameras_info,
            'total': len(cameras_info)
        }
        logger.info("✓ CameraManager.list_cameras completed")
        return result
    
    def get_camera_stream(self, camera_id: str):
        """Get camera stream object for detection API"""
        # DeepStream: return a per-camera view over the shared pipeline. The
        # detection API detects `is_deepstream` and reads cached detections.
        if self._deepstream_mode:
            if camera_id in self.ds_sources and self.ds_pipeline is not None:
                return DeepStreamCameraView(self.ds_pipeline, camera_id)
            return None

        if camera_id in self.single_cameras:
            return self.single_cameras[camera_id]

        if self.multi_stream_manager:
            return self.multi_stream_manager

        logger.info(f"✓ CameraManager.get_camera_stream completed for {camera_id}")
        return None


# Global camera manager instance
camera_manager = CameraManager()