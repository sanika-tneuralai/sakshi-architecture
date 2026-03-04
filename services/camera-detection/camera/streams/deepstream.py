import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, List, Dict

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

from threading import Thread

logger = logging.getLogger(__name__)


class DeepStreamCamera:
    """Single DeepStream camera handler with hardware acceleration"""
    
    def __init__(self, camera_id: str, rtsp_url: str, fps: int = 5, roi_points: Optional[List] = None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.roi_points = roi_points
        self.roi_mask = None
        
        # Pipeline components
        self.pipeline = None
        self.loop = None
        self.thread = None
        self.is_running = False
        
        # Frame management
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.error_count = 0
        self.max_errors = 5
        
    def _create_roi_mask(self, frame_shape: tuple) -> Optional[np.ndarray]:
        """Create binary mask from polygon ROI points"""
        if not self.roi_points or len(self.roi_points) < 3:
            return None
        
        try:
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            points = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            logger.info(f"ROI mask created for {self.camera_id}")
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera._create_roi_mask completed")
            return mask
        except Exception as e:
            logger.error(f"Failed to create ROI mask for {self.camera_id}: {str(e)}")
            return None
    
    def _build_pipeline(self) -> bool:
        """Build flexible GStreamer pipeline that auto-negotiates codec"""
        drop_interval = max(1, 30 // self.fps)
        
        # Flexible pipeline with relaxed RTSP settings for compatibility
        # Try UDP+TCP protocols, increase timeout, disable strict RTSP
        pipeline_str = f"""
            rtspsrc location={self.rtsp_url} 
                latency=2000 
                timeout=10000000
                tcp-timeout=10000000
                retry=5
                drop-on-latency=true 
                protocols=tcp+udp-mcast+udp
                ntp-sync=false
                ntp-time-source=0
                buffer-mode=auto
                do-retransmission=false ! 
            queue max-size-buffers=2 leaky=downstream ! 
            decodebin ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw, format=BGR ! 
            videorate drop-only=true ! 
            video/x-raw, framerate={self.fps}/1 ! 
            appsink name=appsink emit-signals=True max-buffers=1 drop=True sync=false
        """
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Configure appsink
            appsink = self.pipeline.get_by_name("appsink")
            appsink.connect("new-sample", self._on_new_sample)
            
            # Add bus watch for error handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            logger.info(f"Pipeline created successfully for {self.camera_id}")
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera._build_pipeline completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create pipeline for {self.camera_id}: {str(e)}")
            return False
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[{self.camera_id}] Error: {err}, Debug: {debug}")
            self.error_count += 1
            
            if self.error_count >= self.max_errors:
                logger.critical(f"[{self.camera_id}] Max errors reached, stopping")
                self.is_running = False
                
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"[{self.camera_id}] Warning: {warn}")
            
        elif msg_type == Gst.MessageType.EOS:
            logger.info(f"[{self.camera_id}] End of stream")
            self.is_running = False
            
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            logger.debug(f"[{self.camera_id}] State changed: {old.value_nick} -> {new.value_nick}")
        
        logger.debug(f"[{self.camera_id}] ✓ DeepStreamCamera._on_bus_message completed")
    
    def _on_new_sample(self, appsink):
        """Callback for processing new frames"""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Extract frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Map buffer to read data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                logger.warning(f"[{self.camera_id}] Failed to map buffer")
                return Gst.FlowReturn.ERROR
            
            try:
                # Convert to numpy array (BGR format)
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Create ROI mask on first successful frame
                if self.roi_points and self.roi_mask is None:
                    self.roi_mask = self._create_roi_mask(frame.shape)
                
                # Update frame data
                self.current_frame = frame.copy()
                self.frame_count += 1
                self.last_frame_time = datetime.now().timestamp()
                self.error_count = 0  # Reset error count on success
                
            finally:
                buffer.unmap(map_info)
                
        except Exception as e:
            logger.error(f"[{self.camera_id}] Error processing frame: {str(e)}")
            return Gst.FlowReturn.ERROR
        
        logger.debug(f"[{self.camera_id}] ✓ DeepStreamCamera._on_new_sample completed")
        return Gst.FlowReturn.OK
    
    def _run_loop(self):
        """Run GLib main loop in dedicated thread"""
        try:
            self.loop = GLib.MainLoop()
            logger.info(f"[{self.camera_id}] Starting GLib main loop")
            self.loop.run()
        except Exception as e:
            logger.error(f"[{self.camera_id}] Error in GLib loop: {str(e)}")
        finally:
            logger.info(f"[{self.camera_id}] GLib main loop stopped")
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera._run_loop completed")
    
    async def start(self) -> bool:
        """Start camera stream"""
        try:
            if self.is_running:
                logger.warning(f"[{self.camera_id}] Already running")
                return False
            
            # Build pipeline
            if not self._build_pipeline():
                raise Exception("Failed to build pipeline")
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            
            # Start GLib loop in separate thread
            self.is_running = True
            self.thread = Thread(target=self._run_loop, daemon=True, name=f"GLib-{self.camera_id}")
            self.thread.start()
            
            logger.info(f"[{self.camera_id}] Started successfully")
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.start completed")
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Failed to start: {str(e)}")
            await self.stop()
            raise
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """Get latest raw frame"""
        if self.current_frame is not None:
            result = self.current_frame.copy()
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.get_frame completed")
            return result
        logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.get_frame completed (no frame)")
        return None
    
    async def get_preprocessed_frame(self) -> Optional[Dict]:
        """Get preprocessed frame with metadata for detection"""
        frame = await self.get_frame()
        
        if frame is None:
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.get_preprocessed_frame completed (no frame)")
            return None
        
        # Apply ROI mask if available
        masked_frame = frame.copy()
        if self.roi_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        
        result = {
            'frame': masked_frame,
            'original_frame': frame,
            'roi_mask': self.roi_mask,
            'timestamp': datetime.now().isoformat(),
            'camera_id': self.camera_id,
            'shape': list(frame.shape),
            'roi_points': self.roi_points,
            'frame_count': self.frame_count
        }
        logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.get_preprocessed_frame completed")
        return result
    
    async def stop(self):
        """Stop camera stream and cleanup resources"""
        try:
            logger.info(f"[{self.camera_id}] Stopping...")
            self.is_running = False
            
            # Stop pipeline
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            # Stop GLib loop
            if self.loop and self.loop.is_running():
                self.loop.quit()
            
            # Wait for thread to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    logger.warning(f"[{self.camera_id}] Thread did not stop gracefully")
            
            # Cleanup
            self.pipeline = None
            self.loop = None
            self.thread = None
            self.current_frame = None
            
            logger.info(f"[{self.camera_id}] Stopped successfully")
            logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.stop completed")
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Error during stop: {str(e)}")
    
    def get_status(self) -> Dict:
        """Get camera status information"""
        result = {
            'camera_id': self.camera_id,
            'is_running': self.is_running,
            'rtsp_url': self.rtsp_url,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'last_frame_time': self.last_frame_time,
            'error_count': self.error_count,
            'has_current_frame': self.current_frame is not None,
            'backend': 'deepstream-single'
        }
        logger.info(f"[{self.camera_id}] ✓ DeepStreamCamera.get_status completed")
        return result