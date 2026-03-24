"""
Common utility functions for frame processing and validation.
These utilities are standalone and can be used across all services.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_rtsp_url(rtsp_url: str) -> bool:
    """
    Validate RTSP URL format
    
    Args:
        rtsp_url: RTSP URL string
    
    Returns:
        True if valid, False otherwise
    """
    if not rtsp_url:
        return False
    
    valid_protocols = ['rtsp://', 'rtsps://']
    result = any(rtsp_url.startswith(protocol) for protocol in valid_protocols)
    return result


def calculate_fps(frame_count: int, elapsed_time: float) -> float:
    """
    Calculate actual FPS
    
    Args:
        frame_count: Number of frames processed
        elapsed_time: Time elapsed in seconds
    
    Returns:
        FPS value
    """
    if elapsed_time <= 0:
        return 0.0
    result = frame_count / elapsed_time
    return result


def resize_frame(frame: np.ndarray, width: int, height: int, 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize frame to specified dimensions
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        keep_aspect_ratio: Whether to maintain aspect ratio
    
    Returns:
        Resized frame
    """
    try:
        if keep_aspect_ratio:
            h, w = frame.shape[:2]
            aspect = w / h
            
            if width / height > aspect:
                new_width = int(height * aspect)
                new_height = height
            else:
                new_width = width
                new_height = int(width / aspect)
            
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            if new_width < width or new_height < height:
                top = (height - new_height) // 2
                bottom = height - new_height - top
                left = (width - new_width) // 2
                right = width - new_width - left
                resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            return resized
        else:
            result = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            return result
            
    except Exception as e:
        logger.error(f"Failed to resize frame: {str(e)}")
        return frame


def preprocess_frame_for_detection(frame: np.ndarray, 
                                   target_size: Optional[Tuple[int, int]] = None,
                                   normalize: bool = False) -> np.ndarray:
    """
    Preprocess frame for object detection
    
    Args:
        frame: Input frame
        target_size: Optional (width, height) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Preprocessed frame
    """
    try:
        processed = frame.copy()
        
        # Resize if needed
        if target_size:
            processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
        
    except Exception as e:
        logger.error(f"Failed to preprocess frame: {str(e)}")
        return frame


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format timestamp to ISO format
    
    Args:
        timestamp: Unix timestamp or None for current time
    
    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        result = datetime.now().isoformat()
    else:
        result = datetime.fromtimestamp(timestamp).isoformat()
    return result


def calculate_grid_layout(num_items: int) -> Tuple[int, int]:
    """
    Calculate optimal grid layout (rows, cols) for tiling
    
    Args:
        num_items: Number of items to arrange
    
    Returns:
        Tuple of (rows, cols)
    """
    rows = int(np.ceil(np.sqrt(num_items)))
    cols = int(np.ceil(num_items / rows))
    return rows, cols


def get_frame_metadata(frame: np.ndarray, camera_id: str, 
                       frame_count: int, timestamp: Optional[float] = None) -> Dict:
    """
    Extract frame metadata
    
    Args:
        frame: Input frame
        camera_id: Camera identifier
        frame_count: Current frame number
        timestamp: Frame timestamp
    
    Returns:
        Dictionary with frame metadata
    """
    h, w = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    
    result = {
        'camera_id': camera_id,
        'frame_count': frame_count,
        'timestamp': format_timestamp(timestamp),
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(frame.dtype),
        'size_bytes': frame.nbytes
    }
    return result


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of recent frames to track for FPS calculation
        """
        self.window_size = window_size
        self.frame_times = []
        self.start_time = datetime.now()
        
    def add_frame(self):
        """Record a frame processing time"""
        current_time = datetime.now().timestamp()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """
        Calculate current FPS based on recent frames
        
        Returns:
            Current FPS value
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff <= 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_diff
    
    def get_uptime(self) -> float:
        """
        Get uptime in seconds
        
        Returns:
            Uptime in seconds since monitor creation
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with FPS, uptime, and frame count
        """
        result = {
            'fps': round(self.get_fps(), 2),
            'uptime_seconds': round(self.get_uptime(), 2),
            'total_frames': len(self.frame_times),
            'window_size': self.window_size
        }
        return result
    
    def reset(self):
        """Reset the performance monitor"""
        self.frame_times = []
        self.start_time = datetime.now()


def log_system_info():
    """Log system information for debugging"""
    try:
        import platform
        import psutil
        
        logger.info("=" * 50)
        logger.info("System Information:")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        logger.info("=" * 50)
    except ImportError:
        logger.warning("psutil not available, skipping system info")
    except Exception as e:
        logger.error(f"Error logging system info: {str(e)}")
