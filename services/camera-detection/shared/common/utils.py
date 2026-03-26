"""
Common utility functions for frame processing and validation.
These utilities are standalone and can be used across all services.
"""
import logging
import numpy as np
import cv2
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_rtsp_url(rtsp_url: str) -> bool:
    """
    Validate RTSP URL or video file path.

    Accepts:
    - RTSP/RTSPS stream URLs
    - Local video file paths (must exist on disk)

    Args:
        rtsp_url: RTSP URL string or local file path

    Returns:
        True if valid, False otherwise
    """
    import os
    if not rtsp_url:
        return False

    valid_protocols = ['rtsp://', 'rtsps://']
    if any(rtsp_url.startswith(protocol) for protocol in valid_protocols):
        return True

    # Accept local video files that exist on disk
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.wmv', '.flv', '.webm')
    if rtsp_url.lower().endswith(video_extensions) and os.path.isfile(rtsp_url):
        return True

    return False


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
