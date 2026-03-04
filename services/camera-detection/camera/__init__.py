"""
Camera management module.
Handles RTSP stream connections, frame capture, and camera lifecycle.
"""

from camera.api import router as camera_router
from camera.service import CameraManager

__all__ = ["camera_router", "CameraManager"]
