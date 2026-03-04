"""
Object detection module.
Handles YOLO detection within ROI regions and outputs results.
"""

from detection.api import router as detection_router
from detection.service import get_detection_service

__all__ = ["detection_router", "get_detection_service"]
