"""
Camera stream implementations.
"""

from camera.streams.opencv import OpenCVCamera

try:
    from camera.streams.deepstream import DeepStreamCamera
    from camera.streams.multi_stream import MultiStreamManager
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False
    DeepStreamCamera = None
    MultiStreamManager = None

__all__ = ["OpenCVCamera", "DeepStreamCamera", "MultiStreamManager", "DEEPSTREAM_AVAILABLE"]
