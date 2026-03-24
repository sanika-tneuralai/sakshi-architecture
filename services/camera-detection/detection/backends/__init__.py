"""
Detection backends package.
"""
from detection.backends.base import BaseDetector
from detection.backends.factory import create_detector

__all__ = ["BaseDetector", "create_detector"]

print("✓ detection.backends module loaded")
