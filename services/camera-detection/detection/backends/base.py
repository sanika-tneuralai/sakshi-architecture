"""
Abstract base class for all inference backends.

All detector implementations must inherit from BaseDetector and implement
the detect() method. This guarantees a uniform output format regardless
of the underlying accelerator (PyTorch, Hailo, etc.).
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from detection.schemas import Detection


class BaseDetector(ABC):
    """
    Abstract detector interface.

    Concrete subclasses wrap a specific inference backend
    (PyTorch/ultralytics, Hailo HEF, TensorRT, …) and normalise
    their output into a list of Detection objects.
    """

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        classes: Optional[List[int]],
    ) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy array (H x W x 3).
            confidence_threshold: Minimum confidence to keep a detection.
            iou_threshold: IOU threshold for Non-Maximum Suppression.
            classes: Optional list of class IDs to keep; None means all.

        Returns:
            List of Detection objects (may be empty).
        """

    @abstractmethod
    def warmup(self) -> None:
        """
        Optional warmup pass to initialise hardware / JIT compilation.
        Called once at startup before the service starts handling requests.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the backend (e.g. 'pytorch', 'hailo')."""


print("✓ detection.backends.base module loaded")
