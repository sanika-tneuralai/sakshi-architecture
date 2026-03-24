"""
Detector factory.

Reads the INFERENCE_BACKEND environment variable (via Config) and
returns the appropriate BaseDetector subclass instance.

Supported values for INFERENCE_BACKEND:
  pytorch   (default) — ultralytics YOLO on CPU or CUDA
  hailo               — HailoRT SDK on Hailo-8 / Hailo-8L accelerator

The factory performs lazy imports so that heavy dependencies
(torch, hailo_platform) are only loaded when the corresponding
backend is actually selected.
"""
import logging
from typing import Optional

from detection.backends.base import BaseDetector
from shared.common.config import Config

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("pytorch", "hailo")


def create_detector(backend: Optional[str] = None) -> BaseDetector:
    """
    Instantiate and return the configured detector backend.

    Args:
        backend: Override the backend name. When None the value of
                 Config.get_inference_backend() (INFERENCE_BACKEND env var)
                 is used, defaulting to 'pytorch'.

    Returns:
        A fully-initialised BaseDetector subclass.

    Raises:
        ValueError: If the requested backend name is not recognised.
        ImportError: If the required SDK for the backend is not installed.
    """
    selected = (backend or Config.get_inference_backend()).lower().strip()

    if selected not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown inference backend '{selected}'. "
            f"Supported: {_SUPPORTED_BACKENDS}"
        )

    logger.info("Creating detector backend: %s", selected)
    print(f"✓ create_detector: initialising '{selected}' backend")

    if selected == "pytorch":
        from detection.backends.pytorch_detector import PyTorchDetector
        return PyTorchDetector()

    if selected == "hailo":
        from detection.backends.hailo_detector import HailoDetector
        return HailoDetector()

    # Should never reach here due to the check above, but satisfies type checkers.
    raise ValueError(f"Unhandled backend: {selected}")  # pragma: no cover


print("✓ detection.backends.factory module loaded")
