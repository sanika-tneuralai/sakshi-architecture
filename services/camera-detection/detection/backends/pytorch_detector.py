"""
PyTorch / ultralytics YOLO backend.

Wraps the existing ultralytics.YOLO inference logic so that
DetectionService can use it through the BaseDetector interface.
No behaviour changes from the original service.py inference code.
"""
import logging
from typing import List, Optional

import numpy as np

from detection.backends.base import BaseDetector
from detection.device import select_device
from detection.schemas import BoundingBox, Detection
from shared.common.config import YOLO_MODEL_PATH

logger = logging.getLogger(__name__)


class PyTorchDetector(BaseDetector):
    """
    Ultralytics YOLO detector running on PyTorch (CPU or CUDA).
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device if device else select_device()
        self._model = None
        self._load()
        logger.info("PyTorchDetector ready: %s on %s", model_path, self.device)
        print(f"✓ PyTorchDetector.__init__ completed on {self.device}")

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "pytorch"

    def warmup(self) -> None:
        """Run a dummy inference to trigger CUDA/JIT initialisation."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, confidence_threshold=0.5, iou_threshold=0.45, classes=None)
        logger.info("PyTorchDetector warmup complete")
        print("✓ PyTorchDetector.warmup completed")

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        classes: Optional[List[int]],
    ) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=BoundingBox(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                    ),
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            logger.info("YOLO model loaded from %s on %s", self.model_path, self.device)
            print(f"✓ PyTorchDetector._load completed: {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise


print("✓ detection.backends.pytorch_detector module loaded")
