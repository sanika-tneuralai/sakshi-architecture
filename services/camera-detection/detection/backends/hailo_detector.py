"""
Hailo accelerator backend.

Uses the HailoRT Python SDK (hailo_platform) to run inference on a
Hailo-8 / Hailo-8L / M.2 module with a pre-compiled .hef model.

Pipeline
--------
1. Load the .hef network group via VDevice.
2. Configure input/output virtual streams.
3. Pre-process frame: resize → RGB → normalise to [0, 1] float32.
4. Run synchronous inference.
5. Post-process raw output tensors:
   - Decode YOLO-style detections (cx, cy, w, h, obj_conf, cls_probs).
   - Apply confidence threshold + NMS.
6. Return List[Detection] — identical schema to PyTorchDetector.

Requirements
------------
- hailo_platform  (installed with the HailoRT SDK, not on PyPI)
- A compiled .hef model file (use Hailo Dataflow Compiler to convert ONNX → .hef)

Environment variables consumed (via Config)
-------------------------------------------
HAILO_HEF_PATH        Path to compiled .hef model file
HAILO_INPUT_WIDTH     Model input width  (default: 640)
HAILO_INPUT_HEIGHT    Model input height (default: 640)
HAILO_LABELS_PATH     Path to labels .txt file (one class name per line)
                      Falls back to COCO-80 class names when not provided.
"""
import logging
from typing import List, Optional

import cv2
import numpy as np

from detection.backends.base import BaseDetector
from detection.schemas import BoundingBox, Detection
from shared.common.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default class names — must match new model label order
# {0: "fire", 1: "smoke", 2: "car", 3: "gun"}
# ---------------------------------------------------------------------------
_DEFAULT_CLASSES = ["fire", "smoke", "car", "gun"]


class HailoDetector(BaseDetector):
    """
    Object detector backed by the Hailo accelerator via HailoRT SDK.
    """

    def __init__(
        self,
        hef_path: Optional[str] = None,
        input_width: Optional[int] = None,
        input_height: Optional[int] = None,
        labels_path: Optional[str] = None,
    ):
        self.hef_path = hef_path or Config.get_hailo_hef_path()
        self.input_width = input_width or Config.get_hailo_input_width()
        self.input_height = input_height or Config.get_hailo_input_height()
        self.class_names = self._load_labels(labels_path or Config.get_hailo_labels_path())

        # HailoRT objects — populated by _load()
        self._device = None
        self._network_group = None
        self._input_vstreams_params = None
        self._output_vstreams_params = None

        self._load()
        logger.info("HailoDetector ready: %s (%dx%d)", self.hef_path, self.input_width, self.input_height)
        print(f"✓ HailoDetector.__init__ completed: {self.hef_path}")

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "hailo"

    def warmup(self) -> None:
        """Run a dummy inference to prime the Hailo pipeline."""
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.detect(dummy, confidence_threshold=0.5, iou_threshold=0.45, classes=None)
        logger.info("HailoDetector warmup complete")
        print("✓ HailoDetector.warmup completed")

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        classes: Optional[List[int]],
    ) -> List[Detection]:
        try:
            from hailo_platform import InferVStreams

            preprocessed = self._preprocess(frame)

            with self._network_group.activate():
                with InferVStreams(
                    self._network_group,
                    self._input_vstreams_params,
                    self._output_vstreams_params,
                ) as pipeline:
                    input_data = {
                        self._input_name: np.expand_dims(preprocessed, axis=0)
                    }
                    raw_output = pipeline.infer(input_data)

            detections = self._postprocess(
                raw_output,
                original_shape=frame.shape,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                classes=classes,
            )
            return detections

        except Exception as e:
            logger.error("Hailo inference error: %s", e)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError:
            logger.error(
                "hailo_platform not installed. "
                "Install the HailoRT SDK: https://hailo.ai/developer-zone/"
            )
            raise

        hef = HEF(self.hef_path)
        self._device = VDevice()

        self._network_group = self._device.configure(hef)[0]

        self._input_vstreams_params = InputVStreamParams.make(
            self._network_group,
            format_type=FormatType.UINT8,
        )
        self._output_vstreams_params = OutputVStreamParams.make(
            self._network_group,
            format_type=FormatType.FLOAT32,
        )

        # Cache the single input stream name
        self._input_name = hef.get_input_vstream_infos()[0].name

        logger.info("HEF loaded: %s", self.hef_path)
        print(f"✓ HailoDetector._load completed: {self.hef_path}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize to model input size, convert BGR→RGB, return uint8.
        Hailo NMS-baked models handle normalisation internally.
        """
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def _postprocess(
        self,
        raw_output: dict,
        original_shape: tuple,
        confidence_threshold: float,
        iou_threshold: float,
        classes: Optional[List[int]],
    ) -> List[Detection]:
        """
        Decode Hailo YOLOv8 NMS-baked output into Detection objects.

        Output tensor: goec_v2/yolov8_nms_postprocess
        Shape: (num_classes, 5, max_proposals) = (4, 5, 100)
        Each detection row: [y1, x1, y2, x2, score] normalised 0-1.
        """
        orig_h, orig_w = original_shape[:2]

        key = "goec_v2/yolov8_nms_postprocess"
        # Shape: (1, num_classes, 5, max_proposals) — remove batch dim
        data = raw_output[key][0]  # (num_classes, 5, max_proposals)

        detections: List[Detection] = []

        for cls_id in range(data.shape[0]):
            if classes is not None and cls_id not in classes:
                continue

            cls_dets = data[cls_id]  # (5, max_proposals)

            for i in range(cls_dets.shape[1]):
                y1_n, x1_n, y2_n, x2_n, score = cls_dets[:, i]
                score = float(score)

                if score < confidence_threshold:
                    continue
                # Skip zero-padded empty slots
                if y1_n == 0.0 and x1_n == 0.0 and y2_n == 0.0 and x2_n == 0.0:
                    continue

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id),
                        confidence=round(score, 6),
                        bbox=BoundingBox(
                            x1=float(x1_n * orig_w),
                            y1=float(y1_n * orig_h),
                            x2=float(x2_n * orig_w),
                            y2=float(y2_n * orig_h),
                        ),
                    )
                )

        print(f"[HAILO] detections: {len(detections)}")
        return detections

    @staticmethod
    def _load_labels(labels_path: Optional[str]) -> List[str]:
        if not labels_path:
            return _DEFAULT_CLASSES
        try:
            with open(labels_path) as f:
                names = [line.strip() for line in f if line.strip()]
            logger.info("Loaded %d class labels from %s", len(names), labels_path)
            return names
        except Exception as e:
            logger.warning("Could not load labels from %s: %s — using COCO defaults", labels_path, e)
            return _DEFAULT_CLASSES


print("✓ detection.backends.hailo_detector module loaded")
