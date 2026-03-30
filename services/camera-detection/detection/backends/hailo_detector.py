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
# COCO-80 fallback class names (same order as standard YOLO training)
# ---------------------------------------------------------------------------
_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


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
        Decode raw Hailo output tensors into Detection objects.

        Hailo YOLO models output a flat tensor per detection head with layout:
          [batch, num_anchors, 5 + num_classes]
        where columns are: [cx, cy, w, h, obj_conf, cls0_prob, cls1_prob, …]

        Coordinates are normalised to [0, 1] relative to the model input size.
        """
        orig_h, orig_w = original_shape[:2]

        # Hailo YOLOv8 NMS postprocess output is a ragged structure:
        #   raw_output[key] is a list of shape (1, num_classes) where
        #   each element raw_output[key][0][cls_id] is a numpy array of
        #   shape (num_detections, 5): [y1, x1, y2, x2, score] normalised 0-1.
        print(f"[HAILO DEBUG] raw_output keys: {list(raw_output.keys())}")

        detections: List[Detection] = []

        for key, value in raw_output.items():
            print(f"[HAILO DEBUG] key={key}, type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'n/a'}")

            # value is (1, num_classes) ragged — index batch dim first
            batch = value[0]  # shape: (num_classes,) list of arrays
            num_classes = len(batch)
            print(f"[HAILO DEBUG] num_classes={num_classes}")

            for cls_id in range(num_classes):
                class_dets = np.array(batch[cls_id])  # (num_detections, 5)
                print(f"[HAILO DEBUG] cls_id={cls_id}, dets shape={class_dets.shape}")

                if class_dets.size == 0 or class_dets.ndim < 2:
                    continue

                for row in class_dets:
                    # row: [y1, x1, y2, x2, score]  normalised 0-1
                    y1_n, x1_n, y2_n, x2_n, score = row[0], row[1], row[2], row[3], row[4]
                    score = float(score)

                    if score < confidence_threshold:
                        continue
                    if classes is not None and cls_id not in classes:
                        continue

                    cls_name = (
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else str(cls_id)
                    )
                    detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=score,
                            bbox=BoundingBox(
                                x1=float(x1_n * orig_w),
                                y1=float(y1_n * orig_h),
                                x2=float(x2_n * orig_w),
                                y2=float(y2_n * orig_h),
                            ),
                        )
                    )

        print(f"[HAILO DEBUG] total detections after postprocess: {len(detections)}")
        return detections

    @staticmethod
    def _load_labels(labels_path: Optional[str]) -> List[str]:
        if not labels_path:
            return _COCO_CLASSES
        try:
            with open(labels_path) as f:
                names = [line.strip() for line in f if line.strip()]
            logger.info("Loaded %d class labels from %s", len(names), labels_path)
            return names
        except Exception as e:
            logger.warning("Could not load labels from %s: %s — using COCO defaults", labels_path, e)
            return _COCO_CLASSES


print("✓ detection.backends.hailo_detector module loaded")
