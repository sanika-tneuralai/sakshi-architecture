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
        Decode raw Hailo YOLOv8 output tensors into Detection objects.

        This HEF outputs 6 raw tensors — 2 per detection scale, no NMS baked in:
          conv41: (1, 80, 80, 64) — P3 regression (DFL, reg_max=16)
          conv42: (1, 80, 80,  4) — P3 classification (4 classes)
          conv52: (1, 40, 40, 64) — P4 regression
          conv53: (1, 40, 40,  4) — P4 classification
          conv62: (1, 20, 20, 64) — P5 regression
          conv63: (1, 20, 20,  4) — P5 classification

        Steps: DFL decode → xyxy boxes → sigmoid scores → threshold → NMS
        """
        orig_h, orig_w = original_shape[:2]

        # Fixed head pairs: (regression_key, classification_key)
        head_pairs = [
            ("goec_v2/conv41", "goec_v2/conv42"),  # P3 80x80 stride=8
            ("goec_v2/conv52", "goec_v2/conv53"),  # P4 40x40 stride=16
            ("goec_v2/conv62", "goec_v2/conv63"),  # P5 20x20 stride=32
        ]
        strides = [8, 16, 32]
        reg_max = 16  # 64 / 4 coords = 16 bins

        all_boxes, all_scores, all_cls_ids = [], [], []

        for (reg_key, cls_key), stride in zip(head_pairs, strides):
            # Shape: (1, H, W, C) — remove batch dim
            reg = raw_output[reg_key][0]  # (H, W, 64)
            cls = raw_output[cls_key][0]  # (H, W, 4)

            grid_h, grid_w = reg.shape[0], reg.shape[1]
            N = grid_h * grid_w

            reg = reg.reshape(N, 4, reg_max)   # (N, 4, 16)
            cls = cls.reshape(N, -1)            # (N, 4)

            # DFL decode: softmax over 16 bins → weighted sum → ltrb distances
            reg = reg - reg.max(axis=2, keepdims=True)
            exp = np.exp(reg)
            softmax = exp / exp.sum(axis=2, keepdims=True)
            bins = np.arange(reg_max, dtype=np.float32)
            ltrb = (softmax * bins).sum(axis=2)  # (N, 4): left, top, right, bottom

            # Build anchor grid (cell centre coords in pixels)
            ys, xs = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
            cx = (xs.ravel() + 0.5) * stride   # (N,)
            cy = (ys.ravel() + 0.5) * stride   # (N,)

            # Convert ltrb → xyxy pixel coords
            x1 = cx - ltrb[:, 0] * stride
            y1 = cy - ltrb[:, 1] * stride
            x2 = cx + ltrb[:, 2] * stride
            y2 = cy + ltrb[:, 3] * stride

            # Sigmoid scores
            cls_scores = 1.0 / (1.0 + np.exp(-cls))       # (N, num_classes)
            best_cls = cls_scores.argmax(axis=1)            # (N,)
            best_score = cls_scores[np.arange(N), best_cls] # (N,)

            # Filter by confidence threshold
            mask = best_score >= confidence_threshold
            if classes is not None:
                mask &= np.isin(best_cls, classes)

            if mask.any():
                all_boxes.append(np.stack([x1, y1, x2, y2], axis=1)[mask])
                all_scores.append(best_score[mask])
                all_cls_ids.append(best_cls[mask])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        cls_ids = np.concatenate(all_cls_ids)

        print(f"[HAILO DEBUG] pre-NMS: {len(boxes)} detections")

        # Scale boxes from model input space to original image space
        sx = orig_w / self.input_width
        sy = orig_h / self.input_height

        detections: List[Detection] = []
        for idx in self._nms(boxes, scores, cls_ids, iou_threshold):
            x1, y1, x2, y2 = boxes[idx]
            cls_id = int(cls_ids[idx])
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id),
                    confidence=round(float(scores[idx]), 6),
                    bbox=BoundingBox(
                        x1=float(x1 * sx),
                        y1=float(y1 * sy),
                        x2=float(x2 * sx),
                        y2=float(y2 * sy),
                    ),
                )
            )

        print(f"[HAILO DEBUG] post-NMS: {len(detections)} detections")
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_threshold: float) -> List[int]:
        """Per-class greedy NMS. Returns list of kept indices."""
        kept = []
        for cls_id in np.unique(cls_ids):
            idx = np.where(cls_ids == cls_id)[0]
            b = boxes[idx]
            s = scores[idx]
            order = s.argsort()[::-1]
            while len(order):
                i = order[0]
                kept.append(idx[i])
                if len(order) == 1:
                    break
                rest = order[1:]
                xx1 = np.maximum(b[i, 0], b[rest, 0])
                yy1 = np.maximum(b[i, 1], b[rest, 1])
                xx2 = np.minimum(b[i, 2], b[rest, 2])
                yy2 = np.minimum(b[i, 3], b[rest, 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
                area_r = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
                iou = inter / (area_i + area_r - inter + 1e-6)
                order = rest[iou < iou_threshold]
        return kept

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
