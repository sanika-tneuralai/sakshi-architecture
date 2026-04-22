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
# {0: "car", 1: "gun", 2: "no_gun"}
# ---------------------------------------------------------------------------
_DEFAULT_CLASSES = ["car", "gun", "no_gun"]


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
        input_info = hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        fmt = input_info.format
        print(f"[HAILO DEBUG] input stream: name={input_info.name}, shape={input_info.shape}, format type={fmt.type}, format order={fmt.order}")

        # Discover the NMS postprocess output key dynamically
        output_infos = hef.get_output_vstream_infos()
        output_names = [info.name for info in output_infos]
        logger.info("HEF output stream names: %s", output_names)
        print(f"✓ HEF output streams: {output_names}")

        # Prefer a name containing 'nms_postprocess', fall back to single output
        nms_keys = [n for n in output_names if "nms_postprocess" in n]
        if nms_keys:
            self._output_key = nms_keys[0]
        elif len(output_names) == 1:
            self._output_key = output_names[0]
        else:
            raise RuntimeError(
                f"Cannot determine NMS postprocess output key. "
                f"Available outputs: {output_names}. "
                f"Expected one containing 'nms_postprocess'."
            )
        logger.info("Using output key: %s", self._output_key)
        print(f"✓ HailoDetector using output key: {self._output_key}")

        logger.info("HEF loaded: %s", self.hef_path)
        print(f"✓ HailoDetector._load completed: {self.hef_path}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize to model input size, convert BGR→RGB, return uint8.
        Hailo NMS-baked models handle normalisation internally.
        """
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result = rgb.astype(np.uint8)
        print(f"[HAILO DEBUG] preprocess: input shape={frame.shape} dtype={frame.dtype}, output shape={result.shape} dtype={result.dtype} min={result.min()} max={result.max()}")
        return result

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

        Output key is auto-discovered at load time (self._output_key).
        Shape: (num_classes, N, 5) — each row: [y1, x1, y2, x2, score] normalised 0-1.
        """
        orig_h, orig_w = original_shape[:2]

        key = self._output_key
        # Hailo NMS output: list of per-class detection arrays
        # raw_output[key] is a list (batch), [0] gives one list per class
        # Each class entry is an ndarray of shape (N, 5): [y1, x1, y2, x2, score]
        raw_value = raw_output[key]
        print(f"[HAILO DEBUG] raw_output keys: {list(raw_output.keys())}")
        print(f"[HAILO DEBUG] raw_output[key] type: {type(raw_value)}, len: {len(raw_value) if hasattr(raw_value, '__len__') else 'N/A'}")
        data = raw_value[0]  # list of length num_classes
        print(f"[HAILO DEBUG] data type: {type(data)}, len: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        for i, cls_dets in enumerate(data):
            print(f"[HAILO DEBUG] class {i} ({self.class_names[i] if i < len(self.class_names) else i}): type={type(cls_dets)}, shape={getattr(cls_dets, 'shape', None)}, len={len(cls_dets) if hasattr(cls_dets, '__len__') else 'N/A'}")
            if hasattr(cls_dets, '__len__') and len(cls_dets) > 0:
                import numpy as np
                arr = np.array(cls_dets)
                print(f"[HAILO DEBUG]   raw values: {arr}")

        detections: List[Detection] = []

        for cls_id, cls_dets in enumerate(data):
            if classes is not None and cls_id not in classes:
                continue

            if cls_dets is None or len(cls_dets) == 0:
                continue

            import numpy as np
            cls_dets = np.array(cls_dets)  # ensure ndarray (N, 5)

            for det in cls_dets:
                y1_n, x1_n, y2_n, x2_n, score = det
                score = float(score)
                print(f"[HAILO DEBUG] cls={cls_id} score={score:.4f} box=[{y1_n:.3f},{x1_n:.3f},{y2_n:.3f},{x2_n:.3f}] thresh={confidence_threshold}")

                if score < confidence_threshold:
                    continue
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
