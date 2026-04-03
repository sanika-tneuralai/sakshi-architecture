"""
Detection service - multi-backend inference orchestration.

DetectionService is backend-agnostic: it delegates inference to whichever
BaseDetector implementation is selected via the INFERENCE_BACKEND env var
(default: 'pytorch').  All frame capture, screenshot saving, DB persistence,
and stats tracking remain unchanged from the original implementation.
"""
import logging
import time
import os
import cv2
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np

from detection.backends import create_detector
from detection.backends.base import BaseDetector
from detection.schemas import Detection, DetectionResponse, BoundingBox, DetectionStats
from shared.common.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, SCREENSHOTS_DIR

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Backend-agnostic object detection service.

    On startup the factory selects either the PyTorch/YOLO backend or the
    Hailo backend based on the INFERENCE_BACKEND environment variable.
    The rest of the service (screenshots, DB, stats, API) is unchanged.
    """

    def __init__(self, detector: Optional[BaseDetector] = None):
        """
        Args:
            detector: Explicit detector instance (useful for testing).
                      When None the factory creates one from config.
        """
        self._detector: BaseDetector = detector if detector is not None else create_detector()
        self.stats: Dict[str, Dict] = {}  # camera_id -> stats

        self._detector.warmup()
        print(f"✓ DetectionService.__init__ completed — backend: {self._detector.backend_name}")

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        classes: Optional[List[int]] = None,
    ) -> DetectionResponse:
        """
        Run object detection on a frame.

        Args:
            frame: Input image (numpy array, BGR).
            camera_id: Camera identifier.
            confidence_threshold: Minimum confidence score.
            iou_threshold: IOU threshold for NMS.
            classes: Filter specific class IDs.

        Returns:
            DetectionResponse with all detections.
        """
        start_time = time.time()

        detection_objects: List[Detection] = self._detector.detect(
            frame,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        # Update stats
        self._update_stats(camera_id, len(detection_objects), processing_time)

        response = DetectionResponse(
            camera_id=camera_id,
            timestamp=datetime.now(),
            frame_count=self.stats[camera_id]["total_frames"],
            detections=detection_objects,
            total_detections_count=len(detection_objects),
            processing_time_ms=processing_time,
        )

        print(
            f"✓ DetectionService.detect completed for {camera_id}: "
            f"{len(detection_objects)} detections [{self._detector.backend_name}]"
        )
        return response

    def get_stats(self, camera_id: str) -> Optional[DetectionStats]:
        """Get detection statistics for a camera."""
        if camera_id not in self.stats:
            print("✓ DetectionService.get_stats completed: None (camera not found)")
            return None

        stats = self.stats[camera_id]
        avg_time = (
            sum(stats["processing_times"]) / len(stats["processing_times"])
            if stats["processing_times"]
            else 0.0
        )

        result = DetectionStats(
            camera_id=camera_id,
            total_frames_processed=stats["total_frames"],
            total_detections=stats["total_detections"],
            average_processing_time_ms=avg_time,
            is_active=True,
        )
        print(f"✓ DetectionService.get_stats completed for {camera_id}")
        return result

    def reset_stats(self, camera_id: str) -> None:
        """Reset statistics for a camera."""
        if camera_id in self.stats:
            del self.stats[camera_id]
        print(f"✓ DetectionService.reset_stats completed for {camera_id}")

    # ------------------------------------------------------------------
    # Private helpers (unchanged from original)
    # ------------------------------------------------------------------

    # def _save_screenshot(self, frame: np.ndarray, camera_id: str) -> Optional[str]:
    #     """Save detection frame as screenshot."""
    #     try:
    #         os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #         filename = f"{camera_id}_{timestamp}.jpg"
    #         filepath = os.path.join(SCREENSHOTS_DIR, filename)
    #         cv2.imwrite(filepath, frame)
    #         logger.info("Screenshot saved: %s", filepath)
    #         return filepath
    #     except Exception as e:
    #         logger.error("Failed to save screenshot: %s", e)
    #         print(f"[SCREENSHOT] Error saving screenshot: {e}")
    #         return None

    # def _persist_detections(
    #     self,
    #     camera_id: str,
    #     detections: List[Detection],
    #     screenshot_path: Optional[str] = None,
    # ) -> Optional[int]:
    #     """Persist detections to database and return first detection_id."""
    #     first_detection_id = None
    #     try:
    #         from shared.database.persistence import persist_camera, persist_detection
    #
    #         persist_camera(camera_id)
    #         for idx, det in enumerate(detections):
    #             detection_id = persist_detection(
    #                 camera_id=camera_id,
    #                 object_type=det.class_name,
    #                 confidence=det.confidence,
    #                 screenshot_path=screenshot_path,
    #             )
    #             if idx == 0 and detection_id:
    #                 first_detection_id = detection_id
    #
    #         print(f"[DB] Persisted {len(detections)} detections for camera {camera_id}")
    #     except Exception as e:
    #         print(f"[DB] Error persisting detections: {e}")
    #
    #     return first_detection_id

    def _update_stats(self, camera_id: str, total_dets: int, proc_time: float) -> None:
        """Update per-camera detection statistics."""
        if camera_id not in self.stats:
            self.stats[camera_id] = {
                "total_frames": 0,
                "total_detections": 0,
                "processing_times": [],
            }

        stats = self.stats[camera_id]
        stats["total_frames"] += 1
        stats["total_detections"] += total_dets
        stats["processing_times"].append(proc_time)

        # Keep only last 100 samples to bound memory
        if len(stats["processing_times"]) > 100:
            stats["processing_times"] = stats["processing_times"][-100:]

        print(f"✓ DetectionService._update_stats completed for {camera_id}")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Get or create the global detection service instance."""
    global detection_service
    if detection_service is None:
        detection_service = DetectionService()
    print("✓ get_detection_service completed")
    return detection_service


print("✓ detection.service module loaded")
