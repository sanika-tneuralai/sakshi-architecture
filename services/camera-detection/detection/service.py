"""
Detection service - YOLO model integration.
"""
import logging
import time
import os
import cv2
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np

from detection.device import select_device
from detection.schemas import Detection, DetectionResponse, BoundingBox, DetectionStats
from shared.common.config import YOLO_MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, SCREENSHOTS_DIR

logger = logging.getLogger(__name__)


class DetectionService:
    """
    YOLO-based object detection service.
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, device: Optional[str] = None):
        """
        Initialize detection service.

        Args:
            model_path: Path to YOLO model file
            device: Device to use ("cuda" or "cpu"), auto-detect if None
        """
        self.model_path = model_path
        self.device = device if device else select_device()
        self.model = None
        self.stats: Dict[str, Dict] = {}  # camera_id -> stats

        self._load_model()
        print(f"✓ DetectionService.__init__ completed on {self.device}")

    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded from {self.model_path} on {self.device}")
            print(f"✓ DetectionService._load_model completed: {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _save_screenshot(self, frame: np.ndarray, camera_id: str) -> Optional[str]:
        """
        Save detection frame as screenshot.

        Args:
            frame: Frame to save
            camera_id: Camera identifier

        Returns:
            Screenshot file path or None if save failed
        """
        try:
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_{timestamp}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)

            cv2.imwrite(filepath, frame)

            logger.info(f"Screenshot saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            print(f"[SCREENSHOT] Error saving screenshot: {str(e)}")
            return None

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        classes: Optional[List[int]] = None
    ) -> DetectionResponse:
        """
        Run object detection on a frame.

        Args:
            frame: Input image (numpy array)
            camera_id: Camera identifier
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            classes: Filter specific class IDs

        Returns:
            DetectionResponse with all detections
        """
        start_time = time.time()

        # Run YOLO inference
        results = self.model.predict(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False
        )[0]

        # Parse detections
        detection_objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]

            detection_objects.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            ))

        processing_time = (time.time() - start_time) * 1000  # ms

        # Save screenshot if detections found
        screenshot_path = None
        if len(detection_objects) > 0:
            screenshot_path = self._save_screenshot(frame, camera_id)
            print(f"[SCREENSHOT] Screenshot saved: {screenshot_path}")

        # Persist to database and get first detection_id
        first_detection_id = self._persist_detections(camera_id, detection_objects, screenshot_path)

        # Update stats
        self._update_stats(camera_id, len(detection_objects), processing_time)

        response = DetectionResponse(
            camera_id=camera_id,
            timestamp=datetime.now(),
            frame_count=self.stats[camera_id]['total_frames'],
            detections=detection_objects,
            total_detections_count=len(detection_objects),
            processing_time_ms=processing_time,
            first_detection_id=first_detection_id,
            screenshot_path=screenshot_path
        )

        print(f"✓ DetectionService.detect completed for {camera_id}: {len(detection_objects)} detections")
        return response

    def _persist_detections(self, camera_id: str, detections: List[Detection], screenshot_path: Optional[str] = None) -> Optional[int]:
        """Persist detections to database and return first detection_id"""
        first_detection_id = None
        try:
            from shared.database.persistence import persist_camera, persist_detection

            persist_camera(camera_id)

            for idx, det in enumerate(detections):
                detection_id = persist_detection(
                    camera_id=camera_id,
                    object_type=det.class_name,
                    confidence=det.confidence,
                    screenshot_path=screenshot_path
                )
                if idx == 0 and detection_id:
                    first_detection_id = detection_id

            print(f"[DB] Persisted {len(detections)} detections for camera {camera_id}")
        except Exception as e:
            print(f"[DB] Error persisting detections: {str(e)}")

        return first_detection_id

    def _update_stats(self, camera_id: str, total_dets: int, proc_time: float):
        """Update detection statistics"""
        if camera_id not in self.stats:
            self.stats[camera_id] = {
                'total_frames': 0,
                'total_detections': 0,
                'processing_times': []
            }

        stats = self.stats[camera_id]
        stats['total_frames'] += 1
        stats['total_detections'] += total_dets
        stats['processing_times'].append(proc_time)

        if len(stats['processing_times']) > 100:
            stats['processing_times'] = stats['processing_times'][-100:]

        print(f"✓ DetectionService._update_stats completed for {camera_id}")

    def get_stats(self, camera_id: str) -> Optional[DetectionStats]:
        """Get detection statistics for a camera"""
        if camera_id not in self.stats:
            print(f"✓ DetectionService.get_stats completed: None (camera not found)")
            return None

        stats = self.stats[camera_id]
        avg_time = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0.0

        result = DetectionStats(
            camera_id=camera_id,
            total_frames_processed=stats['total_frames'],
            total_detections=stats['total_detections'],
            average_processing_time_ms=avg_time,
            is_active=True
        )

        print(f"✓ DetectionService.get_stats completed for {camera_id}")
        return result

    def reset_stats(self, camera_id: str):
        """Reset statistics for a camera"""
        if camera_id in self.stats:
            del self.stats[camera_id]
        print(f"✓ DetectionService.reset_stats completed for {camera_id}")


# Global detection service instance
detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Get or create the global detection service instance"""
    global detection_service
    if detection_service is None:
        detection_service = DetectionService()
    print("✓ get_detection_service completed")
    return detection_service


print("✓ detection.service module loaded")
