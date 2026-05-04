"""
Pydantic schemas for detection module.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Detection(BaseModel):
    """Single object detection result"""
    class_id: int = Field(..., description="Class ID from model")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionRequest(BaseModel):
    """Request to run detection on a camera stream"""
    camera_id: str = Field(..., description="Camera ID to detect from")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold (base when class_thresholds not provided)")
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IOU threshold for NMS")
    classes: Optional[List[int]] = Field(None, description="Filter specific class IDs (None = all classes)")
    class_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="Per-class confidence thresholds e.g. {\"gun\": 0.3, \"fire\": 0.4, \"smoke\": 0.35, \"car\": 0.6}. "
                    "YOLO runs at min(values); results are post-filtered per class. "
                    "Classes not listed fall back to confidence_threshold."
    )
    logo_rois: Optional[Dict[str, List[List[int]]]] = Field(
        None,
        description="Per-ROI polygons of the painted G-logo region inside each parking slot. "
                    "Used as a YOLO-independent occupancy fallback: if a car covers the logo, "
                    "occupancy fires even when YOLO misses the car (bad parking angle, occlusion). "
                    "Format: {\"ROI_1\": [[x,y],...], \"ROI_2\": [[x,y],...]}. Coords in original "
                    "1920x1080 frame space."
    )


class DetectionResponse(BaseModel):
    """Detection results"""
    camera_id: str
    timestamp: datetime
    frame_count: int
    detections: List[Detection]
    total_detections_count: int = Field(..., description="Total number of detections")
    processing_time_ms: float = Field(..., description="Detection processing time in milliseconds")
    snapshot_url: Optional[str] = Field(None, description="S3 HTTPS URL of the frame snapshot (only when detections > 0 OR any logo is occluded)")
    logo_occluded: Optional[Dict[str, bool]] = Field(
        None,
        description="Per-ROI boolean: True if the painted G-logo region is occluded vs. the empty reference. "
                    "Acts as a YOLO-independent occupancy signal. None when the request did not supply logo_rois."
    )


class DetectionStats(BaseModel):
    """Detection statistics"""
    camera_id: str
    total_frames_processed: int
    total_detections: int
    average_processing_time_ms: float
    is_active: bool


class DetectBatchRequest(BaseModel):
    """Request to run detection on multiple cameras"""
    camera_ids: Optional[List[str]] = Field(None, description="List of camera IDs to detect from (None = all active cameras)")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IOU threshold for NMS")
    classes: Optional[List[int]] = Field(None, description="Filter specific class IDs (None = all classes)")
    logo_rois_by_camera: Optional[Dict[str, Dict[str, List[List[int]]]]] = Field(
        None,
        description="Per-camera logo polygons. Outer key=camera_id, inner=DetectionRequest.logo_rois. "
                    "Optional — cameras without a mapping skip the logo-occlusion check."
    )


class CameraDetectionResult(BaseModel):
    """Detection result for a single camera in batch operation"""
    camera_id: str
    status: str = Field(..., description="success or failed")
    total_detections_count: int
    processing_time_ms: float
    detections: List[Detection]
    error: Optional[str] = None
    snapshot_url: Optional[str] = Field(None, description="S3 HTTPS URL of the frame snapshot (only when detections > 0 OR any logo is occluded)")
    logo_occluded: Optional[Dict[str, bool]] = Field(
        None,
        description="Per-ROI G-logo occlusion booleans. None when no logo polygons were supplied for this camera."
    )


class DetectBatchResponse(BaseModel):
    """Batch detection results for multiple cameras"""
    status: str
    total_cameras: int
    successful: int
    failed: int
    results: List[CameraDetectionResult]


print("✓ detection.schemas loaded")
