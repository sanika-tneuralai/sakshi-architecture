"""
Pydantic schemas for detection module.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    in_roi: bool = Field(..., description="Whether detection is inside ROI")


class DetectionRequest(BaseModel):
    """Request to run detection on a camera stream"""
    camera_id: str = Field(..., description="Camera ID to detect from")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IOU threshold for NMS")
    classes: Optional[List[int]] = Field(None, description="Filter specific class IDs (None = all classes)")


class DetectionResponse(BaseModel):
    """Detection results"""
    camera_id: str
    timestamp: datetime
    frame_count: int
    detections: List[Detection]
    roi_detections_count: int = Field(..., description="Number of detections inside ROI")
    total_detections_count: int = Field(..., description="Total number of detections")
    processing_time_ms: float = Field(..., description="Detection processing time in milliseconds")
    first_detection_id: Optional[int] = Field(None, description="ID of first persisted detection")
    screenshot_path: Optional[str] = Field(None, description="Path to detection screenshot")


class DetectionStats(BaseModel):
    """Detection statistics"""
    camera_id: str
    total_frames_processed: int
    total_detections: int
    roi_detections: int
    average_processing_time_ms: float
    is_active: bool


print("✓ detection.schemas loaded")
