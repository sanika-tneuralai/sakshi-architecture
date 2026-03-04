from pydantic import BaseModel, Field
from typing import Optional, List


class RTSPConfig(BaseModel):
    """Configuration for single RTSP camera stream"""
    rtsp_url: str = Field(..., description="RTSP stream URL")
    camera_id: str = Field(..., description="Unique identifier for the camera")
    fps: Optional[int] = Field(default=5, description="Frame extraction rate", ge=1, le=30)
    roi_points: Optional[List[List[float]]] = Field(
        default=None, 
        description="Region of Interest as polygon points [[x1,y1], [x2,y2], ...]"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
                "camera_id": "queue_cam_1",
                "fps": 5,
                "roi_points": [[1055.55, 536.47], [951.55, 452.47], [1101.55, 347.47], [1228.55, 413.47]]
            }
        }


class MultiStreamConfig(BaseModel):
    """Configuration for multiple camera streams in batch mode"""
    streams: List[RTSPConfig] = Field(..., description="List of camera streams", min_length=1)
    batch_size: Optional[int] = Field(default=30, description="Batch size for processing", ge=1, le=128)
    width: Optional[int] = Field(default=1920, description="Stream width", ge=640)
    height: Optional[int] = Field(default=1080, description="Stream height", ge=480)
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_size": 30,
                "width": 1920,
                "height": 1080,
                "streams": [
                    {
                        "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
                        "camera_id": "queue_cam_1",
                        "fps": 5,
                        "roi_points": [[1055.55, 536.47], [951.55, 452.47], [1101.55, 347.47], [1228.55, 413.47]]
                    }
                ]
            }
        }


class CameraStatus(BaseModel):
    """Camera status response"""
    camera_id: str
    is_running: bool
    backend: str
    fps: Optional[int] = None
    frame_count: Optional[int] = None
    last_frame_time: Optional[float] = None
    rtsp_url: Optional[str] = None


class FrameResponse(BaseModel):
    """Frame data response"""
    camera_id: str
    timestamp: str
    shape: Optional[List[int]] = None
    roi_points: Optional[List[List[float]]] = None
    frame_count: int
    status: str
    backend: str