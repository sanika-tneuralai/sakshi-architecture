"""
Pydantic schemas for usecase module.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class UsecaseRequest(BaseModel):
    """Request to evaluate multiple usecases"""
    camera_id: str = Field(..., description="Camera ID")
    detection_output: dict = Field(..., description="Detection API output")
    usecases: List[str] = Field(..., description="List of usecase IDs to evaluate")


class UsecaseResult(BaseModel):
    """Result for a single usecase evaluation"""
    usecase_id: str = Field(..., description="Usecase identifier")
    triggered: bool = Field(..., description="Whether usecase was triggered")
    matched_count: int = Field(..., description="Number of matched objects")
    matched_objects: List[dict] = Field(default_factory=list, description="Objects that matched the rule")
    detection_id: Optional[int] = Field(None, description="ID of associated detection")
    screenshot_path: Optional[str] = Field(None, description="Path to detection screenshot")


class UsecaseResponse(BaseModel):
    """Response containing results for all evaluated usecases"""
    camera_id: str = Field(..., description="Camera ID")
    results: List[UsecaseResult] = Field(..., description="Results for each usecase")
